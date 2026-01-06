import numpy as np
import matplotlib.pyplot as plt

#_____________________________
# parameters
#_____________________________

N = 500
w_inhib = -1.0
tau = 1.0
g = 3.0
dt = 0.05
R = np.pi / 8
h_base = 0.0
h_vec_base = np.full(N, h_base)

seed_init = 0
seed_off  = 1


ON_WINDOW  = 100
OFF_WINDOW = 100


ZERO_THR = 0.12


REL_TOL_ON  = 0.02
REL_TOL_OFF = 0.50

MAX_STEPS_ON  = 50000
MAX_STEPS_OFF = 50000

h_on = 2.0
noise_shift = -2.0

#_____________________________
# connectivity construction
#_____________________________

x = np.linspace(0, 2*np.pi, N, endpoint=False)
dx = np.abs(x[:, None] - x[None, :])
dist = np.minimum(dx, 2*np.pi - dx)
H = (dist > R).astype(float)
W = w_inhib * H
np.fill_diagonal(W, 0.0)

#_____________________________________________________
# One-step Euler integration
#_____________________________________________________

def euler_step(s, h_vec):
    if np.isscalar(h_vec):
        h_vec = np.full(N, float(h_vec))
    u = W @ s + h_vec
    r = np.maximum(u, 0.0)
    dsdt = (-s + g * r) / tau
    s = s + dt * dsdt
    return np.maximum(s, 0.0)

#_____________________________________________________________
# bump angle
#_____________________________________________________________

def bump_angle(s):
    z = np.sum(s * np.exp(1j * x))
    if np.abs(z) < 1e-12:
        return np.nan
    ang = np.angle(z)
    return ang + 2*np.pi if ang < 0 else ang

#________________
# init & pre-run
#________________

def init_and_stabilize(pre_steps=6000):
   
    rng_init = np.random.default_rng(seed_init)
    c0 = rng_init.uniform(0, 2*np.pi)
    d0 = np.minimum(np.abs(x - c0), 2*np.pi - np.abs(x - c0))
    s = np.exp(-(d0**2) / (2*(0.25**2)))
    s /= (s.max() + 1e-12)

    for _ in range(pre_steps):
        s = euler_step(s, h_vec_base)
    return s

#___________________________________________
# Task 2: minimal ON duration
#___________________________________________

def on_condition_stable(max_hist, eps_nonzero=1e-2, rel_tol=REL_TOL_ON):
    
    m = float(np.mean(max_hist))
    if m < eps_nonzero:
        return False
    rel_var = float(np.std(max_hist) / (m + 1e-12))
    return rel_var < rel_tol

def find_on_steps(s0, h_on=h_on, max_steps=MAX_STEPS_ON, window=ON_WINDOW):
    s = s0.copy()
    max_buf = []

    for t in range(1, max_steps + 1):
        s = euler_step(s, h_on)
        max_buf.append(float(np.max(s)))

        if len(max_buf) > window:
            max_buf.pop(0)

        if len(max_buf) == window:
            if on_condition_stable(max_buf):
                return t - window + 1, s

    return None, s

#______________________________________________
# Task 3: minimal OFF duration (hover around zero)
#______________________________________________

def off_condition_zero(max_hist, zero_thr=ZERO_THR, rel_tol=REL_TOL_OFF):
    m = float(np.mean(max_hist))
    if m > zero_thr:
        return False

    denom = max(m, 1e-3)
    rel_var = float(np.std(max_hist) / denom)
    return rel_var < rel_tol

def find_off_steps(
    s0,
    noise_shift=noise_shift,
    max_steps=MAX_STEPS_OFF,
    window=OFF_WINDOW,
    zero_thr=ZERO_THR
):

    rng_off = np.random.default_rng(seed_off)

    s = s0.copy()
    max_buf = []
    max_trace = []
    ang_trace = []

    for t in range(1, max_steps + 1):
        h_off = rng_off.standard_normal(N) + noise_shift
        s = euler_step(s, h_off)

        mx = float(np.max(s))
        max_trace.append(mx)
        ang_trace.append(bump_angle(s))

        max_buf.append(mx)
        if len(max_buf) > window:
            max_buf.pop(0)

        if len(max_buf) == window:
            if off_condition_zero(max_buf, zero_thr=zero_thr):
                off_steps_min = t - window + 1
                mean_max_off = float(np.mean(max_buf))
                return off_steps_min, s, mean_max_off, np.array(max_trace), np.array(ang_trace)

    return None, s, None, np.array(max_trace), np.array(ang_trace)

#—————————————————————————————————————
# Run
#—————————————————————————————————————


s_start = init_and_stabilize(pre_steps=6000)
s_zero = np.zeros(N)

# 1) minimal ON
on_steps_min, s_after_on = find_on_steps(s_zero, h_on=h_on, window=ON_WINDOW)
print("ON: on_steps_min =", on_steps_min)

# 2) minimal OFF
off_steps_min, s_after_off, mean_max_off, max_trace_off, ang_trace_off = find_off_steps(
    s_start,
    noise_shift=noise_shift,
    window=OFF_WINDOW,
    zero_thr=ZERO_THR
)
print("OFF: off_steps_min =", off_steps_min)
print("    mean max(s) over OFF window =", mean_max_off)

#—————————————————————————————————
# Visual proof
#—————————————————————————————————


if on_steps_min is not None:
    s = s_zero.copy()
    max_on = []
    for _ in range(on_steps_min + ON_WINDOW):  
        s = euler_step(s, h_on)
        max_on.append(float(np.max(s)))

    t_on = np.arange(len(max_on)) 
    plt.figure(figsize=(10, 3))
    plt.plot(t_on, max_on, lw=1.5)
    plt.axvline(on_steps_min, ls="--", lw=1.2)
    plt.xlabel("ON time (steps, start from zero activity)")
    plt.ylabel("max(s)")
    plt.title("Minimal ON duration: max(s) reaches a non-zero stable plateau")
    plt.tight_layout()
    plt.show()


if off_steps_min is not None:

    t_off = np.arange(len(max_trace_off))  # OFF steps from 0

    plt.figure(figsize=(10, 3))
    plt.plot(t_off, max_trace_off, lw=1.5)
    plt.axvline(off_steps_min, ls="--", lw=1.2)
    plt.axhline(ZERO_THR, ls=":", lw=1.2)
    plt.xlabel("OFF time (steps, t=0 at OFF switch)")
    plt.ylabel("max(s)")
    plt.title("Minimal OFF duration: max(s) first hovers around zero")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 3))
    plt.plot(t_off, ang_trace_off, ".", ms=2)
    plt.axvline(off_steps_min, ls="--", lw=1.2)
    plt.xlabel("OFF time (steps, t=0 at OFF switch)")
    plt.ylabel("bump angle (rad)")
    plt.title("OFF period: bump angle vs time (same OFF time axis)")
    plt.tight_layout()
    plt.show()
