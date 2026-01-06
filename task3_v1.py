import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from ripser import ripser
from persim import plot_diagrams


#_____________________________
# Parameters 
#_____________________________

seed = 0
rng = np.random.default_rng(seed)
# Network size
N = 500
N_side = int(np.sqrt(N))
N2 = N_side * N_side
print("2D sheet: N_side =", N_side, "=> N =", N2)
# Dynamics parameters
dt = 1.0
tau = 10.0
g = 1.0
I_const = 3.0
alpha = 0.0
# Connectivity parameters
w_inhib = -1.0
l_offset = 2.0
R_grid = 6
print("R_grid =", R_grid)
# External field parameters
field_steps = 100
n_cycles = 50
h_on_amp = 2.0
h_off_sigma = 1.0
h_off_shift = -2.0
sigma_bump = 2.0
# Recording and TDA
DO_PLOT = True
m_active = 1200

#______________________________________
# Direction preference θ_i
#______________________________________


def tile_preference(N_side):
    base = np.array([[0.0, 0.5*np.pi],
                     [1.0*np.pi, 1.5*np.pi]])
    reps = (N_side // 2, N_side // 2)
    return np.tile(base, reps)

theta_pref = tile_preference(N_side)

group = np.zeros_like(theta_pref, dtype=np.int8)
group[np.isclose(theta_pref, 0.0)] = 0
group[np.isclose(theta_pref, 0.5*np.pi)] = 1
group[np.isclose(theta_pref, 1.0*np.pi)] = 2
group[np.isclose(theta_pref, 1.5*np.pi)] = 3


#_________________________________
# Shifted-disk inhibitory kernel
#_________________________________


def build_shifted_disk_kernel(N_side, R_grid, W0, l, theta):
    dx = np.arange(N_side)
    dy = np.arange(N_side)
    dx = np.where(dx <= N_side//2, dx, dx - N_side)
    dy = np.where(dy <= N_side//2, dy, dy - N_side)
    DX, DY = np.meshgrid(dx, dy, indexing="ij")

    sx = l * np.cos(theta)
    sy = l * np.sin(theta)
    dist = np.sqrt((DX - sx)**2 + (DY - sy)**2)

    K = np.where(dist <= R_grid, W0, 0.0).astype(np.float32)
    return K

def periodic_conv_fft(s_grid, K_fft):
    return np.fft.ifft2(np.fft.fft2(s_grid) * K_fft).real

K_tmp = build_shifted_disk_kernel(N_side, R_grid, 1.0, l_offset, 0.0)
disk_area = float(np.sum(K_tmp != 0.0))
W0 = float(w_inhib) / max(1.0, disk_area)
print("disk_area =", int(disk_area), "=> normalized W0 =", W0)

thetas = [0.0, 0.5*np.pi, 1.0*np.pi, 1.5*np.pi]
K_ffts = []
for th in thetas:
    K = build_shifted_disk_kernel(N_side, R_grid, W0, l_offset, th)
    K_ffts.append(np.fft.fft2(K))


#__________________________________________
# Local Gaussian external field (ON phase)
#__________________________________________


def gaussian_bump_field(N_side, cx, cy, amp, sigma):
    xs = np.arange(N_side)
    ys = np.arange(N_side)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    dx = np.minimum(np.abs(X - cx), N_side - np.abs(X - cx))
    dy = np.minimum(np.abs(Y - cy), N_side - np.abs(Y - cy))
    d2 = dx*dx + dy*dy

    return (amp * np.exp(-d2 / (2.0 * sigma * sigma))).astype(np.float32)


#_____________________________
# Euler update step
#_____________________________


def euler_step_2d(s, h_field):
    if np.isscalar(h_field):
        h = np.full_like(s, float(h_field), dtype=np.float32)
    else:
        h = h_field.astype(np.float32)

    rec0 = periodic_conv_fft(s, K_ffts[0])
    rec1 = periodic_conv_fft(s, K_ffts[1])
    rec2 = periodic_conv_fft(s, K_ffts[2])
    rec3 = periodic_conv_fft(s, K_ffts[3])

    rec = np.zeros_like(s, dtype=np.float32)
    rec[group == 0] = rec0[group == 0]
    rec[group == 1] = rec1[group == 1]
    rec[group == 2] = rec2[group == 2]
    rec[group == 3] = rec3[group == 3]

    u = rec + I_const + h
    r = np.maximum(u, 0.0)
    s = s + (dt / tau) * (-s + g * r)
    s[s < 0] = 0.0
    return s.astype(np.float32)


#_____________________________
# Phase estimation
#_____________________________


xs = np.arange(N_side)
ys = np.arange(N_side)
ex = np.exp(1j * 2*np.pi * xs / N_side)
ey = np.exp(1j * 2*np.pi * ys / N_side)

def phase_xy(s_flat):
    s = s_flat.reshape(N_side, N_side)
    sx = np.sum(s, axis=1)
    sy = np.sum(s, axis=0)

    zx = np.sum(sx * ex)
    zy = np.sum(sy * ey)

    phix = np.angle(zx)
    phiy = np.angle(zy)

    if phix < 0:
        phix += 2*np.pi
    if phiy < 0:
        phiy += 2*np.pi
    return float(phix), float(phiy)


#________________________________________
# Stratified Active sampling over cycles
#________________________________________


def select_active_2d_stratified(X, cycle_ids, m, rng_local, pool_mult=8, n_bins=8):
    cycles = np.unique(cycle_ids)
    nC = len(cycles)
    m_per = max(1, m // max(1, nC))

    picked = []
    for c in cycles:
        idx_c = np.where(cycle_ids == c)[0]
        if len(idx_c) == 0:
            continue

        Xc = X[idx_c]
        X_mean = np.mean(Xc, axis=1)
        order = np.argsort(X_mean)[::-1]

        pool_size = min(pool_mult * m_per, len(idx_c))
        idx_pool = idx_c[order[:pool_size]]

        phases = np.array([phase_xy(X[i]) for i in idx_pool], dtype=float)
        phix = phases[:, 0]
        phiy = phases[:, 1]

        bins = np.linspace(0.0, 2*np.pi, n_bins + 1)
        k_per_cell = max(1, m_per // (n_bins * n_bins))

        sel_c = []
        for ix in range(n_bins):
            for iy in range(n_bins):
                mask = (phix >= bins[ix]) & (phix < bins[ix+1]) & \
                       (phiy >= bins[iy]) & (phiy < bins[iy+1])
                ids = idx_pool[mask]
                if len(ids) == 0:
                    continue
                ids_sorted = ids[np.argsort(np.mean(X[ids], axis=1))[::-1]]
                sel_c.extend(ids_sorted[:k_per_cell])

        sel_c = np.array(sel_c, dtype=int)
        if len(sel_c) >= m_per:
            sel_c = sel_c[:m_per]
        else:
            rest = np.setdiff1d(idx_pool, sel_c, assume_unique=False)
            need = m_per - len(sel_c)
            if need > 0 and len(rest) > 0:
                extra = rng_local.choice(rest, size=min(need, len(rest)), replace=False)
                sel_c = np.concatenate([sel_c, extra])

        picked.extend(sel_c.tolist())

    picked = np.array(picked, dtype=int)
    if len(picked) < m:
        rest_all = np.setdiff1d(np.arange(X.shape[0]), picked, assume_unique=False)
        need = m - len(picked)
        if need > 0 and len(rest_all) > 0:
            extra = rng_local.choice(rest_all, size=min(need, len(rest_all)), replace=False)
            picked = np.concatenate([picked, extra])
    if len(picked) > m:
        picked = picked[:m]
    return picked


#_____________________________
# Initialization and pre-run
#_____________________________


s = np.zeros((N_side, N_side), dtype=np.float32)
n_on_init = max(1, int(0.01 * N2))
idx0 = rng.choice(N2, size=n_on_init, replace=False)
s.flat[idx0] = 1.0

pre_steps = 6000
for _ in range(pre_steps):
    s = euler_step_2d(s, 0.0)
print("Pre-run done.")


#_____________________________
# Main ON/OFF cycles
#_____________________________


X_keep = []
cycle_ids = []
centers = []

save_from = 0

for cyc in range(n_cycles):
    for t in range(field_steps):
        h_off = (rng.normal(0.0, h_off_sigma, size=(N_side, N_side)) + h_off_shift).astype(np.float32)
        s = euler_step_2d(s, h_off)
        if cyc >= save_from:
            X_keep.append(s.reshape(-1).copy())
            cycle_ids.append(cyc)

    cx = int(rng.integers(0, N_side))
    cy = int(rng.integers(0, N_side))
    centers.append((cx, cy))
    h_on = gaussian_bump_field(N_side, cx, cy, h_on_amp, sigma_bump)

    for t in range(field_steps):
        s = euler_step_2d(s, h_on)
        if cyc >= save_from:
            X_keep.append(s.reshape(-1).copy())
            cycle_ids.append(cyc)

X = np.array(X_keep, dtype=np.float32)
print("Saved X shape =", X.shape)


#_____________________________
# Visualization
#_____________________________


if DO_PLOT:
    plt.figure(figsize=(5, 4))
    plt.imshow(s, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title("Final activity snapshot (2D recurrent)")
    plt.tight_layout()
    plt.show()

    cs = centers[-10:]
    xs = [p[0] for p in cs]
    ys = [p[1] for p in cs]
    plt.figure(figsize=(4, 4))
    plt.scatter(xs, ys)
    plt.gca().set_aspect("equal", "box")
    plt.title("Last 10 ON-field centers")
    plt.xlabel("cx")
    plt.ylabel("cy")
    plt.tight_layout()
    plt.show()


#_____________________________
# TDA (Active sampling, NO PCA)
#_____________________________


m = min(m_active, X.shape[0])
rng_tda = np.random.default_rng(0)
idx_act = select_active_2d_stratified(
    X, np.array(cycle_ids, dtype=int),
    m=m, rng_local=rng_tda, pool_mult=6, n_bins=8
)
X_active = X[idx_act]

X_std = StandardScaler().fit_transform(X_active).astype(np.float32)
res = ripser(X_std, maxdim=2)

dgms = res["dgms"]
H1 = dgms[1]
H2 = dgms[2] if len(dgms) > 2 else np.zeros((0, 2))

print("H1 points:", H1.shape)
print("H2 points:", H2.shape)

if DO_PLOT:
    plot_diagrams([H1], labels=["H1"])
    plt.title("Persistence diagram: H1 (Active, NO PCA)")
    plt.show()

    plot_diagrams([H2], labels=["H2"])
    plt.title("Persistence diagram: H2 (Active, NO PCA)")
    plt.show()
