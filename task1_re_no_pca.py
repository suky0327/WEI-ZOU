import numpy as np
import matplotlib.pyplot as plt

#_____________________________
# parameters
#_____________________________
N = 500                               # Number of neurons
w_inhib = -1.0                        # inhibitory weight 
on_steps = 1000         # number of Euler steps in each on  segment
off_steps = 300         # number of Euler steps in each off segment
n_cycles = 100           # number of cycles
last_k_cycles = 200       # how many last cycles to save into X_samples. If last_k_cycles > n_cycles, then it saves all cycles
tau = 1.0                             # time constant
g = 3.0                               # gain
dt = 0.05                             # Euler step
R = np.pi / 8                         # distance threshold for inhibition
h_base = 0.0                          # base input
seed = 0
rng = np.random.default_rng(seed)

#_____________________________
# connectivity construction 
#_____________________________
x = np.linspace(0, 2*np.pi, N, endpoint=False)  #Creates N equally spaced angles on [0,2pi)
dx = np.abs(x[:, None] - x[None, :])  # compute all pairwise absolute differences:
dist = np.minimum(dx, 2*np.pi - dx)   # circular distance
H = (dist > R).astype(float)          # Heaviside
W = w_inhib * H                       #w_ij
np.fill_diagonal(W, 0.0)            # sets diagonal entries to zero: no self-connection.   
h_vec_base = np.full(N, h_base) 

#_____________________________________________________
# One-step Euler integration of ring-network dynamics
#_____________________________________________________
def euler_step(s, h_vec):          # define a one-step update function.    
    if np.isscalar(h_vec):
        h_vec = np.full(N, float(h_vec))  # if h_vec is a scalar, convert it to a N-length vector 
    u = W @ s + h_vec                 
    r = np.maximum(u, 0.0)             
    dsdt = (-s + g * r) / tau          
    s = s + dt * dsdt                  # Euler update step
    return np.maximum(s, 0.0)          #ensure non-neagtive

#_____________________________________________________________
# calculate the central angle of the activity bump on the ring
#_____________________________________________________________
def bump_angle(s):                        # define a function that takes the current network state s
    z = np.sum(s * np.exp(1j * x))
    ang = np.angle(z)                    # the angle of the complex number z, interval (-pi,pi]
    return ang + 2*np.pi if ang < 0 else ang     # map angles to intervals to interval (0,2pi]


#—————————————————————————————————————
# Run
#—————————————————————————————————————
c0 = rng.uniform(0, 2*np.pi)  # sample a random center angle uniformly
d0 = np.minimum(np.abs(x - c0), 2*np.pi - np.abs(x - c0))  # compute shortest circular distance
s = np.exp(-(d0**2) / (2*(0.25**2)))  # create a Gaussian bump with width σ=0.25 (radians)
s = s / (s.max() + 1e-12) # prevents division-by-zero

for _ in range(6000):
    s = euler_step(s, h_vec_base)

#____________________________
# main cycle
#____________________________
angles_after_off = []
X_samples = []
for cyc in range(n_cycles):

    #on
    h_on = 2
    for _ in range(on_steps):
        s = euler_step(s, h_on)
        if cyc >= n_cycles - last_k_cycles:
            X_samples.append(s.copy())

    #off
    for _ in range(off_steps):
        h_off = rng.standard_normal(N) - 2
        s = euler_step(s, h_off)
        if cyc >= n_cycles - last_k_cycles:
            X_samples.append(s.copy())

    angles_after_off.append(bump_angle(s))

angles_after_off = np.array(angles_after_off)
X_samples = np.array(X_samples)

print("Angle range =", float(angles_after_off.max() - angles_after_off.min()))
print("unique-ish count =", len(np.unique(np.round(angles_after_off, 2))))
print("X_samples shape =", X_samples.shape)  

plt.figure(figsize=(8,3))
plt.plot(np.arange(n_cycles), angles_after_off, marker="o", lw=1)
plt.xlabel("cycle")
plt.ylabel("bump angle after OFF (rad)")
plt.title("Bump relocates after each ON/OFF")
plt.tight_layout()
plt.show()

# ______
# TDA
# ______
from sklearn.preprocessing import StandardScaler
from ripser import ripser
from persim import plot_diagrams

X = X_samples
m = min(4000, X.shape[0])
# IMPORTANT: use a separate RNG for TDA so sampling is reproducible
rng_tda = np.random.default_rng(0)
idx = rng_tda.choice(X.shape[0], size=m, replace=False)
X_sub = X[idx]
X_sub = StandardScaler().fit_transform(X_sub)
print("\nX_sub shape (NO PCA) =", X_sub.shape)


res = ripser(X_sub, maxdim=1)   # only need H1
dgms = res["dgms"]
H1 = dgms[1]

print("H1 points:", H1.shape)
if len(H1) > 0:
    lifetimes = H1[:, 1] - H1[:, 0]
    i = np.argmax(lifetimes)
    b, d = H1[i]
    print("Longest H1 bar lifetime =", float(d - b),
          "(birth=", float(b), "death=", float(d), ")")
else:
    print("No H1 found -> not S^1")
# ________
#label as H1
# __________
plt.figure()
plot_diagrams([H1], labels=["H1"], show=False)
plt.title("Persistence diagram: H1 (NO PCA)")
plt.tight_layout()
plt.show()


lifetimes = H1[:,1] - H1[:,0]
order = np.argsort(lifetimes)[::-1]
L1 = lifetimes[order[0]]
L2 = lifetimes[order[1]] if len(lifetimes) > 1 else np.nan
print("L1 =", float(L1), "L2 =", float(L2), "L1/L2 =", float(L1/L2))
