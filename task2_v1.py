import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs, LinearOperator


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


#_____________________________
# parameters
#_____________________________
N = 500
tau = 1.0
dt = 0.05
g = 3.0
seed = 0
rng = np.random.default_rng(seed)
a = 0.18
delta = 0.18
ubar = 1.0
M_ref = 7    # used only for analytical derivation, not for multi-bump simulation

#_____________________________
# threshold for D
#_____________________________
D_THRESH = 1e-6
max_relax_steps = 30000
tol_max = 1e-8
tol_rmse = 1e-8 
dt_lin = 0.05
T_lin = 400


#_____________________________
# ring coordinates
#_____________________________
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
dtheta = 2*np.pi / N   # discrete integration weight
def relu(z):
    return np.maximum(z, 0.0)

def circ_dist(theta_arr, c):
    d = np.abs(theta_arr - c)
    return np.minimum(d, 2*np.pi - d)


#_______________________________________
# analytical single bump (single center)
#_______________________________________
def analytic_bump_single(theta_arr, a, delta, ubar, center=0.0):
    u = np.zeros_like(theta_arr)
    wfreq = -np.pi / (2.0 * delta)   

    d = circ_dist(theta_arr, center)

    # flat top region
    mask_flat = d <= a
    u[mask_flat] = ubar

    # curved region
    mask_curve = (d > a) & (d <= a + delta)
    x_local = d[mask_curve] - a
    u_curve = -ubar * np.sin(wfreq * (delta - x_local))
    u[mask_curve] = np.maximum(u[mask_curve], u_curve)

    u[u < 0] = 0.0
    return u

# single bump with one center
u_star = analytic_bump_single(theta, a=a, delta=delta, ubar=ubar, center=0.0)


w_conn = -np.pi / (2.0 * delta)             
R = (2*np.pi / M_ref) - a - delta           
h = (3.0 - a * w_conn) * ubar                
h_vec = np.full(N, h)

# construct W: w_conn * H(dist > R) * dtheta
dist_mat = np.abs(theta[:, None] - theta[None, :])
dist_mat = np.minimum(dist_mat, 2*np.pi - dist_mat)
H = (dist_mat > R).astype(float)
W = (w_conn * H) * dtheta
np.fill_diagonal(W, 0.0)


#______________________________________________
# nonlinear dynamics 
#______________________________________________
def step_nonlinear(u):
    u_in = h_vec + W @ u
    du = (-u + g * relu(u_in)) / tau
    u_new = u + dt * du
    return np.maximum(u_new, 0.0)

def fixed_point_residual(u):
    u_in = h_vec + W @ u
    F = -u + g * relu(u_in)
    maxabs = float(np.max(np.abs(F)))
    rmse = float(np.linalg.norm(F) / np.sqrt(N))
    return maxabs, rmse


#____________________________________________________________
# relaxation: from analytic solution to discrete fixed point
#____________________________________________________________
u = u_star.copy()
converged = False

for t in range(max_relax_steps):
    u = step_nonlinear(u)

    if (t + 1) % 500 == 0:
        maxabs, rmse = fixed_point_residual(u)
        if (maxabs < tol_max) and (rmse < tol_rmse):
            converged = True
            print(f"Relaxation converged at step {t+1}: max|F|={maxabs:.2e}, RMSE={rmse:.2e}")
            break

if not converged:
    maxabs, rmse = fixed_point_residual(u)
    print(f"Relaxation NOT fully converged: max|F|={maxabs:.2e}, RMSE={rmse:.2e}")

u_fp = u.copy()


#____________________________________________
# linearization: maximum eigenvalue real part
#____________________________________________
u_in_fp = h_vec + W @ u_fp

D = (u_in_fp > D_THRESH).astype(float)
active_frac = float(np.mean(D))

def A_matvec(v):
    return (-v + g * (D * (W @ v))) / tau

Aop = LinearOperator((N, N), matvec=A_matvec, dtype=np.float64)
vals = eigs(Aop, k=6, which="LR", return_eigenvectors=False, tol=1e-6)
maxRe = float(np.max(np.real(vals)))

eps = 1e-4
if maxRe < -eps:
    stability = "Linear stability (disturbance decay)"
elif maxRe > eps:
    stability = "Linear instability (perturbation growth)"
else:
    stability = "Critical / near-neutral (likely translational modes)"

print("\n=== Linear stability ===")
print("max Re(eig(A)) =", maxRe)
print("stability =", stability)


#_______________________________________
# linear perturbation norm evolution
#_______________________________________
delta_vec = 1e-4 * rng.standard_normal(N)
norms = []
for _ in range(T_lin):
    norms.append(float(np.linalg.norm(delta_vec)))
    delta_vec = delta_vec + dt_lin * A_matvec(delta_vec)

plt.figure(figsize=(8, 3))
plt.plot(norms)
plt.yscale("log")
plt.xlabel("time step")
plt.ylabel(r"$||\delta||$")
plt.title("Linearized Disturbance Norm Evolution (Logarithmic Scale)")
plt.tight_layout()
plt.show()


#_______________________________________________________
# plot: analytic initial condition vs relaxed fixed point
#_______________________________________________________
plt.figure(figsize=(8, 3))
plt.plot(theta, u_star, lw=2, label="analytic single bump $u_*$")
plt.plot(theta, u_fp, lw=2, linestyle="--", label="relaxed fixed point $u_{fp}$")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$u(\theta)$")
plt.title("Single-bump: analytic vs relaxed fixed point")
plt.legend()
plt.tight_layout()
plt.show()


#_____________________________
# summary
#_____________________________
maxabs, rmse = fixed_point_residual(u_fp)
print("\n=== Summary ===")
print(f"N={N}, g={g}, tau={tau}, dt={dt}")
print(f"a={a}, delta={delta}, ubar={ubar}")
print(f"M_ref (derivation)={M_ref}, R={R:.4f}, w_conn={w_conn:.4f}, h={h:.4f}")
print(f"D threshold = {D_THRESH:.1e}")
print(f"Active fraction (D=1) = {active_frac:.3f}")
print(f"Fixed point residual: max|F|={maxabs:.2e}, RMSE={rmse:.2e}")
