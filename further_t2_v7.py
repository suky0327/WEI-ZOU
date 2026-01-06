import numpy as np
import matplotlib.pyplot as plt

#_____________________________
# parameters
#_____________________________
N = 500                               # Number of neurons
w_inhib = -1.0                        # inhibitory weight 
tau = 1.0                             # time constant
g = 3.0                               # gain
dt = 0.05                             # Euler step
R = np.pi / 8                         # distance threshold for inhibition
h_base = 0.0                          # base input
seed = 0
rng = np.random.default_rng(seed)
on_steps = 1000         # number of Euler steps in each on  segment
off_steps = 300        # number of Euler steps in each off segment
n_cycles = 100           # number of on/off cycles total
last_k_cycles = 200  # how many last cycles to save into X_samples. If last_k_cycles > n_cycles, then it saves all cycles.
m = 4000                    # subsample size for TDA point cloud
SIGNIF_RATIO = 1.5          # threshold for circular detection
USE_N_PERM = True           # whether to use Ripser’s n_perm landmark approximation
N_PERM = 1000               # number of landmarks for the approximation if enabled
DO_PLOT =   True             # whether to generate plots
n_repeats = 10 # run the same TDA pipeline 10 times, each time with a different random subsample



#_____________________________
# connectivity construction
#_____________________________
x = np.linspace(0, 2*np.pi, N, endpoint=False)  
dx = np.abs(x[:, None] - x[None, :])  
dist = np.minimum(dx, 2*np.pi - dx)  
H = (dist > R).astype(float)         
W = w_inhib * H                    
np.fill_diagonal(W, 0.0)          
h_vec_base = np.full(N, h_base)

#_____________________________________________________
# One-step Euler integration of ring-network dynamics
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
# calculate the central angle of the activity bump on the ring
#_____________________________________________________________
def bump_angle(s):                      
    z = np.sum(s * np.exp(1j * x))       
    if np.abs(z) < 1e-12:
        return np.nan
    ang = np.angle(z)               
    return ang + 2*np.pi if ang < 0 else ang   
#___________________________________________________________________________________
# measures how much of a bump's angle covers the circumference over a period of time
#___________________________________________________________________________________
def circ_range(angles):                                        
    a = np.array([v for v in angles if np.isfinite(v)])       
    if len(a) == 0:
        return np.inf
    a = np.sort(a)                                             
    gaps = np.diff(np.r_[a, a[0] + 2*np.pi])                 
    max_gap = np.max(gaps)
    return 2*np.pi - max_gap                                    

#—————————————————————————————————————
# Run
#————————————————————————————————————=
c0 = rng.uniform(0, 2*np.pi)  
d0 = np.minimum(np.abs(x - c0), 2*np.pi - np.abs(x - c0))  
s = np.exp(-(d0**2) / (2*(0.25**2))) 
s = s / (s.max() + 1e-12) 

for _ in range(6000):
    s = euler_step(s, h_vec_base)

#____________________________
# main cycle
#____________________________
angles_after_off = []  # bump angle after each cycle ends (after OFF).
X_samples = []  # store all saved neural states s

# on
for cyc in range(n_cycles):
    h_on = 2
    for _ in range(on_steps):
        s = euler_step(s, h_on)
        if cyc >= n_cycles - last_k_cycles:
            X_samples.append(s.copy())

# off
    for _ in range(off_steps):
        h_off = rng.standard_normal(N) - 2
        s = euler_step(s, h_off)
        if cyc >= n_cycles - last_k_cycles:
            X_samples.append(s.copy())

    angles_after_off.append(bump_angle(s))

angles_after_off = np.array(angles_after_off)
X_samples = np.array(X_samples)

print("Angle range =", float(np.nanmax(angles_after_off) - np.nanmin(angles_after_off)))
print("unique-ish count =", len(np.unique(np.round(angles_after_off[np.isfinite(angles_after_off)], 2))))
print("X_samples shape =", X_samples.shape)

if DO_PLOT:
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(n_cycles), angles_after_off, marker="o", lw=1)
    plt.xlabel("cycle")
    plt.ylabel("bump angle after OFF (rad)")
    plt.title("bump relocates after each ON/OFF")
    plt.tight_layout()
    plt.show()

# ______
# TDA
# ______
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ripser import ripser

X = X_samples
m = min(m, X.shape[0])
rng_tda = np.random.default_rng(0) 

def tda_metrics(X_block, mode="pca6"):  # given a point cloud X_block, compute persistent homology in H_1
    X_std = StandardScaler().fit_transform(X_block)
    explained_var = np.nan  
    if mode == "pca6":
        pca = PCA(n_components=6, random_state=0)
        Y = pca.fit_transform(X_std)
        explained_var = float(np.sum(pca.explained_variance_ratio_))  
        data = Y
    elif mode == "nopca":       
        data = X_std
    else:
        raise ValueError("mode must be 'pca6' or 'nopca'")

    if USE_N_PERM: # compute persistent homology up to dimension 1
        res = ripser(data, maxdim=1, n_perm=N_PERM)
    else:
        res = ripser(data, maxdim=1)

    H1 = res["dgms"][1]
    if H1.shape[0] == 0:
        return 0, 0.0, 0.0, np.inf, (np.nan, np.nan), H1, explained_var

    lifetimes = H1[:, 1] - H1[:, 0]
    order = np.argsort(lifetimes)[::-1]
    lifetimes_sorted = lifetimes[order] 

    L1 = float(lifetimes_sorted[0])
    L2 = float(lifetimes_sorted[1]) if len(lifetimes_sorted) > 1 else 0.0
    ratio = (L1 / (L2 + 1e-12)) if L2 > 0 else np.inf

    b, d = H1[order[0]]
    return H1.shape[0], L1, L2, ratio, (float(b), float(d)), H1, explained_var

def select_active(X, m, rng_local): # select m neural activity snapshots from a large dataset X
    X_mean = np.mean(X, axis=1)  
    x_sorted_idx = np.argsort(X_mean)[::-1] 
    pool_size = min(5*m, X.shape[0]) 
    idx_pool = x_sorted_idx[:pool_size]

    angles_pool = np.array([bump_angle(X[i]) for i in idx_pool]) # compute the bump angle for each candidate state
    valid = np.isfinite(angles_pool)
    idx_pool = idx_pool[valid]
    angles_pool = angles_pool[valid]

    n_bins = 40
    bins = np.linspace(0, 2*np.pi, n_bins + 1)
    k_per_bin = max(1, m // n_bins) # how many points to pick per bin

    idx_selected = []
    for b in range(n_bins):
        mask = (angles_pool >= bins[b]) & (angles_pool < bins[b+1])
        ids = idx_pool[mask] # find candidate states whose bump angle lies in this bin
        if len(ids) == 0:
            continue # skip empty bins
        ids_sorted = ids[np.argsort(X_mean[ids])[::-1]]
        idx_selected.extend(ids_sorted[:k_per_bin])  # within this bin, pick the most active states

    idx_selected = np.array(idx_selected, dtype=int)

    if len(idx_selected) >= m:
        idx_active = idx_selected[:m] # truncate to exactly m
    else:
        need = m - len(idx_selected)
        rest = np.setdiff1d(idx_pool, idx_selected) # find how many more points are needed
        extra = rng_local.choice(rest, size=need, replace=False) if need > 0 else np.array([], dtype=int) # randomly fill the remaining slots from the candidate pool
        idx_active = np.concatenate([idx_selected, extra])
    return idx_active

def plot_persistence(H1, title="H1 persistence"):  # plot H1 persistence diagram and highlight the longest bar
    plt.figure(figsize=(4, 4))

    if H1.shape[0] == 0:
        plt.title(title + " (empty)")
        plt.xlabel("birth"); plt.ylabel("death")
        plt.tight_layout(); plt.show()
        return

    births = H1[:, 0]                             
    deaths = H1[:, 1]                              
    lifetimes = deaths - births                    
    idx_max = int(np.argmax(lifetimes))            

    plt.scatter(births, deaths, s=25, alpha=0.6)
    plt.scatter(births[idx_max], deaths[idx_max],  
                color="red", s=70, label="longest bar")

    lim = float(max(deaths.max(), births.max()) * 1.05)
    plt.plot([0, lim], [0, lim], "k--", lw=1)            

    plt.xlabel("birth"); plt.ylabel("death")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_random_vs_active(ratios_random, ratios_active, title, threshold=SIGNIF_RATIO, bins=12):
    r = np.asarray(ratios_random, dtype=float)
    a = np.asarray(ratios_active, dtype=float)
    r = r[np.isfinite(r)]
    a = a[np.isfinite(a)]
    allv = np.concatenate([r, a]) if (len(r)+len(a)) > 0 else np.array([0, 1])

    edges = np.linspace(allv.min(), allv.max(), bins + 1)

    
    r_mean = float(np.mean(r)) if len(r) > 0 else np.nan
    r_med  = float(np.median(r)) if len(r) > 0 else np.nan
    a_mean = float(np.mean(a)) if len(a) > 0 else np.nan
    a_med  = float(np.median(a)) if len(a) > 0 else np.nan

    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.hist(r, bins=edges, alpha=0.6, label=f"Random (n={len(r)})")
    plt.hist(a, bins=edges, alpha=0.6, label=f"Active (n={len(a)})")
    plt.axvline(threshold, linestyle="--", color="k", label=f"threshold={threshold}")

    
    stats_txt = (
        f"Random: mean={r_mean:.2f}, median={r_med:.2f}\n"
        f"Active: mean={a_mean:.2f}, median={a_med:.2f}"
    )
    plt.gca().text(
        0.02, 0.98, stats_txt,
        transform=plt.gca().transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.xlabel("L1/L2")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.show()


def verdict_text(H1n_r, gap_r, H1n_a, gap_a, tag=""):                     # generate a textual verdict comparing Random vs Active
    det_r = (H1n_r > 0) and (gap_r >= SIGNIF_RATIO)                      # random detects a circle if H1 exists and L1/L2 exceeds threshold
    det_a = (H1n_a > 0) and (gap_a >= SIGNIF_RATIO)                      # active detects a circle if H1 exists and L1/L2 exceeds threshold

    if det_a and (not det_r):                                            # only Active detects a significant circle
        return (f"{tag}Conclusion: Active performs better "
                f"(Active L1/L2={gap_a:.3f} ≥ {SIGNIF_RATIO}, "
                f"Random L1/L2={gap_r:.3f} below threshold).")

    elif det_r and (not det_a):                                          # only Random detects a significant circle
        return (f"{tag}Conclusion: Random performs better "
                f"(Random L1/L2={gap_r:.3f} ≥ {SIGNIF_RATIO}, "
                f"Active L1/L2={gap_a:.3f} below threshold).")

    elif det_a and det_r:                                                # both methods detect a circle
        if gap_a > gap_r:                                                # active has stronger dominance ratio
            return (f"{tag}Conclusion: Both detect a circle, "
                    f"but Active is more pronounced "
                    f"({gap_a:.3f} > {gap_r:.3f}).")
        else:                                                            # random has stronger or equal dominance ratio
            return (f"{tag}Conclusion: Both detect a circle, "
                    f"but Random is more pronounced or more stable "
                    f"({gap_r:.3f} ≥ {gap_a:.3f}).")

    else:                                                                # neither method reaches the detection threshold
        return (f"{tag}Conclusion: Neither method reaches the threshold "
                f"(Random={gap_r:.3f}, Active={gap_a:.3f}, "
                f"threshold={SIGNIF_RATIO}).")


#_________________
# PCA vs NO PCA
#_________________
def run_one_mode(mode_name, mode_key):                                              # Run TDA comparison for one mode (PCA or NO PCA)
    print("\n" + "="*70)
    print(f"MODE: {mode_name}")
    print("="*70)

    # singel random
    idx_rand = rng_tda.choice(X.shape[0], size=m, replace=False)                
    X_rand = X[idx_rand]                                                           
    H1n_r, L1r, L2r, gap_r, (br, dr), H1_rand, ev_r = tda_metrics(X_rand, mode=mode_key)  

    print("\n=== Random subsampling ===")
    print("m =", m)
    print("H1 points:", (H1n_r, 2))
    print("L1 =", L1r, "L2 =", L2r, "L1/L2 =", gap_r)
    if mode_key == "pca6":
        print("Explained variance (PCA-6) =", ev_r) 
    if np.isfinite(br):                                                             
        print("Longest bar: birth =", br, "death =", dr, "lifetime =", dr - br)     

    # sigel active
    idx_active = select_active(X, m, rng_local=rng_tda)                             
    X_active = X[idx_active]                                                       
    angles_active = np.array([bump_angle(v) for v in X_active])                     
    ang_rng_active = circ_range(angles_active)                                      

    print("\nActive selection check:")
    X_mean = np.mean(X, axis=1)
    x_sorted_idx = np.argsort(X_mean)[::-1]
    print("X_mean[top-1 overall] =", float(X_mean[x_sorted_idx[0]]))
    print("mean(X_active[0,:])   =", float(np.mean(X_active[0, :])))
    print("Active angle circular range =",
          float(ang_rng_active) if np.isfinite(ang_rng_active) else ang_rng_active)

    H1n_a, L1a, L2a, gap_a, (ba, da), H1_act, ev_a = tda_metrics(X_active, mode=mode_key)  

    print("\n=== Active ===")
    print("m =", X_active.shape[0])
    print("H1 points:", (H1n_a, 2))
    print("L1 =", L1a, "L2 =", L2a, "L1/L2 =", gap_a)
    if mode_key == "pca6":
        print("Explained variance (PCA-6) =", ev_a)  
    if np.isfinite(ba):                                                             
        print("Longest bar: birth =", ba, "death =", da, "lifetime =", da - ba)    

    ratios_random = []
    ratios_active = []                                                             

    rng_rep = np.random.default_rng(123 if mode_key == "pca6" else 456)

    for _ in range(n_repeats):                                                      
        idx_k = rng_rep.choice(X.shape[0], size=m, replace=False)                  
        H1n_k, _, _, gk, _, _, _ = tda_metrics(X[idx_k], mode=mode_key)              
        ratios_random.append(gk)                                                    

        idx_a = select_active(X, m, rng_local=rng_rep)                              
        H1n_k2, _, _, gk2, _, _, _ = tda_metrics(X[idx_a], mode=mode_key)          
        ratios_active.append(gk2)                                                    

    ratios_random = np.array(ratios_random, dtype=float)                           
    ratios_active = np.array(ratios_active, dtype=float)                            

    # summary
    print(f"\n[Detection rule] circle if (H1n>0) and (L1/L2 >= {SIGNIF_RATIO})")      
    print(f"Random: detected={(H1n_r>0 and gap_r>=SIGNIF_RATIO)}, "
          f"H1n={H1n_r}, L1/L2={gap_r:.3f}")
    print(f"Active: detected={(H1n_a>0 and gap_a>=SIGNIF_RATIO)}, "
          f"H1n={H1n_a}, L1/L2={gap_a:.3f}")
    print(verdict_text(H1n_r, gap_r, H1n_a, gap_a, tag=f"[{mode_name}] "))

    rr = ratios_random[np.isfinite(ratios_random)]                                  
    aa = ratios_active[np.isfinite(ratios_active)]                                  
    if len(rr) > 0:
        print("\nRandom robustness (L1/L2 over repeats):")
        print("mean =", float(np.mean(rr)), "median =", float(np.median(rr)), "std =", float(np.std(rr)))  
        print("min/max =", float(np.min(rr)), float(np.max(rr)))
    if len(aa) > 0:
        print("\nActive robustness (L1/L2 over repeats):")
        print("mean =", float(np.mean(aa)), "median =", float(np.median(aa)), "std =", float(np.std(aa)))  
        print("min/max =", float(np.min(aa)), float(np.max(aa)))

    # plot
    if DO_PLOT:

        tag_r = "large" if (np.isfinite(gap_r) and gap_r >= SIGNIF_RATIO) else "small"
        tag_a = "large" if (np.isfinite(gap_a) and gap_a >= SIGNIF_RATIO) else "small"

        plot_persistence(H1_rand, title=f"{mode_name}: Random ({tag_r} L1/L2={gap_r:.2f})")
        plot_persistence(H1_act,  title=f"{mode_name}: Active ({tag_a} L1/L2={gap_a:.2f})")

        plot_random_vs_active(rr, aa, title=f"{mode_name}: Random vs Active (L1/L2)")

    return {
        "H1n_r": H1n_r, "gap_r": gap_r, "H1n_a": H1n_a, "gap_a": gap_a,
        "ratios_random": rr, "ratios_active": aa,
        "ev_r": ev_r, "ev_a": ev_a  
    }

# run PCA(6D for ripser) & NO PCA(500D for ripser)
res_pca   = run_one_mode(mode_name="With PCA (TDA on PCA-6D)", mode_key="pca6")
res_nopca = run_one_mode(mode_name="NO PCA (TDA on 500D)", mode_key="nopca")

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print("PCA ：", verdict_text(res_pca["H1n_r"], res_pca["gap_r"],
                               res_pca["H1n_a"], res_pca["gap_a"]))
print("NO PCA ：", verdict_text(res_nopca["H1n_r"], res_nopca["gap_r"],
                                  res_nopca["H1n_a"], res_nopca["gap_a"]))
