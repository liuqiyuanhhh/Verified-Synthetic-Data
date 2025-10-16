import numpy as np
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse

# ----------------------------
# Global true parameter (kept for compatibility)
# ----------------------------
np.random.seed(0)
beta_star = np.ones(8)  # shape (d,)

parser = argparse.ArgumentParser()
parser.add_argument("n2", type=float, help="per-direction kept size after filtering")
args = parser.parse_args()
bias_l = args.n2     # (kept for backward compatibility with your scripts)
n2 = 100

# ----------------------------
# Core simulator: one-round, per-direction retraining
# Uses a FIXED global X0 and its SVD directions V (both passed in)
# ----------------------------
def simulate_one_round_per_direction(
    a, gamma, X0, V, n2=500, d=5, sigma=1.0,
    raw_batch_start=2000, raw_growth=2.0, max_batches=100,
    rng=None, direction_u=None
):
    """
    One Monte Carlo replicate with fixed design X0 (global) and SVD directions V:
      - y0 = X0 @ beta_star + eps0
      - beta0 = OLS(X0, y0)
      - Synthetic per-direction using mean v_j^T beta0 + noise
      - Three estimators:
          1) beta_real_only (no synthetic)
          2) beta_synth_filter (filter, per-direction kept exactly n2)
          3) beta_synth_nofilter (no filter, per-direction sample size n2)
      - Return their squared L2 losses relative to beta_star

    Verifier center: beta_prime = beta_star + a * u
      where u is a unit vector (here passed in as the single global direction).
    Threshold: |y - v_j^T beta_prime| <= gamma * ||v_j|| + sqrt(2/pi)*sigma
    """
    if rng is None:
        rng = np.random.default_rng()
    if direction_u is None:
        raise ValueError("direction_u must be provided as the global unit vector.")

    u = np.asarray(direction_u, dtype=float)
    u = u / np.linalg.norm(u)
    d = V.shape[0]  # ensure consistent dimensionality with X0/V
    beta_prime = beta_star[:d] + a * u

    # ----- real data -> beta0 (X0 is fixed; noise changes per replicate) -----
    n1 = X0.shape[0]
    eps0 = rng.normal(scale=sigma, size=n1)
    y0 = X0 @ beta_star[:d] + eps0
    beta0, *_ = np.linalg.lstsq(X0, y0, rcond=None)  # (d,)

    # ----- Estimator 1: real-only -----
    beta_real_only = beta0.copy()

    # ----- Estimator 2: per-direction synthetic with filtering (keep exactly n2 per direction) -----
    a_coords_fil = np.zeros(d, dtype=float)
    for j in range(d):
        vj = V[:, j]
        vj_norm = float(np.sqrt(vj @ vj))  # ~1
        center = float(vj @ beta_prime)
        mean_along = float(vj @ beta0)

        kept = []
        batch = int(max(1, raw_batch_start))
        batches_used = 0
        thresh = gamma * vj_norm + np.sqrt(2.0/np.pi) * sigma

        while len(kept) < n2:
            y_raw = mean_along + rng.normal(scale=sigma, size=batch)
            mask = np.abs(y_raw - center) <= thresh
            if np.any(mask):
                kept.extend(y_raw[mask].tolist())
            if len(kept) < n2:
                batch = int(np.ceil(batch * raw_growth))
                batches_used += 1
                if batches_used > max_batches:
                    a_coords_fil[j] = float(np.mean(kept)) if kept else mean_along
                    break

        if len(kept) >= n2:
            y_kept = np.array(kept[:n2], dtype=float)
            a_coords_fil[j] = float(np.mean(y_kept))

    beta_synth_filter = V @ a_coords_fil  # recombine

    # ----- Estimator 3: per-direction synthetic without filtering (exact n2 per direction) -----
    a_coords_nof = np.zeros(d, dtype=float)
    for j in range(d):
        vj = V[:, j]
        mean_along = float(vj @ beta0)
        y_raw = mean_along + rng.normal(scale=sigma, size=n2)
        a_coords_nof[j] = float(np.mean(y_raw))
    beta_synth_nofilter = V @ a_coords_nof

    # ----- losses -----
    loss_real_only      = float(np.linalg.norm(beta_real_only      - beta_star[:d])**2)
    loss_synth_filter   = float(np.linalg.norm(beta_synth_filter   - beta_star[:d])**2)
    loss_synth_nofilter = float(np.linalg.norm(beta_synth_nofilter - beta_star[:d])**2)

    return loss_real_only, loss_synth_filter, loss_synth_nofilter


# ----------------------------
# Monte Carlo average for a (bias), gamma (width), with fixed X0 and V
# ----------------------------
def compute_loss_entry(i, j, bias, width, X0, V, n2, d, sigma, sim_n, seed_base, global_u):
    """
    Return averaged losses over sim_n replicates for (bias, width).
    Uses a SINGLE GLOBAL direction 'global_u' for the whole run:
      beta_prime = beta_star + bias * global_u
    Only three variants:
      0: real_only
      1: synth_filter
      2: synth_nofilter
    """
    if width <= 0:
        return i, j, np.nan, np.nan, np.nan, -1

    losses = np.zeros(3, dtype=float)  # [real_only, synth_filter, synth_nofilter]

    for t in range(sim_n):
        rng = np.random.default_rng(seed_base + 97*i + 131*j + 17*t)
        l0, l1, l2 = simulate_one_round_per_direction(
            a=bias, gamma=width, X0=X0, V=V, n2=n2, d=d, sigma=sigma,
            raw_batch_start=2000, raw_growth=2.0, max_batches=100,
            rng=rng, direction_u=global_u
        )
        losses[0] += l0
        losses[1] += l1
        losses[2] += l2

    losses /= sim_n
    best_idx = int(np.nanargmin(losses))  # 0,1,2
    return i, j, losses[0], losses[1], losses[2], best_idx


# --- Parameters (keep your style) ---
random_seed = 0
n1 = 800
sim_n = 100
d = 8
sigma = 1.0
n_bias = 300
n_width = 150
bias_vals = np.linspace(0, bias_l, n_bias)   # bias magnitude -> ||beta_star - beta_prime|| = bias
width_vals = np.linspace(0.01, 0.5, n_width)
np.random.seed(random_seed)

# --- SINGLE GLOBAL RANDOM DIRECTION U (unit) FOR THE ENTIRE RUN ---
rng_dir = np.random.default_rng(random_seed + 777)  # deterministic from random_seed
u = rng_dir.normal(size=d)
u = u / np.linalg.norm(u)

# --- FIXED GLOBAL X0 and its SVD directions V ---
rng_x0 = np.random.default_rng(random_seed + 12345)
X0 = rng_x0.normal(size=(n1, d))
# Precompute V from X0 (RIGHT singular vectors)
Vt = np.linalg.svd(X0, full_matrices=False)[2]  # shape (d, d)
V = Vt.T

# --- Containers ---
which_min = np.full((n_bias, n_width), -1, dtype=int)
loss_real_only_all   = np.full((n_bias, n_width), np.nan)
loss_synth_fil_all   = np.full((n_bias, n_width), np.nan)
loss_synth_nofil_all = np.full((n_bias, n_width), np.nan)

# --- Parallel jobs ---
tasks = (
    delayed(compute_loss_entry)(
        i, j, bias_vals[i], width_vals[j],
        X0, V, n2, d, sigma, sim_n, seed_base=random_seed, global_u=u
    )
    for i in range(n_bias) for j in range(n_width)
)

results = Parallel(n_jobs=-1)(
    task for task in tqdm(
        tasks, total=n_bias * n_width,
        desc="Simulating (one-round per-direction)", dynamic_ncols=True
    )
)

# --- Store results ---
for res in results:
    i, j, l_real, l_fil, l_nofil, idx = res
    loss_real_only_all[i, j]   = l_real
    loss_synth_fil_all[i, j]   = l_fil
    loss_synth_nofil_all[i, j] = l_nofil
    which_min[i, j] = idx

# --- Save ---
out_path = (
    f"/home/qiyuanliu/data_filter/Verified-Synthetic-Data/simulation/"
    f"simulation_results_lr_one_round_per_dir_n1{n1}_n2{n2}_bias_l{bias_l}_d{1}.pkl"
)

joblib.dump({
    'which_min': which_min,         # 0:real_only, 1:synth_filter, 2:synth_nofilter
    'bias': bias_vals,
    'width': width_vals,
    'loss_real_only':     loss_real_only_all,
    'loss_synth_filter':  loss_synth_fil_all,
    'loss_synth_nofilter': loss_synth_nofil_all,
    # Persist reproducibility artifacts
    'u': u,                         # single global bias direction (unit vector)
    'X0': X0,                       # FIXED global design matrix
    'V': V,                         # SVD right singular vectors of X0 (columns)
    'meta': {
        'n1': n1, 'n2': n2, 'd': d, 'sigma': sigma, 'sim_n': sim_n,
        'random_seed': random_seed,
        'u_seed_expr': 'random_seed + 777',
        'X0_seed_expr': 'random_seed + 12345',
        'note': 'One-round per-direction; fixed global X0 & V; single global u with ||beta*-beta_prime||=bias.'
    }
}, out_path)

print(f"Saved results to: {out_path}")
print("Global bias direction u (unit) was saved under key 'u'. Norm:", np.linalg.norm(u))
print("Global X0 shape:", X0.shape, "and V shape:", V.shape)
