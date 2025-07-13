import numpy as np
import math
from math import sqrt, exp, pi
from statistics import NormalDist
import joblib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(0)
beta_star = np.random.randn(5)

def simulate_multivariate_filtering(a, gamma, n1=500, n2=500, d=5, sigma=1.0):
    """
    Simulates 4 types of beta estimators under linear regression with filtering.
    """
    beta_prime = beta_star + np.full(d, a)  # verifier's belief

    # --- Step 1: Generate real data ---
    X0 = np.random.randn(n1, d)
    eps0 = np.random.randn(n1) * sigma
    y0 = X0 @ beta_star + eps0

    # --- Estimator 1: No filtering ---
    beta0 = np.linalg.pinv(X0.T @ X0) @ X0.T @ y0

    # --- Step 2: Filter real data using verifier rule ---
    residuals_real = np.abs(y0 - X0 @ beta_prime)
    threshold_real = gamma * np.linalg.norm(X0, axis=1) + np.sqrt(2 / np.pi) * sigma
    keep_real = residuals_real <= threshold_real
    X0_filt = X0[keep_real]
    y0_filt = y0[keep_real]

    if len(X0_filt) > 0:
        beta1_real_filtered = np.linalg.pinv(X0_filt.T @ X0_filt) @ X0_filt.T @ y0_filt
    else:
        beta1_real_filtered = beta0.copy()

    # --- Step 3: Generate synthetic data for both methods ---
    X1 = np.random.randn(n2, d)

    # 3a. Based on beta1 (filtered real)
    y1_from_beta1 = X1 @ beta1_real_filtered + np.random.randn(n2) * sigma
    residuals_synth_1 = np.abs(y1_from_beta1 - X1 @ beta_prime)
    threshold_synth = gamma * np.linalg.norm(X1, axis=1) + np.sqrt(2 / np.pi) * sigma
    keep_synth_1 = residuals_synth_1 <= threshold_synth
    X1_filt_1 = X1[keep_synth_1]
    y1_filt_1 = y1_from_beta1[keep_synth_1]

    if len(X1_filt_1) > 0:
        beta2_double_filtered = np.linalg.pinv(X1_filt_1.T @ X1_filt_1) @ X1_filt_1.T @ y1_filt_1
    else:
        beta2_double_filtered = beta1_real_filtered.copy()

    # 3b. Based on beta0 (unfiltered real)
    y1_from_beta0 = X1 @ beta0 + np.random.randn(n2) * sigma
    residuals_synth_2 = np.abs(y1_from_beta0 - X1 @ beta_prime)
    keep_synth_2 = residuals_synth_2 <= threshold_synth
    X1_filt_2 = X1[keep_synth_2]
    y1_filt_2 = y1_from_beta0[keep_synth_2]

    if len(X1_filt_2) > 0:
        beta3_filtered_synth_only = np.linalg.pinv(X1_filt_2.T @ X1_filt_2) @ X1_filt_2.T @ y1_filt_2
    else:
        beta3_filtered_synth_only = beta0.copy()

    # --- Losses relative to true beta* ---
    losses = {
        'unfiltered_x': np.linalg.norm(beta0 - beta_star)**2,
        'filtered_x': np.linalg.norm(beta1_real_filtered - beta_star)**2,
        'filtered_xy': np.linalg.norm(beta2_double_filtered - beta_star)**2,
        'filter_y': np.linalg.norm(beta3_filtered_synth_only - beta_star)**2
    }
    return losses

    

# --- Parameters
random_seed = 0
n1 = 100
n2 = 100
sim_n = 100
d = 5
sigma = 1.0
n_bias = 300
n_width = 150
bias_vals = np.linspace(0, 0.25, n_bias)
width_vals = np.linspace(0.01, 0.5, n_width)
np.random.seed(random_seed)

# --- Results containers
which_min = np.full((n_bias, n_width), -1, dtype=int)
loss_unfiltered_x_all = np.full((n_bias, n_width), np.nan)
loss_filtered_x_all = np.full((n_bias, n_width), np.nan)
loss_filtered_xy_all = np.full((n_bias, n_width), np.nan)
loss_filter_y_only_all = np.full((n_bias, n_width), np.nan)

# --- Simulation job function (no progress bar or globals inside)
def compute_loss_entry(i, j, bias, width, n1, n2, sim_n):
    if width <= 0:
        return i, j, -1, None, None, None, None

    losses = {
        'unfiltered_x': 0.0,
        'filtered_x': 0.0,
        'filtered_xy': 0.0,
        'filter_y': 0.0
    }

    for _ in range(sim_n):
        out = simulate_multivariate_filtering(
            a=bias, gamma=width, n1=n1, n2=n2, d=d, sigma=sigma
        )
        for key in losses:
            losses[key] += out[key]

    for key in losses:
        losses[key] /= sim_n

    loss_list = [
        losses['filtered_x'],
        losses['filter_y'],
        losses['filtered_xy'],
        losses['unfiltered_x']
    ]
    best_idx = np.argmin(loss_list)

    return i, j, best_idx, *loss_list

# --- Run simulations in parallel with progress bar
tasks = (
    delayed(compute_loss_entry)(i, j, bias_vals[i], width_vals[j], n1, n2, sim_n)
    for i in range(n_bias) for j in range(n_width)
)

results = Parallel(n_jobs=-1)(
    task for task in tqdm(tasks, total=n_bias * n_width, desc="Simulating (multivariate)", dynamic_ncols=True)
)

# --- Store results
for res in results:
    i, j, idx, loss_fx, loss_fy, loss_fxy, loss_ux = res
    if idx != -1:
        which_min[i, j] = idx
        loss_filtered_x_all[i, j] = loss_fx
        loss_filter_y_only_all[i, j] = loss_fy
        loss_filtered_xy_all[i, j] = loss_fxy
        loss_unfiltered_x_all[i, j] = loss_ux

# --- Save
joblib.dump({
    'which_min': which_min,
    'bias': bias_vals,
    'width': width_vals,
    'loss_fil_x_all': loss_filtered_x_all,
    'loss_fil_y_all': loss_filter_y_only_all,
    'loss_fil_xy_all': loss_filtered_xy_all,
    'loss_unf_x_all': loss_unfiltered_x_all
}, "/home/qiyuanliu/simulation_results_lr_n2100_2.pkl")
