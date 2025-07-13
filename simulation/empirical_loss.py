import numpy as np
import math
from math import sqrt, exp, pi
from statistics import NormalDist
import joblib
import numpy as np
from joblib import Parallel, delayed

from tqdm import tqdm

def filter_data(data, a, b):
    return data[(data >= a) & (data <= b)]

def simulation_3(n1, n2, a, b):

    data = np.random.normal(0, 1, n1)
    bar_x = np.mean(data)
    filtered_x_data = filter_data(data, a, b)
    #if len(filtered_X) == 0: data_filtered_X_mean = bar_x
    #else: data_filtered_X_mean = np.mean(filtered_X)
    if len(filtered_x_data) == 0: data_filtered_X_mean = bar_x
    else: data_filtered_X_mean = np.mean(filtered_x_data)

    y = np.random.normal(bar_x, 1, n2)
    y_2 = np.random.normal(data_filtered_X_mean, 1, n2)
    filtered_x_y_data = filter_data(y_2, a, b)
    if len(filtered_x_y_data) == 0: filter_x_y = data_filtered_X_mean
    else: filter_x_y = np.mean(filtered_x_y_data)
    filtered_y_data = filter_data(y, a, b)
    if len(filtered_y_data) == 0: filter_y = bar_x
    else: filter_y = np.mean(filtered_y_data)
    unfilter_y = np.mean(y)

    return (bar_x,unfilter_y,data_filtered_X_mean,filter_x_y, filter_y)

# Parameters
random_seed = 0
n1 = 100
n2 = 100000
sim_n = 1000
n_bias = 300
n_width = 300
bias_vals = np.linspace(0, 0.25, n_bias)
width_vals = np.linspace(0.01, 4, n_width)
np.random.seed(random_seed)


# Create a progress bar
progress_bar = tqdm(total=n_bias * n_width, desc="Simulations", dynamic_ncols=True)

# Define a callback wrapper
def with_progress(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        progress_bar.update(1)
        return result
    return wrapper


# Initialize arrays
which_min = np.full((n_bias, n_width), -1, dtype=int)
loss_fil_x_all = np.full((n_bias, n_width), np.nan)
loss_fil_y_all = np.full((n_bias, n_width), np.nan)
loss_fil_xy_all = np.full((n_bias, n_width), np.nan)
loss_unf_x_all = np.full((n_bias, n_width), np.nan)
loss_unf_y_all = np.full((n_bias, n_width), np.nan)

# Parallelized computation
def compute_empirical_loss(i, j, bias, width, n1, n2, sim_n):
    a = bias - width / 2
    b = bias + width / 2
    if b <= a:
        return i, j, -1, None, None, None, None, None

    loss_unf_x = loss_unf_y = loss_fil_x = loss_fil_xy = loss_fil_y = 0.0
    for _ in range(sim_n):
        bar_x, unfilter_y, filtered_x, filter_x_y, filter_y = simulation_3(n1, n2, a, b)
        loss_unf_x  += bar_x**2
        loss_unf_y  += unfilter_y**2
        loss_fil_x  += filtered_x**2
        loss_fil_xy += filter_x_y**2
        loss_fil_y  += filter_y**2
    loss_unf_x /= sim_n
    loss_unf_y /= sim_n
    loss_fil_x /= sim_n
    loss_fil_xy /= sim_n
    loss_fil_y /= sim_n
    idx = np.argmin([loss_fil_x, loss_fil_y, loss_fil_xy, loss_unf_x])
    return i, j, idx, loss_fil_x, loss_fil_y, loss_fil_xy, loss_unf_x, loss_unf_y


results = Parallel(n_jobs=-1, verbose=1)(
    delayed(compute_empirical_loss)(i, j, bias_vals[i], width_vals[j], n1, n2, sim_n)
    for i in range(n_bias) for j in range(n_width)
)
# close
progress_bar.close()

# Store results
for i, j, idx, loss_fx, loss_fy, loss_fxy, loss_ux, loss_uy in results:
    if idx != -1:
        which_min[i, j] = idx
        loss_fil_x_all[i, j] = loss_fx
        loss_fil_y_all[i, j] = loss_fy
        loss_fil_xy_all[i, j] = loss_fxy
        loss_unf_x_all[i, j] = loss_ux
        loss_unf_y_all[i, j] = loss_uy
joblib.dump({
    'which_min': which_min,
    'loss_fil_x_all': loss_fil_x_all,
    'loss_fil_y_all': loss_fil_y_all,
    'loss_fil_xy_all': loss_fil_xy_all,
    'loss_unf_x_all': loss_unf_x_all,
    'loss_unf_y_all': loss_unf_y_all
}, '/home/qiyuanliu/simulation_results_n2400_1000.pkl')