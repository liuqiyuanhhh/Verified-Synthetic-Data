import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import math
from math import sqrt, exp, pi
from statistics import NormalDist

def phi(x):
    return 1.0 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)

def Phi(x):
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2)))

def C_func(a, b):
    return (phi(b) - phi(a)) / (Phi(b) - Phi(a))

def theta_func(a, b):
    r"""
    θ(a,b)
      = 1
        - [b φ(b) - a φ(a)] / [Φ(b) - Φ(a)]
        - { [φ(b) - φ(a)] / [Φ(b) - Φ(a)] }^2
    """
    denom = Phi(b) - Phi(a)
    return 1.0  - (b*phi(b) - a*phi(a))/denom - ((phi(b) - phi(a))/denom)**2

def lambda_func(a, b):
    r"""
    λ(a,b)
     = [ (b^2 - 1)φ(b) - (a^2 - 1)φ(a) ] / [ 2(Φ(b)-Φ(a)) ]
       + [ 3(φ(b)-φ(a)) * (bφ(b) - aφ(a)) ] / [ 2(Φ(b)-Φ(a))^2 ]
       + [ (φ(b)-φ(a))^3 ] / [ (Φ(b)-Φ(a))^3 ].
    """
    denom  = Phi(b) - Phi(a)
    denom2 = denom**2
    denom3 = denom**3

    part1 = ((b**2 - 1)*phi(b) - ((a**2) - 1)*phi(a)) / (2*denom)
    part2 = (3*(phi(b)-phi(a))*(b*phi(b) - a*phi(a))) / (2*denom2)
    part3 = ((phi(b)-phi(a))**3) / denom3

    return part1 + part2 + part3


def x_bar_p(a, b, n1):
    r"""
    x_bar_p = C^2(a,b)
              + θ(a,b) / [ n1 * (Φ(b) - Φ(a)) ]
    """
    return C_func(a,b)**2 \
           + theta_func(a,b)/(n1*(Phi(b)-Phi(a)))

def y_bar_p(a, b, n1):
    r"""
    y_bar_p = C^2(a,b)
              + [ θ(a,b)^2 + 2 λ(a,b) C(a,b) ] / n1
    """
    return C_func(a,b)**2 \
           + (theta_func(a,b)**2 + 2*lambda_func(a,b)*C_func(a,b))/n1

def y_bar_pp(a, b, n1):
    r"""
    y_bar_pp = 4*C^2(a,b)
               + [ θ(a,b)/(Φ(b)-Φ(a)) ] 
                 * [ θ(a,b)^2 + 4 C(a,b) λ(a,b) ] / n1
    """
    return 4*C_func(a,b)**2 \
           + (theta_func(a,b)/(Phi(b)-Phi(a))) * \
             ((theta_func(a,b)**2 + 4*C_func(a,b)*lambda_func(a,b))/n1)

def y_bar_pp_2(a, b, n1):
    r"""
    y_bar_pp = 4*C^2(a,b)
               + [ θ(a,b)/(Φ(b)-Φ(a)) ] 
                 * [ θ(a,b)^2 + 4 C(a,b) λ(a,b) ] / n1
    """
    a1 = C_func(a,b) + a
    b1 = C_func(a,b) + b

    return (C_func(a,b)+C_func(a1,b1))**2 \
              + (theta_func(a,b)/(Phi(b)-Phi(a))) * \
                 ((theta_func(a1,b1)**2 + 2*(C_func(a,b)+C_func(a1,b1))*lambda_func(a1,b1))/n1)


def y_bar_p_finite(a, b, n1, n2):
    r"""
    y_bar_p = C^2(a,b)
              + [ θ(a,b)^2 + 2 λ(a,b) C(a,b) ] / n1
    """
    return theta_func(a,b)/(n2*(Phi(b)-Phi(a)))+C_func(a,b)**2 \
           + (theta_func(a,b)**2 + 2*lambda_func(a,b)*C_func(a,b))/n1

def y_bar_pp_2_finite(a, b, n1, n2):
    r"""
    y_bar_pp = 4*C^2(a,b)
               + [ θ(a,b)/(Φ(b)-Φ(a)) ] 
                 * [ θ(a,b)^2 + 4 C(a,b) λ(a,b) ] / n1
    """
    a1 = C_func(a,b) + a
    b1 = C_func(a,b) + b

    return theta_func(a1,b1)/(n2*(Phi(b1)-Phi(a1)))+(C_func(a,b)+C_func(a1,b1))**2 \
              + (theta_func(a,b)/(Phi(b)-Phi(a))) * \
                 ((theta_func(a1,b1)**2 + 2*(C_func(a,b)+C_func(a1,b1))*lambda_func(a1,b1))/n1)

import joblib

# Parameters
n1 = 100
n2 = 100
n_bias = 300
n_width = 300

bias_vals = np.linspace(0, 0.25, n_bias)
width_vals = np.linspace(0.01, 4, n_width)

bias_list, width_list, which_min_list = [], [], []
loss_fil_x_all = []
loss_fil_y_all = []
loss_fil_xy_all = []
loss_unf_x_all = []

x_unfilter = 1.0 / n1  # MSE of unfiltered X

# Evaluate all combinations
for bias in bias_vals:
    for width in width_vals:
        a = bias - width / 2
        b = bias + width / 2
        if b > a:
            val_x = x_bar_p(a, b, n1)
            val_y = y_bar_p_finite(a, b, n1, n2)
            val_z = y_bar_pp_2_finite(a, b, n1, n2)
            val_u = x_unfilter

            idx = np.argmin([val_x, val_y, val_z, val_u])

            bias_list.append(bias)
            width_list.append(width)
            which_min_list.append(idx)

            loss_fil_x_all.append(val_x)
            loss_fil_y_all.append(val_y)
            loss_fil_xy_all.append(val_z)
            loss_unf_x_all.append(val_u)

# Convert to arrays
bias_array = np.array(bias_list)
width_array = np.array(width_list)
which_min_array = np.array(which_min_list)
loss_fil_x_all = np.array(loss_fil_x_all)
loss_fil_y_all = np.array(loss_fil_y_all)
loss_fil_xy_all = np.array(loss_fil_xy_all)
loss_unf_x_all = np.array(loss_unf_x_all)

# Save to file
joblib.dump({
    'bias': bias_array,
    'width': width_array,
    'which_min': which_min_array,
    'loss_fil_x_all': loss_fil_x_all,
    'loss_fil_y_all': loss_fil_y_all,
    'loss_fil_xy_all': loss_fil_xy_all,
    'loss_unf_x_all': loss_unf_x_all,
}, '/Users/qiyuanliu/Desktop/model_collapse/simulation_results_theo_n2100_2.pkl')
