#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 09:46:00 2026

@author: jicezeng
"""
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import tueplots.bundles
import tueplots.axes
custom_params = {
    'font.family':'Serif', 
    'font.weight':'ultralight', 
    'font.serif':'Times New Roman',
    "axes.labelweight": "light",
    "mathtext.fontset": "stix",
    'text.usetex': False, # this for LaTex
}
sns.set_theme(style="ticks", context='paper', rc=custom_params)
plt.rcParams.update(tueplots.bundles.icml2022())
plt.rcParams.update(tueplots.axes.lines(base_width=0.5))

plt.rcParams['text.usetex'] = False
# ==========================================
# 1. 基础有限元模型部分 (保持不变)
# ==========================================
L = 2.5   # 轨枕长度 (m)
b = 0.32  # 截面宽度 (m)
h = 0.18  # 截面高度 (m)
E_nominal = 38e9  # 标称弹性模量 (Pa)
G = 15.2e9  # 剪切模量 (Pa)
density = 2200  # 密度 (kg/m³)
rail_mass_nominal = 42  # 标称钢轨质量 (kg)
base_stiffness_nominal = 150e6  # 标称道床刚度 (N/m²)

# 截面属性
A = b * h
A_shear = 0.8 * A
I = b * h ** 3 / 12
num_elements = 48
num_nodes = num_elements + 1
dx = L / num_elements
num_zones = 3

def timoshenko_beam_stiffness(E, G, A, A_shear, I, L_elem, foundation_stiffness=0):
    phi = (12 * E * I) / (G * A_shear * L_elem ** 2)
    k = np.zeros((4, 4))
    factor = E * I / ((1 + phi) * L_elem ** 3)
    k[0, 0] = 12 * factor + foundation_stiffness
    k[0, 1] = -6 * L_elem * factor
    k[0, 2] = -12 * factor
    k[0, 3] = -6 * L_elem * factor
    k[1, 0] = -6 * L_elem * factor
    k[1, 1] = (4 + phi) * L_elem ** 2 * factor
    k[1, 2] = 6 * L_elem * factor
    k[1, 3] = (2 - phi) * L_elem ** 2 * factor
    k[2, 0] = -12 * factor
    k[2, 1] = 6 * L_elem * factor
    k[2, 2] = 12 * factor + foundation_stiffness
    k[2, 3] = 6 * L_elem * factor
    k[3, 0] = -6 * L_elem * factor
    k[3, 1] = (2 - phi) * L_elem ** 2 * factor
    k[3, 2] = 6 * L_elem * factor
    k[3, 3] = (4 + phi) * L_elem ** 2 * factor
    return k

def timoshenko_beam_mass(density, A, L_elem):
    m = np.zeros((4, 4))
    total_mass = density * A * L_elem
    m[0, 0] = 13 * total_mass / 35
    m[0, 1] = 11 * total_mass * L_elem / 210
    m[0, 2] = 9 * total_mass / 70
    m[0, 3] = -13 * total_mass * L_elem / 420
    m[1, 0] = 11 * total_mass * L_elem / 210
    m[1, 1] = total_mass * L_elem ** 2 / 105
    m[1, 2] = 13 * total_mass * L_elem / 420
    m[1, 3] = -total_mass * L_elem ** 2 / 140
    m[2, 0] = 9 * total_mass / 70
    m[2, 1] = 13 * total_mass * L_elem / 420
    m[2, 2] = 13 * total_mass / 35
    m[2, 3] = -11 * total_mass * L_elem / 210
    m[3, 0] = -13 * total_mass * L_elem / 420
    m[3, 1] = -total_mass * L_elem ** 2 / 140
    m[3, 2] = -11 * total_mass * L_elem / 210
    m[3, 3] = total_mass * L_elem ** 2 / 105
    return m

def determine_element_zone(element_index, num_elements, num_zones):
    elements_per_zone = num_elements // num_zones
    zone_id = element_index // elements_per_zone
    return min(zone_id, num_zones - 1)

def assemble_three_zone_bed_model(zone_stiffness_factors, E_scale=1.0, 
                                  rail_mass_left=1.0, rail_mass_right=1.0):
    E_local = E_nominal * E_scale
    rail_mass_L = rail_mass_nominal * rail_mass_left
    rail_mass_R = rail_mass_nominal * rail_mass_right
    K_global = np.zeros((2 * num_nodes, 2 * num_nodes))
    M_global = np.zeros((2 * num_nodes, 2 * num_nodes))

    for i in range(num_elements):
        dofs = [2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1]
        L_elem = dx
        zone_id = determine_element_zone(i, num_elements, num_zones)
        foundation_stiffness_per_length = base_stiffness_nominal * zone_stiffness_factors[zone_id]
        foundation_stiffness = foundation_stiffness_per_length * L_elem
        ke = timoshenko_beam_stiffness(E_local, G, A, A_shear, I, L_elem, foundation_stiffness)
        K_global[np.ix_(dofs, dofs)] += ke
        me = timoshenko_beam_mass(density, A, L_elem)
        M_global[np.ix_(dofs, dofs)] += me

    rail_gauge = 1.435
    rail_positions = [L / 2 - rail_gauge / 2, L / 2 + rail_gauge / 2]
    nodes_coords = np.linspace(0, L, num_nodes)
    for pos_idx, pos in enumerate(rail_positions):
        node_idx = np.argmin(np.abs(nodes_coords - pos))
        rail_mass = rail_mass_L if pos_idx == 0 else rail_mass_R
        M_global[2 * node_idx, 2 * node_idx] += rail_mass

    return K_global, M_global

# %%
# Vertical displacement DOFs are affected, Rotational DOFs are NOT
ndof = 2 * num_nodes
B = np.zeros((ndof, 1))

# Apply ground motion to vertical displacement DOFs only
for i in range(num_nodes):
    B[2 * i, 0] = 1.0   # vertical DOF
# Rayleigh damping function
def get_rayleigh_damping(K, M, modes=(1, 3), zeta=(0.05, 0.05)):
    eigvals, _ = eigh(K, M)
    omega = np.sqrt(np.real(eigvals))

    w1 = omega[modes[0] - 1]
    w2 = omega[modes[1] - 1]

    A = np.array([[1/(2*w1), w1/2],
                  [1/(2*w2), w2/2]])
    b = np.array(zeta)

    alpha, beta = np.linalg.solve(A, b)
    C = alpha * M + beta * K
    return C
# Newmark time integration
def newmark(K, M, C, f, record, beta, gamma, fs_original, u0, v0, a0):
    # Check that beta and gamma are within acceptable ranges
    if beta > 1/2 or beta < 1/4:
        raise ValueError("Beta is not in the appropriate range")
    if gamma != 1/2:
        raise ValueError("Gamma is not in the appropriate range")

    # Initialize parameters
    dof = K.shape[0]
    samples = f.shape[1]
    dt = 1 / fs_original

    # Step 1: Compute effective stiffness parameters
    a1 = 1 / (beta * dt ** 2) * M + gamma / (beta * dt) * C
    a2 = (1 / (beta * dt)) * M + (gamma / beta - 1) * C
    a3 = (1 / (2 * beta) - 1) * M + dt / 2 * (gamma / beta - 2) * C

    Khat = a1 + K

    # Initialize response arrays
    f_hat = np.zeros((dof, samples))
    u = np.zeros((dof, samples))
    v = np.zeros((dof, samples))
    a_rel = np.zeros((dof, samples))
    a_abs = np.zeros((dof, samples))

    # Set initial conditions
    u[:, 0] = u0
    v[:, 0] = v0
    a_rel[:, 0] = a0
    a_abs[:, 0] = a0 + record[0]

    # Time-stepping loop
    for i in range(samples - 1):
        f_hat[:, i + 1] = f[:, i + 1] + a1 @ u[:, i] + a2 @ v[:, i] + a3 @ a_rel[:, i]
        u[:, i + 1] = np.linalg.solve(Khat, f_hat[:, i + 1])  # Solves for u(:, i+1)
        a_rel[:, i + 1] = (1 / (beta * dt ** 2)) * (u[:, i + 1] - u[:, i] - dt * v[:, i] - (1 / 2 - beta) * dt ** 2 * a_rel[:, i])
        v[:, i + 1] = v[:, i] + dt * ((1 - gamma) * a_rel[:, i] + gamma * a_rel[:, i + 1])
        a_abs[:, i + 1] = a_rel[:, i + 1] + record[i + 1]

    return a_rel.T, a_abs.T, v.T, u.T,

def newmark_vary(K, M, C, f, ug, beta, gamma, fs_original, u0, v0, a0):

    dof = K.shape[0]
    samples = f.shape[1]
    dt = 1 / fs_original

    a1 = 1 / (beta * dt ** 2) * M + gamma / (beta * dt) * C
    a2 = (1 / (beta * dt)) * M + (gamma / beta - 1) * C
    a3 = (1 / (2 * beta) - 1) * M + dt / 2 * (gamma / beta - 2) * C

    Khat = a1 + K

    f_hat = np.zeros((dof, samples))
    u = np.zeros((dof, samples))
    v = np.zeros((dof, samples))
    a_rel = np.zeros((dof, samples))
    a_abs = np.zeros((dof, samples))

    u[:, 0] = u0
    v[:, 0] = v0
    a_rel[:, 0] = a0
    a_abs[:, 0] = a0 + ug[:, 0]

    for i in range(samples - 1):
        f_hat[:, i + 1] = (
            f[:, i + 1]
            + a1 @ u[:, i]
            + a2 @ v[:, i]
            + a3 @ a_rel[:, i]
        )

        u[:, i + 1] = np.linalg.solve(Khat, f_hat[:, i + 1])

        a_rel[:, i + 1] = (
            (1 / (beta * dt ** 2)) *
            (u[:, i + 1] - u[:, i] - dt * v[:, i]
             - (1 / 2 - beta) * dt ** 2 * a_rel[:, i])
        )

        v[:, i + 1] = v[:, i] + dt * (
            (1 - gamma) * a_rel[:, i] + gamma * a_rel[:, i + 1]
        )

        a_abs[:, i + 1] = a_rel[:, i + 1] + ug[:, i + 1]

    return a_rel.T, a_abs.T, v.T, u.T

# %% Full earthquake simulation for rail sleeper
theta = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # example parameters

K_global, M_global = assemble_three_zone_bed_model(
    zone_stiffness_factors=theta[0:3],
    E_scale=theta[3],
    rail_mass_left=theta[4],
    rail_mass_right=theta[5]
)

C_global = get_rayleigh_damping(
    K_global, M_global,
    modes=(1, 3),
    zeta=(0.05, 0.05)
)


u0 = np.zeros(ndof)
v0 = np.zeros(ndof)
a0 = np.zeros(ndof)

sensor_node_indices = np.array([0, 5, 13, 17, 20, 24, 28, 31, 35, 43, 48])
meas_dofs = sensor_node_indices * 2   # vertical DOFs

# %% Hammer force
fs = 6400          # Hz (hammer tests need high fs), 6.4 KHz
dt = 1 / fs
T = 0.03            # total duration (s)
t = np.arange(0, T, dt)
nt = len(t)

F0 = 1000.0         # peak force (N)
Tp = 0.005          # pulse duration (s)
# Half-sine hammer force
force = np.zeros(nt)
idx = t <= Tp
force[idx] = F0 * np.sin(np.pi * t[idx] / Tp)
# ===============================
# Plot
# ===============================
plt.figure(figsize=(6, 4), dpi=600)
plt.plot(t, force, linewidth=1.5)
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("Force (N)", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.title("Hammer Impact Force (Half-Sine Pulse)", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# impact location: middle span
nodes_coords = np.linspace(0, L, num_nodes)
impact_node = np.argmin(np.abs(nodes_coords - L/2))
impact_dof = 2 * impact_node   # vertical DOF

# Build force vector 
ndof = 2 * num_nodes
f = np.zeros((ndof, nt))
f[impact_dof, :] = force

u0 = np.zeros(ndof)
v0 = np.zeros(ndof)
a0 = np.zeros(ndof)

a_rel, a_abs, v, u = newmark(
    K_global, M_global, C_global,
    f, record=np.zeros(nt),   # no base excitation
    beta=1/4, gamma=1/2,
    fs_original=fs,
    u0=u0, v0=v0, a0=a0
)

Accel_meas = a_abs[:, meas_dofs]
# Plot figure
n_sensors = len(sensor_node_indices)
nrows = 4
ncols = 3  # 12 slots, 1 unused

TITLE_FONTSIZE = 23
LABEL_FONTSIZE = 23
TICK_FONTSIZE  = 17

T_total = 0.03  # seconds
nt = Accel_meas.shape[0]
t = np.linspace(0.0, T_total, nt)

plt.figure(figsize=(14, 11), dpi=600)

for i in range(n_sensors):
    ax = plt.subplot(nrows, ncols, i + 1)
    ax.plot(t, Accel_meas[:, i], linewidth=1.5)

    ax.set_xlabel("Time (s)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Acceleration (g)", fontsize=LABEL_FONTSIZE)

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_xlim(0.0, T_total)
    ax.grid(True)

# Hide unused subplot
ax = plt.subplot(nrows, ncols, n_sensors + 1)
ax.axis("off")

plt.tight_layout()
plt.show()

# %% Generate trainining data
import numpy as np

n_samples = 1000
theta_dim = 6   # adjust if needed

# Uniform prior: theta ~ U(0, 2)
theta_samples = np.random.uniform(
    low=0.0,
    high=2.0,
    size=(n_samples, theta_dim)
)
nt = f.shape[1]
n_sensors = len(meas_dofs)

Accel_dataset = np.zeros((n_samples, nt, n_sensors))

for i in range(n_samples):

    theta = theta_samples[i]

    # --- unpack theta (example) ---
    zone_stiffness_factors = theta[0:3]
    E_scale = theta[3]
    rail_mass_left = theta[4]
    rail_mass_right = theta[5]

    # --- assemble system ---
    K_global, M_global = assemble_three_zone_bed_model(
        zone_stiffness_factors=zone_stiffness_factors,
        E_scale=E_scale,
        rail_mass_left=rail_mass_left,
        rail_mass_right=rail_mass_right
    )

    C_global = get_rayleigh_damping(
        K_global, M_global,
        modes=(1, 3),
        zeta=(0.05, 0.05)
    )

    a_rel, a_abs, v, u = newmark(
        K_global, M_global, C_global,
        f, record=np.zeros(nt),   # hammer force only
        beta=1/4, gamma=1/2,
        fs_original=fs,
        u0=u0, v0=v0, a0=a0
    )

    Accel_dataset[i] = a_abs[:, meas_dofs]

    if (i + 1) % 100 == 0:
        print(f"Generated {i+1}/{n_samples}")


n_sensors = len(sensor_node_indices)
nrows = 4
ncols = 3  # 12 slots, 1 unused

TITLE_FONTSIZE = 23
LABEL_FONTSIZE = 23
TICK_FONTSIZE  = 17

T_total = 0.03  # seconds
nt = Accel_meas.shape[0]
t = np.linspace(0.0, T_total, nt)

plt.figure(figsize=(14, 11), dpi=600)
index=330
print(theta_samples[index,:])
for i in range(n_sensors):
    ax = plt.subplot(nrows, ncols, i + 1)
    ax.plot(t, Accel_dataset[index,:, i], linewidth=1.5)
    # ax.set_title(f"Node {sensor_node_indices[i]}", fontsize=TITLE_FONTSIZE)
    ax.set_title(f"Sensor {i+1}", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Time (s)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Vertical acc.", fontsize=LABEL_FONTSIZE)

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_xlim(0.0, T_total)
    ax.grid(True)

# Hide unused subplot
ax = plt.subplot(nrows, ncols, n_sensors + 1)
ax.axis("off")

plt.tight_layout()
plt.show()

np.savez(
    "sleeper_hammer_dataset.npz",
    theta=theta_samples,
    accel=Accel_dataset,
    fs=fs,
    sensor_nodes=sensor_node_indices,
    prior_range=[0.0, 2.0],
    excitation="hammer_half_sine_1kN_5ms"
)

# save train, valid, and test
theta_train = theta_samples[0:800]
theta_valid = theta_samples[800:900]
theta_test  = theta_samples[900:1000]

accel_train = Accel_dataset[0:800]
accel_valid = Accel_dataset[800:900]
accel_test  = Accel_dataset[900:1000]

# %% scale the data
mu = Accel_dataset.mean(axis=(0, 1))      # (11,)
sigma = Accel_dataset.std(axis=(0, 1))    # (11,)

sigma[sigma < 1e-8] = 1.0
import torch
from torch import Tensor

class AccelScaler:
    def __init__(self, mu, sigma):
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

    def preprocess(self, x: Tensor) -> Tensor:
        return (x - self.mu) / self.sigma

    def postprocess(self, x: Tensor) -> Tensor:
        return self.mu + self.sigma * x

scaler = AccelScaler(mu, sigma)

accel_train_scaled = scaler.preprocess(torch.tensor(accel_train, dtype=torch.float32))
accel_valid_scaled = scaler.preprocess(torch.tensor(accel_valid, dtype=torch.float32))
accel_test_scaled  = scaler.preprocess(torch.tensor(accel_test,  dtype=torch.float32))

print(accel_train_scaled.mean(dim=(0,1)))
print(accel_train_scaled.std(dim=(0,1)))

plt.figure(figsize=(14, 11), dpi=600)
index=3
print(theta_samples[index,:])
for i in range(n_sensors):
    ax = plt.subplot(nrows, ncols, i + 1)
    ax.plot(t, accel_test_scaled[index,:, i], linewidth=1.5)
    ax.set_title(f"Sensor {i+1}", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Time (s)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Vertical acc.", fontsize=LABEL_FONTSIZE)

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_xlim(0.0, T_total)
    ax.grid(True)

# Hide unused subplot
ax = plt.subplot(nrows, ncols, n_sensors + 1)
ax.axis("off")

plt.tight_layout()
plt.show()

import h5py

with h5py.File("rail_train_scale.h5", "w") as f:
    f.create_dataset("x", data=accel_train_scaled)

with h5py.File("rail_valid_scale.h5", "w") as f:
    f.create_dataset("x", data=accel_valid_scaled)

with h5py.File("rail_test_scale.h5", "w") as f:
    f.create_dataset("x", data=accel_test_scaled)






