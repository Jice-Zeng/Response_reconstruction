#!/usr/bin/env python

import wandb

from dawgz import job, schedule
from typing import *
import csv

import sys
from pathlib import Path

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root))

from diffusion.mcs import *
from diffusion.score import *
from diffusion.utils import *

from experiments.rail_sleeper.utils import *
import matplotlib.pyplot as plt
import pickle
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


GLOBAL_CONFIG = {
    # Architecture
    'embedding': 64,
    'hidden_channels': (64, 128, 256),
    'hidden_blocks': (3, 3, 3),
    'kernel_size': 5,
    # 'kernel_size': 7,
    'activation': 'SiLU',
    # Training
    'epochs': 1000,
    'batch_size': 64,
    'optimizer': 'AdamW',
    'learning_rate': 1e-4,
    'weight_decay': 1e-3,
    'scheduler': 'linear',
} # for tune model

# @job(array=3, cpus=4, gpus=1, ram='8GB', time='06:00:00')
def train_global(i: int):
    runpath = PATH / f'Jice_test/model_checkpoints/'
    runpath.mkdir(parents=True, exist_ok=True)

    # Network
    score = make_global_score(**GLOBAL_CONFIG)
    sde = VPSDE(score, shape=(64, 11)).cpu() # originial is 32
    trainset = TrajectoryDataset(PATH / 'data/rail_train_scale.h5', window=64)  
    validset = TrajectoryDataset(PATH / 'data/rail_valid_scale.h5', window=64)
    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        **GLOBAL_CONFIG,
        device='cpu',
    )
    
    loss_best = float('inf')
    train_losses = []
    valid_losses = []

    for epoch, (loss_train, loss_valid, lr) in enumerate(generator):
        train_losses.append(loss_train)
        valid_losses.append(loss_valid)
        
        if (loss_valid < loss_best): 
            loss_best = loss_valid
            torch.save(
                score.state_dict(),
                runpath / f'rail_test_state_best.pth',
            )
            print(f'[Epoch {epoch}] Best model saved with validation loss {loss_best:.5f}')

if __name__ == '__main__':
    train_global(0)
    
# %% Scale strategy
Accel_dataset = np.load('data/sleeper_hammer_dataset.npz')['accel']
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
# %%
score_model = make_global_score(**GLOBAL_CONFIG)
state_file =  "saved model path"
score_model.load_state_dict(torch.load(state_file, map_location='cpu'))
score_model.eval()

sde = VPSDE(score_model, shape=(64, 11))
x = sde.sample((1024,), steps=64, corrections=0).cpu()
x = scaler.postprocess(x.float())

# %%
sigma = 0.05
step = 6
mear_loc = np.array([1, 3, 5, 7, 11]) - 1
with h5py.File(PATH / 'data/rail_test_scale.h5', mode='r') as f:
    x_star = torch.from_numpy(f['x'][10, :192])
    y_star = torch.normal(x_star[0:128:step, mear_loc], sigma) # partial observation
    print({'Partial observation': y_star.shape})
    print({'Target observation': x_star.shape})
    print({'Observation percentage': len(y_star)/len(x_star)})

A = lambda x: x[..., mear_loc] 
sde = VPSDE(
        GaussianScore(
            y_star,
            A=lambda x: x[..., 0:128:step, mear_loc],
            std=sigma,
            sde=VPSDE(score_model, shape=()),
        ),
        shape=x_star.shape,
    )

x = sde.sample((500,), steps=128, corrections=1, tau=0.5).cpu() # steps=256, corrections=5, tau=0.6 is ok

x_est = scaler.postprocess(x.float())
x_star_back = scaler.postprocess(x_star.float())
y_star_back = torch.normal(x_star_back[0:128:step, mear_loc], 0.05)

# %%
block = 16
context_end = 128
total_end = 192

current_end = context_end
x_current = None  
x_Posample = None

while current_end < total_end:
    horizon = min(block, total_end - current_end)
    target_end = current_end + horizon
    # redefine measurement operator to include current known trajectory
    A = lambda x: x[..., 0:current_end:step, mear_loc]

    sde = VPSDE(
        GaussianScore(
            y_star,
            A=A,
            std=sigma,
            sde=VPSDE(score_model, shape=()),
        ),
        shape=(target_end, 11),
    )

    x_sample = sde.sample((10,), steps=64, corrections=2, tau=0.5).cpu()
    # take mean or ensemble
    x_Posample = x_sample
    x_mean = x_sample.mean(0)
    # update known trajectory
    x_current = x_mean
    current_end = target_end
    y_star = x_current[0:current_end:step]

# %% prediction + true + observed
n_samples, nt, n_features = x.shape
assert n_features == 11

# time axes
T_total = 0.03  # seconds
nt = x.shape[1]

t_full = np.linspace(0.0, T_total, nt)
t_obs  = t_full[0:128:step]    # t_full[::step]
# convert observed feature indices to set for fast lookup
mear_loc = list(mear_loc)
obs_map = {feat: i for i, feat in enumerate(mear_loc)}

nrows = 4
ncols = 1  # 12 slots, 1 unused

plt.figure(figsize=(14, 11), dpi=600)
for feat in range(n_features):
    ax = plt.subplot(nrows, ncols, feat + 1)
    # --- prediction statistics ---
    slice_i = x_est[:, :, feat].cpu().numpy()
    mean = np.nanmean(slice_i, axis=0)
    std  = np.nanstd(slice_i, axis=0)

    ax.fill_between(
        t_full,
        mean - 3 * std,
        mean + 3 * std,
        color="lightblue",
        alpha=0.6,
        label="±2σ" if feat == 0 else None
    )
    # predictive mean
    ax.plot(
        t_full,
        mean,
        color="blue",
        linewidth=2.0,
        linestyle = '--',
        label="Pred mean" if feat == 0 else None
    )
    # true signal
    ax.plot(
        t_full,
        x_star_back[:, feat],
        color="red",
        linewidth=2.0,
        label="True" if feat == 0 else None
    )
    # observed data (only for selected features)
    if feat in obs_map:
        obs_idx = obs_map[feat]
        ax.scatter(
            t_obs,
            y_star_back[:, obs_idx],
            color="black",
            s=20,  # 20
            marker='s',
            zorder=5,
            label="Observed" if feat == 0 else None
        )
    ax.set_title(f"Sensor {feat+1}", fontsize=23)
    ax.set_xlabel("Time (s)", fontsize=23)
    ax.set_ylabel("Acceleration", fontsize=23)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlim(0.0, T_total)
    ax.grid(True)
# hide unused subplot
ax = plt.subplot(nrows, ncols, n_features + 1)
ax.axis("off")

# global legend (only once)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_handles = [
    Line2D([0], [0], color="blue", lw=2,linestyle='--', label="Pred mean"),
    Patch(facecolor="lightblue", alpha=0.6, label="±2σ"),
    Line2D([0], [0], color="red", lw=2, label="True"),
    Line2D(
        [0], [0],
        marker='s',
        markersize=15,
        linestyle='None',
        markerfacecolor='black',
        markeredgecolor='black',
        markeredgewidth=1.2,
        label="Observed"
    )
]
plt.figlegend(
    handles=legend_handles,
    loc="upper center",
    ncol=4,
    fontsize=25,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)  # (x, y) in figure coords
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
