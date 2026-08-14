import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import use_base_dir

use_base_dir()
import numpy as np
from matplotlib import pyplot as plt
from plot_utils import blue, orange, gray, ops, c2, c3, c4

fontsize = 10
ResidualNormsHistory_noisy = np.load('measurements/qao_residual_norm_history_noisy_3modes.npy')
convergence_threshold = 1e-10

x = np.arange(len(ResidualNormsHistory_noisy))
x = (ops[0] + ops[1]) * 6 * x + ops[0] * 2
x[0] = 0

width  = 1.75/1.21
height = width * 4/3
fig, ax = plt.subplots(figsize=(height, width))

linewidth = 2
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Paired(np.linspace(0, 1, 8)))


def convergence_cutoff(residuals, threshold):
    below = np.where(residuals < threshold)[0]
    return int(below[0]) + 1 if len(below) else len(residuals)


for i, color in [(2, c2), (1, c3), (0, c4)]:
    final = convergence_cutoff(ResidualNormsHistory_noisy[:, i], convergence_threshold)
    plt.plot(x[:final], ResidualNormsHistory_noisy[:final, i],
             label=r"$\varphi_" + str(i) + r"$", linewidth=linewidth, color=color)

plt.yscale('log')
plt.xlabel('Total Number of OPs', fontsize=fontsize)
plt.ylabel('Residual Norm',       fontsize=fontsize)
ax.set_xticks([0, 0.5e5, 1e5], ['0.0', '0.5', '1.0'], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize*0.8, loc='upper right', handlelength=0.75)
plt.hlines(1e-10, 0, np.max(x), linestyles='dashed', colors=blue, linewidth=linewidth*0.6)

# x values are in units of 1e5; state that below the plot's bottom-right corner,
# as in plot_pc_residuals.py.
ax.text(1.0, -0.05, '1e5', transform=ax.transAxes,
        ha='right', va='top', fontsize=fontsize*0.8)

plt.savefig('plots/qao_residuals.svg', dpi=1200, bbox_inches='tight')
print("wrote plots/qao_residuals.svg")
