import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import use_base_dir

use_base_dir()
import numpy as np
from matplotlib import pyplot as plt
from plot_utils import blue, orange, gray, ops_pc

fontsize = 10

# --- data ---
r_array            = np.load('measurements/pc_hybrid_mpmgp_residual_checkpoints_8x5.npy')
residual_norms_low = np.load('measurements/pc_analog_richardson_residual_251iters.npy')

cum_count = []
r_norms   = []
counter   = 0
cum_count.append(counter)
for array in r_array:
    for i in range(len(array) - 1):
        r_norms.append(array[i])
        counter += 5
        cum_count.append(counter)
    counter += 1
r_norms.append(r_array[-1][-1])
r_norms   = np.array(r_norms)
cum_count = np.array(cum_count)

# Both traces count fine-operator matrix-vector products; the figure reports
# operations, so each is worth ops_pc[0] of them -- as in plot_pc_operations.py.
ops = ops_pc[0]
analog_mvms = np.arange(len(residual_norms_low[:175]))

# --- plot ---
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2(np.linspace(0, 1, 8)))

width  = 1.75/1.21
height = width * 4/3
fig, ax = plt.subplots(figsize=(height, width))

linewidth = 2
plt.plot(analog_mvms * ops, residual_norms_low[:175],
         label='Analog', linewidth=linewidth, color=orange, linestyle="dashed")
plt.plot(cum_count * ops, r_norms, label='Hybrid', linewidth=linewidth, color=orange)

plt.yscale("log")
plt.xlabel('Total Number of OPs', fontsize=fontsize)
plt.ylabel('Residual Norm',        fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.hlines(1e-10, 0, len(residual_norms_low[:175]) * ops,
           linestyles='dashed', colors=blue, linewidth=linewidth*0.6)

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[i] for i in [0, 1]], [labels[i] for i in [0, 1]],
           loc='center left', fontsize=fontsize*0.8, handlelength=2.20,
           bbox_to_anchor=(0.44, 0.69))

# The "1e6" x-axis multiplier defaults to the same row as the "Total Number of
# OPs" label and collides with it in this narrow a figure. Matplotlib
# recomputes the built-in offset text's position on every draw, in display
# pixels, so it can't just be given a new transform -- draw it as a separate
# text object instead, anchored to the axes' own coordinates, just below the
# plot's bottom-right corner.
fig.canvas.draw()
offset_label = ax.xaxis.get_offset_text().get_text()
ax.xaxis.get_offset_text().set_visible(False)
ax.text(1.0, -0.05, offset_label, transform=ax.transAxes,
        ha='right', va='top', fontsize=fontsize*0.8)

plt.savefig('plots/pc_residual_norms.svg', dpi=1200, bbox_inches='tight')
print("wrote plots/pc_residual_norms.svg")
