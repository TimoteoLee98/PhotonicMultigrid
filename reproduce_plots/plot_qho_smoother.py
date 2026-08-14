import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import use_base_dir

use_base_dir()
import numpy as np
from matplotlib import pyplot as plt
from plot_utils import blue, orange, gray

linewidth = 1.5
markersize = 3
fontsize = 9

width_in = (5.0 * 1.105371900826446) / 2.54
height_in = (4.7 * 1.106194690265487) / 2.54
labelpad = 1

fig, ax = plt.subplots(figsize=(width_in, height_in))
fig.subplots_adjust(left=0.14, right=0.83, bottom=0.14, top=0.83)

plt.plot(np.load("measurements/qho_smoother_contribution_analog.npy"),  marker="s", color=orange, markersize=markersize,
         label="Photonic", linewidth=linewidth)
plt.plot(np.load("measurements/qho_smoother_contribution_fp64.npy"), marker="o", color=blue,   markersize=markersize,
         label="Digital",  linewidth=linewidth)
plt.hlines(1, 0, 120, colors="C7", linestyles="dashed", linewidth=linewidth)
plt.xlabel(r"Frequency modes $n$",               fontsize=fontsize, labelpad=labelpad)
plt.ylabel(r"Error spectrum $\left|c_n\right|$", fontsize=fontsize, labelpad=labelpad)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax.tick_params(axis='both', which='major', pad=labelpad)
ax.set_yticks([0, 0.5, 1.0])

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[i] for i in [1, 0]], [labels[i] for i in [1, 0]],
           fontsize=fontsize, loc="center right")
plt.savefig("plots/qho_smoother.svg", bbox_inches="tight")
print("wrote plots/qho_smoother.svg")
