import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import use_base_dir

use_base_dir()
import numpy as np
from matplotlib import pyplot as plt
from plot_utils import blue, orange, gray

box_measured = np.load('measurements/qao_error_analog.npy')
box_expected = np.load('measurements/qao_error_expected_fp64.npy')

# Both axes share one divisor -- the largest absolute value either array reaches -- so
# the data lands inside the fixed [-1, 1] ideal line drawn below.
max_overall = np.max([np.max(np.abs(box_measured)), np.max(np.abs(box_expected))])

normalized_box_measured = box_measured / max_overall
normalized_box_expected = box_expected / max_overall

linewidth = 1.5
markersize = 4
fontsize = 9

width_in  = (5.0 * 1.105371900826446) / 2.54
height_in = (4.7 * 1.106194690265487) / 2.54
labelpad = 1

fig, ax = plt.subplots(figsize=(width_in, height_in))
fig.subplots_adjust(left=0.14, right=0.83, bottom=0.14, top=0.83)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2(np.linspace(0, 1, 8)))

plt.plot(normalized_box_expected, normalized_box_measured,
         linestyle='none', marker='.', label='Photonic',
         markersize=markersize, linewidth=linewidth, color=orange, rasterized=True)
plt.plot([-1, 1], [-1, 1], linestyle='dashed', color='black', label='Ideal', linewidth=linewidth)
plt.xlabel('Expected', fontsize=fontsize, labelpad=labelpad)
plt.ylabel('Measured', fontsize=fontsize, labelpad=labelpad)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax.tick_params(axis='both', which='major', pad=labelpad)
ax.set_yticks([-1, 0, 1])

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[i] for i in [1, 0]], [labels[i] for i in [1, 0]],
           fontsize=fontsize, loc="upper left", handlelength=1.5)
plt.savefig("plots/qao_error.svg", bbox_inches="tight", dpi=2000)
print("wrote plots/qao_error.svg")
