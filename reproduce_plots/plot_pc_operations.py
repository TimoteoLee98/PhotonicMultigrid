import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import use_base_dir

use_base_dir()
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from plot_utils import blue, orange, gray, ops_pc

fontsize = 10
counter = np.average(np.load('measurements/pc_solver_counters_100trials.npy'), axis=0)

# The three digital settings, [iterations, restarts] per row, in the order
# fp64 / ideal 8-bit / noisy 8-bit. Written by reproduce_pc_counters.py.
(it_double, re_double), (it_8, re_8), (it_noisy, re_noisy) = np.load(
    'measurements/pc_solver_counters_digital_3settings.npy')

# Counters are matrix-vector products; the figure reports operations, so each is worth
# ops_pc[0] of them. Every product counted here is on the fine operator. Per solve the
# digital cost is restarts + iterations, and each inner iteration applies the
# preconditioner once, costing 5 products (numPre=4 smoothing steps plus the residual).
# For fp64 those 5 are digital too, so that bar carries them instead of a photonic one.
ops = ops_pc[0]

double_double = (re_double + it_double + it_double*5) * ops
double_8      = (re_8 + it_8) * ops;         low_8     = it_8*5 * ops
double_noisy  = (re_noisy + it_noisy) * ops; low_noisy = it_noisy*5 * ops
double_box    = (counter[1] + counter[0]) * ops; low_box = counter[0]*5 * ops

bar_width = 6
positions = 16 * np.arange(4)
width  = 1.75/1.21
height = width * 4/3
fig, ax = plt.subplots(figsize=(height, width))

ax.bar(positions, [double_double, double_box, double_noisy, double_8],
       width=bar_width, align='edge', color=blue)
ax.set_ylabel('Digital OPs required', fontsize=fontsize)
ax.set_xticks(positions + bar_width / 2,
              ['Baseline', 'Exp.', 'Emul.', 'Ideal'], fontsize=fontsize*0.9)
plt.yticks(fontsize=fontsize)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(6, 6))
ax.yaxis.get_offset_text().set_fontsize(10)
plt.savefig('plots/pc_operations.svg', dpi=1200, bbox_inches='tight')
print("wrote plots/pc_operations.svg")
