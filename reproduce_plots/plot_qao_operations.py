import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import use_base_dir

use_base_dir()
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from plot_utils import blue, orange, gray, ops

fontsize = 10

counters_box    = np.load('measurements/qao_solver_counters_box_100trials.npy')
counters_8      = np.load('measurements/qao_solver_counters_ideal_100trials.npy')
counters_noisy  = np.load('measurements/qao_solver_counters_noisy8bit_100trials.npy')
counters_double = np.load('measurements/qao_solver_counters_double_100trials.npy')

result_double = np.average(counters_double, axis=0)
result_box    = np.average(counters_box,    axis=0)
result_noisy  = np.average(counters_noisy,  axis=0)

double_total       = ops[0]*result_double[0] + (ops[0]+ops[1])*result_double[1]*5
mixed_double       = ops[0]*result_box[0]
mixed_double_noisy = ops[0]*result_noisy[0]

# -----------------------------------------------------------------------
# Figure: qao_operations_final.svg
# -----------------------------------------------------------------------
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2(np.linspace(0, 1, 8)))

result_8      = np.average(counters_8, axis=0)
mixed_double8 = ops[0]*result_8[0]

bar_width = 6
positions = 16 * np.arange(4)
width  = 1.75/1.21
height = width * 4/3
fig, ax = plt.subplots(figsize=(height, width))

ax.bar(positions, [double_total, mixed_double, mixed_double_noisy, mixed_double8],
       width=bar_width, align='edge', label='Double', color=blue)
ax.set_ylabel('Digital OPs required', fontsize=fontsize)
ax.set_xticks(positions + bar_width / 2,
              ['Baseline', 'Exp.', 'Emul.', 'Ideal'])
plt.xticks(fontsize=fontsize*0.9); plt.yticks(fontsize=fontsize)
ax.set_ylim([0, 2.3e5])
ax.set_yticks([0.0e5, 1.1e5, 2.2e5])
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(5, 5))
ax.yaxis.get_offset_text().set_fontsize(10)
plt.savefig('plots/qao_operations_final.svg', dpi=1200, bbox_inches='tight')
print("wrote plots/qao_operations_final.svg")
