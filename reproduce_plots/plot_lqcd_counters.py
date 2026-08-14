import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import use_base_dir

use_base_dir()
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from plot_utils import blue, orange, gray, ops_lqcd

fontsize = 10

# --- data ---
# Each counter file holds [iterations, outer_iterations] per gauge config / RHS;
# a strategy's digital-MVM cost is outer_iterations + 2*iterations. The figure reports
# operations, so each product is worth ops_lqcd of them -- all of them are on the fine
# operator.
index_mass = 0

counter_cg = np.load("measurements/lqcd_cg_counters_fp64.npy").mean()

counter_noisy = np.load("measurements/lqcd_mgcg_counters_noisy8bit.npy")
iterations_noisy, counter_high_noisy = counter_noisy[:, index_mass].mean(axis=(0, 1))
counter_high_noisy += iterations_noisy * 2

counter_quantized = np.load("measurements/lqcd_mgcg_counters_quantized8bit.npy")
iterations_quantized, counter_high_quantized = counter_quantized.mean(axis=(0, 1))
counter_high_quantized += iterations_quantized * 2

# The mixed setup phase used by the noisy strategy costs 42*8 digital MVMs; the
# standard setup phase used by the quantized strategy is photonic-only and costs none.
setup_high_noisy = 42 * 8

# -----------------------------------------------------------------------
# Figure: lqcd_operations.svg
# -----------------------------------------------------------------------
bar_width = 4
positions = 8 * np.arange(3)

width  = 2.3622/1.3
height = width * 4/3
fig, ax = plt.subplots(figsize=(height, width))

y_high = [counter_cg * ops_lqcd,
          (counter_high_noisy + setup_high_noisy) * ops_lqcd,
          counter_high_quantized * ops_lqcd]
ax.bar(positions, y_high, width=bar_width, align='edge', color=blue)

ax.set_ylim([0, 1.1 * max(y_high)])
ax.set_ylabel('Digital OPs required', fontsize=fontsize)
ax.set_xticks(positions + bar_width / 2, ['', '', ''], fontsize=fontsize*0.9)
plt.yticks(fontsize=fontsize)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(9, 9))
ax.yaxis.get_offset_text().set_fontsize(10)
plt.savefig('plots/lqcd_operations.svg', dpi=1200, bbox_inches='tight')
print("wrote plots/lqcd_operations.svg")
