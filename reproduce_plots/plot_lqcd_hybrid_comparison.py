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
# Both curves cover the five masses this figure plots.
# Total MVMs of the mixed-precision preconditioned multigrid solver, one per mass.
result = np.load("measurements/lqcd_mppmg_total_mvms_5masses.npy")

# Iterative-refinement counters, columns [digital, photonic] MVMs, same masses.
# Reproduced by reproduce_results/reproduce_lqcd_hybrid_comparison.py.
y_double, y_low = np.load("measurements/lqcd_ir_counters_5masses.npy").T
y = y_double + y_low

# The figure reports operations rather than products. Both curves count fine-operator
# products only, digital and photonic alike, so each is worth ops_lqcd operations.
result = result * ops_lqcd
y = y * ops_lqcd

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2(np.linspace(0, 1, 8)))

# The two curves' lightest mass differs: the IR sweep runs at -0.06, the MPPMG one at
# the near-critical -0.06284 that reproduce_lqcd_counters.py uses.
m  = [1, 0.5, 0.25, 0.1, -0.06]
m2 = [1, 0.5, 0.25, 0.1, -0.063]

width  = 2.3622/1.3
height = width * 4/3
fig, ax = plt.subplots(figsize=(height, width))

markersize = 4
linewidth  = 2

ax.plot(m,  y,      label="Hybrid (IR method)", c='C7',   marker='o',
        linewidth=linewidth, markersize=markersize)
ax.plot(m2, result, label="Hybrid (our work)",  c=orange, marker='s',
        linewidth=linewidth, markersize=markersize)

ax.set_xlabel(r'Fermionic mass $\mathregular{m}$', fontsize=fontsize)
ax.set_ylabel("Total Number of OPs",               fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)

lines1, labels1 = ax.get_legend_handles_labels()
ax.legend(lines1, labels1, loc="upper right", fontsize=fontsize, handlelength=1.5)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(4, 4))
ax.yaxis.get_offset_text().set_fontsize(10)
plt.yscale("log")
plt.savefig('plots/lqcd_hybrid_comparison.svg', dpi=1200, bbox_inches='tight')
print("wrote plots/lqcd_hybrid_comparison.svg")
