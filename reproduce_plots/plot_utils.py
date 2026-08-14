import argparse
import os

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman", "Times", "Liberation Serif", "DejaVu Serif",
]
plt.rcParams["mathtext.fontset"] = "cm"

_ROOT = os.path.dirname(os.path.abspath(__file__))


def use_base_dir():
    """Handle --base-dir: work in it, and make sure it has a plots/ directory.

    Every plot script reads `measurements/` and writes `plots/` relative to the
    working directory, so pointing this at reproduce_results/ renders the same
    figures from freshly reproduced data instead of the cached data here.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default=_ROOT,
                        help="Directory containing measurements/ and plots/ "
                             "(default: the directory this script lives in)")
    args = parser.parse_args()
    os.chdir(args.base_dir)
    os.makedirs("plots", exist_ok=True)

# Brand colors
blue   = tuple(c/255 for c in (69,  97, 151))
orange = tuple(c/255 for c in (161, 73,  43))
gray   = tuple(c/255 for c in (114, 114, 114))

# Operations per matrix-vector product, per multigrid level. This is the formula the
# multigrid object uses (phoamg/multigrid/multigrid.py): 2*nnz - n, i.e. nnz multiplies
# plus nnz - n additions.
ops = [596, 196, 66]        # QAO, three levels

# Parallel capacitors: mg.ops of the stored hierarchy, [fine, coarse].
# n = 4096, nnz = 19968 and n = 686, nnz = 5936.
ops_pc = [35840, 11186]

# Lattice QCD: the fine operator is A = D D^dagger, and it is applied as two
# matrix-vector products rather than as the formed product, which is how lattice QCD
# codes work -- forming D D^dagger doubles the stencil and discards the structure.
# D has n = 32768 rows with a 6-point stencil, so one application is
# 2*196608 - 32768 = 360448, and A costs twice that. (TwoGridQCD does build A_high
# explicitly for convenience; applying that single 12-point operator would cost 753664,
# one extra addition per row.)
ops_lqcd = 2 * 360448

# Eigenfunction palette
c2 = "#FABF7B"
c3 = "#F08F6E"
c4 = "#E05C5C"


def create_things_for_plot(Nx, m=1, x_min=-6, x_max=6, omega=1, l=0, minus_sign=False):
    x = np.linspace(x_min, x_max, Nx)
    if minus_sign:
        V = -(0.5*m**2 * omega**2 * x**2) + (l*x**4)
    else:
        V =  (0.5*m**2 * omega**2 * x**2) + (l*x**4)
    return x, V
