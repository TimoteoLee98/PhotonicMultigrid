"""Reproduce the QAO solver iteration counters behind qao_operations_final.svg.

LOBPCG is run once per starting block in `qao_inputs/X_100.npy`, preconditioned by
the QAO multigrid hierarchy, at four precision settings: fp64, noiseless 8-bit,
8-bit with noise through phoamg's built-in emulation, and the same with the
preconditioner's matrix-vector product swapped for `photonic_dot`. What the figure
plots is the operator-application count each setting costs.

The last run also keeps the first trial's residual-norm history, which is the data
behind qao_residuals.svg.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_ROOT, '..')))
sys.path.insert(0, _ROOT)

import numpy as np

from phoamg.eigensolver.lobpcg import lobpcg
from mg_utils import load_mg
from photonic_dot import photonic_dot

MEASUREMENTS_DIR = os.path.normpath(os.path.join(_ROOT, '..', 'reproduce_plots', 'measurements'))
SAVE_DIR = os.path.join(_ROOT, 'measurements')
NOISE_STRENGTH = 2**3 / np.sqrt(2)
SEED = 0

LOBPCG_KWARGS = dict(
    tol=1e-10, largest=False, maxiter=2000, restartControl=20,
    verbosityLevel=0, retLambdaHistory=True, retResidualNormsHistory=True,
)


def run_lobpcg_sweep(tag, H, X_all, M, keep_history=False):
    """Solve once per starting block in X_all; return the per-trial counters.

    With keep_history, also return the residual-norm history of the first trial,
    which is what the QAO residual figure plots.
    """
    counters_list = []
    first_history = None
    for i in range(len(X_all)):
        _, _, _, residual_norms, counters = lobpcg(H, X_all[i], M=M, **LOBPCG_KWARGS)
        counters_list.append(counters)
        if i == 0:
            first_history = np.array(residual_norms)
    counters = np.array(counters_list)
    print(f"[{tag}] mean={counters.mean(axis=0)}  std={counters.std(axis=0)}")
    return (counters, first_history) if keep_history else counters


if __name__ == '__main__':
    np.random.seed(SEED)

    mg = load_mg(os.path.join(_ROOT, 'mg_load_files', 'qao') + os.sep)
    mg.coefficients = np.load(os.path.join(_ROOT, 'qao_inputs', 'coefficients.npy'))
    H = mg.As[0]
    X_all = np.load(os.path.join(_ROOT, 'qao_inputs', 'X_100.npy'))

    box_ground_truth = np.load(os.path.join(MEASUREMENTS_DIR, 'qao_solver_counters_box_100trials.npy'))
    print(f"[box (real hardware)] mean={box_ground_truth.mean(axis=0)}  std={box_ground_truth.std(axis=0)}\n")

    print("=== Run: fp64 ===")
    mg.change_relax_method(False)
    M = mg.aspreconditioner(numPre=2, numPost=2, low_precision_residual=True)
    counters_double = run_lobpcg_sweep('double (fp64)', H, X_all, M)

    print("\n=== Run: ideal 8-bit, noiseless ===")
    mg.change_relax_method(True, 8, None, None, 8)
    M = mg.aspreconditioner(numPre=2, numPost=2, low_precision_residual=True)
    counters_ideal = run_lobpcg_sweep('ideal 8-bit (noiseless)', H, X_all, M)

    print("\n=== Run: noisy 8-bit, phoamg's built-in emulation ===")
    mg.change_relax_method(True, 8, NOISE_STRENGTH, None, 8)
    M = mg.aspreconditioner(numPre=4, numPost=0, low_precision_residual=True)
    counters_noisy = run_lobpcg_sweep('noisy (existing emulated)', H, X_all, M)

    print("\n=== Run: photonic_dot swap (same settings as 'noisy' run) ===")

    mg.change_relax_method(True, 8, NOISE_STRENGTH, None, 8)

    def dot_low_precision_photonic(lvl, x):
        return photonic_dot(mg.As_low[lvl], x)

    # change_relax_method has already populated mg.dots and mg.As_low, so assigning
    # into the list here is independent of ordering.
    for lvl in range(len(mg.dots)):
        mg.dots[lvl] = dot_low_precision_photonic

    M = mg.aspreconditioner(numPre=4, numPost=0, low_precision_residual=True)

    counters_photonic, representative_history = run_lobpcg_sweep(
        'photonic_dot', H, X_all, M, keep_history=True)

    print("\n=== summary (mean +/- std of total iteration count, column 0) ===")
    for tag, counters in [
        ('box (real hardware)', box_ground_truth),
        ('double (fp64)', counters_double),
        ('ideal 8-bit (noiseless)', counters_ideal),
        ('noisy (existing emulated)', counters_noisy),
        ('photonic_dot', counters_photonic),
    ]:
        m, s = counters[:, 0].mean(), counters[:, 0].std()
        print(f"  {tag:28s} mean={m:7.2f}  std={s:7.2f}  sem={s/np.sqrt(len(counters)):.2f}")

    print("\n=== QAO residual norm history: one representative photonic_dot trial vs ground truth ===")
    history_ground_truth = np.load(os.path.join(MEASUREMENTS_DIR, 'qao_residual_norm_history_noisy_3modes.npy'))
    print(f"  reproduced shape={representative_history.shape}  ground truth shape={history_ground_truth.shape}")
    print(f"  reproduced:   start={representative_history[0]}  end={representative_history[-1]}")
    print(f"  ground truth: start={history_ground_truth[0]}  end={history_ground_truth[-1]}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(os.path.join(SAVE_DIR, 'qao_solver_counters_double_100trials.npy'), counters_double)
    np.save(os.path.join(SAVE_DIR, 'qao_solver_counters_ideal_100trials.npy'), counters_ideal)
    np.save(os.path.join(SAVE_DIR, 'qao_solver_counters_noisy8bit_100trials.npy'), counters_noisy)
    # The next two are the photonic_dot *emulation* of the analog hardware, not hardware
    # measurements; the same names under reproduce_plots/measurements/ are the real
    # device. Same names so the plotting code needs no special case.
    np.save(os.path.join(SAVE_DIR, 'qao_solver_counters_box_100trials.npy'), counters_photonic)
    np.save(os.path.join(SAVE_DIR, 'qao_residual_norm_history_noisy_3modes.npy'), representative_history)
    print(f"\nSaved reproduced results to {SAVE_DIR}")
    print("  note: qao_solver_counters_box_100trials.npy and "
          "qao_residual_norm_history_noisy_3modes.npy here are the photonic_dot\n"
          "        emulation of the analog hardware, not hardware measurements")
