"""Reproduce the parallel-capacitors solver counters behind pc_operations.svg.

Solves the 2-D Poisson "parallel capacitors" problem with multigrid-preconditioned
iterative refinement at four settings -- fp64, noiseless 8-bit, 8-bit with noise, and
a `photonic_dot` swap repeated over NUM_TRIALS stochastic trials -- and reports the
inner-CG iteration count each costs. Also saves one representative field solution,
the data behind pc_field.svg.

The real-hardware counters are loaded for comparison; our noise model does not
reproduce them (36 against 49.8), which is a documented calibration gap rather than a
bug -- see README.md.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_ROOT, '..')))
sys.path.insert(0, _ROOT)

import numpy as np

from mg_utils import load_mg
from photonic_dot import photonic_dot
from pc_problem import build_capacitor_rhs

MEASUREMENTS_DIR = os.path.normpath(os.path.join(_ROOT, '..', 'reproduce_plots', 'measurements'))
SAVE_DIR = os.path.join(_ROOT, 'measurements')
NOISE_STRENGTH = 2**3 / np.sqrt(2)
SEED = 0

# Trials of the stochastic photonic solve; the count the output filename records.
NUM_TRIALS = 100

# The three digital settings pc_operations.svg plots next to the photonic bar, stored
# as one (3, 2) array of [iterations, restarts] in this row order.
DIGITAL_SETTINGS_FILE = 'pc_solver_counters_digital_3settings.npy'
DIGITAL_SETTING_NAMES = ('fp64', 'ideal 8-bit', 'noisy 8-bit')
SOLVE_KWARGS = dict(rtol=1e-12, rtol_internal=1e-3, maxiter=4)


def report(tag, counters):
    counters = np.array(counters)
    m, s = counters[:, 0].mean(), counters[:, 0].std()
    print(f"  {tag:28s} mean_iterations={m:7.2f}  std={s:6.2f}")
    return counters


if __name__ == '__main__':
    np.random.seed(SEED)

    mg = load_mg(os.path.join(_ROOT, 'mg_load_files', 'parallel_capacitors') + os.sep)
    b = build_capacitor_rhs(Nx=64, Ny=64)
    b_norm = np.linalg.norm(b)

    box_ground_truth = np.load(os.path.join(MEASUREMENTS_DIR, 'pc_solver_counters_100trials.npy'))
    print(f"[real hardware, 100 trials] mean_iterations={box_ground_truth[:, 0].mean():.2f}  "
          f"std={box_ground_truth[:, 0].std():.2f}")

    print("\n=== fp64 ===")
    mg.change_relax_method(False)
    M = mg.aspreconditioner(numPre=4, numPost=0, low_precision_residual=True)
    _, counters_double, rS = mg.solve_mixed(mg.As[0], b, M=M, **SOLVE_KWARGS)
    print(f"  counters=[iterations, restarts]={counters_double}  "
          f"final relative residual={rS / b_norm:.3e}")

    print("\n=== Ideal 8-bit (noiseless quantization, no noise) ===")
    mg.change_relax_method(True, 8, None, None, 8)
    M = mg.aspreconditioner(numPre=4, numPost=0, low_precision_residual=True)
    _, counters_ideal, rS = mg.solve_mixed(mg.As[0], b, M=M, **SOLVE_KWARGS)
    print(f"  counters=[iterations, restarts]={counters_ideal}  "
          f"final relative residual={rS / b_norm:.3e}")

    print("\n=== existing built-in noisy-8-bit emulation ===")
    mg.change_relax_method(True, 8, NOISE_STRENGTH, None, 8)
    M = mg.aspreconditioner(numPre=4, numPost=0, low_precision_residual=True)
    _, counters_emulated, rS = mg.solve_mixed(mg.As[0], b, M=M, **SOLVE_KWARGS)
    print(f"  counters=[iterations, restarts]={counters_emulated}  "
          f"final relative residual={rS / b_norm:.3e}")

    # The three digital settings above are what pc_operations.svg plots alongside the
    # photonic bar. They are deterministic -- one solve each, no trial averaging -- so
    # the comparison against the cached file is an exact-match check.
    counters_digital = np.array([counters_double, counters_ideal, counters_emulated])
    cached_digital = np.load(os.path.join(MEASUREMENTS_DIR, DIGITAL_SETTINGS_FILE))
    print("\n=== digital settings: reproduced vs cached, [iterations, restarts] ===")
    for tag, repro, cached in zip(DIGITAL_SETTING_NAMES, counters_digital, cached_digital):
        status = 'matches' if np.array_equal(repro, cached) else 'MISMATCH'
        print(f"  {tag:12s} reproduced={repro.tolist()}  cached={cached.tolist()}  -> {status}")

    print("\n=== photonic_dot swap, 100 trials ===")

    def dot_photonic(lvl, x):
        return photonic_dot(mg.As[lvl], x)

    counters_photonic = []
    u_representative = None
    for trial in range(NUM_TRIALS):
        mg.change_relax_method(False)
        mg.dots[0] = dot_photonic
        M = mg.aspreconditioner(numPre=4, numPost=0, low_precision_residual=True)
        u, counters, rS = mg.solve_mixed(mg.As[0], b, M=M, **SOLVE_KWARGS)
        counters_photonic.append(counters)
        if trial == 0:
            u_representative = u.copy()

    print("\n=== summary (mean +/- std of total inner-CG iteration count) ===")
    report("real hardware (box)", box_ground_truth)
    report("fp64", [counters_double] * 1)
    report("ideal 8-bit (noiseless)", [counters_ideal] * 1)
    report("existing emulated (noisy 8-bit)", [counters_emulated] * 1)
    report("photonic_dot (100 trials)", counters_photonic)

    print("\n=== PC field solution: one representative photonic_dot trial vs ground truth ===")
    field_ground_truth = np.load(os.path.join(MEASUREMENTS_DIR, 'pc_field_solution_64x64.npy'))
    true_residual = np.linalg.norm(b - mg.As[0].dot(u_representative)) / np.linalg.norm(b)
    print(f"  reproduced solution true relative residual: {true_residual:.3e}")
    print(f"  ||u_reproduced||={np.linalg.norm(u_representative):.4f}  "
          f"||u_ground_truth||={np.linalg.norm(field_ground_truth):.4f}")
    corr = np.corrcoef(u_representative, field_ground_truth)[0, 1]
    rel_err = np.linalg.norm(u_representative - field_ground_truth) / np.linalg.norm(field_ground_truth)
    print(f"  corr(u_reproduced, u_ground_truth)={corr:.4f}  rel_err={rel_err:.4f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    # Digital reproduction, software in both places -- no hardware involved.
    np.save(os.path.join(SAVE_DIR, DIGITAL_SETTINGS_FILE), counters_digital)
    # The next two are the photonic_dot *emulation* of the analog hardware, not hardware
    # measurements; the same names under reproduce_plots/measurements/ are the real
    # device. Same names so the plotting code needs no special case.
    np.save(os.path.join(SAVE_DIR, 'pc_solver_counters_100trials.npy'), np.array(counters_photonic))
    np.save(os.path.join(SAVE_DIR, 'pc_field_solution_64x64.npy'), u_representative)
    print(f"\nSaved reproduced results to {SAVE_DIR}")
    print("  note: these are the photonic_dot emulation of the analog hardware, not "
          "hardware measurements")
