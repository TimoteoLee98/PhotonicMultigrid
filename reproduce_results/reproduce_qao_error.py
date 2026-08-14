"""Reproduce the QAO operator-error data behind qao_error.svg.

The figure compares what the photonic hardware returned for H @ x against the exact
fp64 answer, over 100 fixed input vectors stored in `qao_inputs/inputs4.npy`. Three
things are computed here: the fp64 products (which must match the stored "expected"
data exactly), the same products through phoamg's built-in low-precision emulation,
and through the standalone `photonic_dot`. The latter two are compared against the
real hardware measurement, which they match statistically rather than pointwise --
each is one noisy realisation.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_ROOT, '..')))
sys.path.insert(0, _ROOT)

import numpy as np

from mg_utils import load_mg
from photonic_dot import photonic_dot

MEASUREMENTS_DIR = os.path.normpath(os.path.join(_ROOT, '..', 'reproduce_plots', 'measurements'))
SAVE_DIR = os.path.join(_ROOT, 'measurements')
NOISE_STRENGTH = 2**3 / np.sqrt(2)
SEED = 0


def relative_error(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def compare(name, candidate, reference):
    corr = np.corrcoef(candidate, reference)[0, 1]
    mae = np.mean(np.abs(candidate - reference))
    rel_err = relative_error(candidate, reference)
    print(f"  {name:22s} corr={corr:.4f}  mean_abs_err={mae:8.3f}  rel_err={rel_err:.4f}")


if __name__ == '__main__':
    np.random.seed(SEED)

    mg = load_mg(os.path.join(_ROOT, 'mg_load_files', 'qao') + os.sep)
    H = mg.As[0]

    X = np.load(os.path.join(_ROOT, 'qao_inputs', 'inputs4.npy')).reshape(-1, 120)
    num_trials = len(X)
    expected_ground_truth = np.load(os.path.join(MEASUREMENTS_DIR, 'qao_error_expected_fp64.npy'))
    measured_ground_truth = np.load(os.path.join(MEASUREMENTS_DIR, 'qao_error_analog.npy'))

    print("=== fp64 (expected) ===")
    expected_repro = np.array([H.dot(X[i]) for i in range(num_trials)]).flatten()
    print(f"  relative error vs qao_error_expected_fp64.npy: {relative_error(expected_repro, expected_ground_truth):.3e}")

    print("\n=== existing built-in low-precision emulation (noisy 8-bit) ===")
    mg.change_relax_method(True, 8, NOISE_STRENGTH, None, 8)
    emulated_repro = np.array([mg.dots[0](0, X[i]) for i in range(num_trials)]).flatten()
    compare("existing emulated", emulated_repro, measured_ground_truth)

    print("\n=== photonic_dot swap ===")

    mg.change_relax_method(True, 8, NOISE_STRENGTH, None, 8)

    def dot_low_precision_photonic(lvl, x):
        return photonic_dot(mg.As_low[lvl], x)

    # change_relax_method has already populated mg.dots and mg.As_low, so assigning
    # into the list here is independent of ordering.
    for lvl in range(len(mg.dots)):
        mg.dots[lvl] = dot_low_precision_photonic

    photonic_repro = np.array([mg.dots[0](0, X[i]) for i in range(num_trials)]).flatten()
    compare("photonic_dot", photonic_repro, measured_ground_truth)

    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(os.path.join(SAVE_DIR, 'qao_error_expected_fp64.npy'), expected_repro)
    # The `_analog` file written here is the photonic_dot *emulation* of the analog
    # hardware, not a hardware measurement; the same name under
    # reproduce_plots/measurements/ is the real device. Same name so the plotting
    # code needs no special case.
    np.save(os.path.join(SAVE_DIR, 'qao_error_analog.npy'), photonic_repro)
    print(f"\nSaved reproduced results to {SAVE_DIR}")
    print("  note: qao_error_analog.npy here is the photonic_dot emulation of the "
          "analog hardware, not a hardware measurement")
