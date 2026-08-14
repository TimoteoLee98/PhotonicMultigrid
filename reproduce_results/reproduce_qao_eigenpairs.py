"""Reproduce the QAO eigenvalues and eigenfunctions behind qao_eigenfunction.svg.

Solves the anharmonic-oscillator eigenproblem with LOBPCG twice -- once in fp64 and
once with the preconditioner running on `photonic_dot` -- and compares both against
the stored eigenpairs. The point of the comparison is that noise costs iterations but
not accuracy: the operator itself stays exact, so both runs converge to the same
eigenpairs to machine precision.
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


def report(tag, eigenvalues, eigenvectors, counters, eigenvalues_gt, eigenvectors_gt):
    print(f"\n--- {tag} ---")
    print(f"  counters={counters}")
    print(f"  eigenvalues reproduced:  {eigenvalues}")
    print(f"  eigenvalues ground truth: {eigenvalues_gt}")
    print(f"  max abs eigenvalue error: {np.max(np.abs(eigenvalues - eigenvalues_gt)):.3e}")
    for i in range(3):
        v = eigenvectors[:, i]
        v_gt = eigenvectors_gt[:, i]
        if np.dot(v, v_gt) < 0:
            v = -v
        rel_err = np.linalg.norm(v - v_gt) / np.linalg.norm(v_gt)
        corr = np.corrcoef(v, v_gt)[0, 1]
        print(f"  mode {i}: corr={corr:.6f}  rel_err={rel_err:.3e}")


if __name__ == '__main__':
    np.random.seed(SEED)

    mg = load_mg(os.path.join(_ROOT, 'mg_load_files', 'qao') + os.sep)
    mg.coefficients = np.load(os.path.join(_ROOT, 'qao_inputs', 'coefficients.npy'))
    H = mg.As[0]
    X_all = np.load(os.path.join(_ROOT, 'qao_inputs', 'X_100.npy'))

    eigenvalues_gt = np.load(os.path.join(MEASUREMENTS_DIR, 'qao_eigenvalues_n0to2.npy'))
    eigenvectors_gt = np.load(os.path.join(MEASUREMENTS_DIR, 'qao_eigenvectors_n0to2_grid120.npy'))

    mg.change_relax_method(False)
    M = mg.aspreconditioner(numPre=2, numPost=2, low_precision_residual=True)
    eigenvalues, eigenvectors, _, _, counters = lobpcg(H, X_all[0], M=M, **LOBPCG_KWARGS)
    report("fp64", eigenvalues, eigenvectors, counters, eigenvalues_gt, eigenvectors_gt)

    mg.change_relax_method(True, 8, NOISE_STRENGTH, None, 8)

    def dot_low_precision_photonic(lvl, x):
        return photonic_dot(mg.As_low[lvl], x)

    # change_relax_method has already populated mg.dots and mg.As_low, so assigning
    # into the list here is independent of ordering.
    for lvl in range(len(mg.dots)):
        mg.dots[lvl] = dot_low_precision_photonic

    M = mg.aspreconditioner(numPre=4, numPost=0, low_precision_residual=True)
    eigenvalues_ph, eigenvectors_ph, _, _, counters_ph = lobpcg(H, X_all[0], M=M, **LOBPCG_KWARGS)
    report("photonic_dot (used inside the preconditioner only; H itself stays exact)",
           eigenvalues_ph, eigenvectors_ph, counters_ph, eigenvalues_gt, eigenvectors_gt)

    os.makedirs(SAVE_DIR, exist_ok=True)
    # Both files are the photonic_dot *emulation* of the analog hardware, not hardware
    # measurements; the same names under reproduce_plots/measurements/ are the real
    # device. Same names so the plotting code needs no special case.
    np.save(os.path.join(SAVE_DIR, 'qao_eigenvalues_n0to2.npy'), eigenvalues_ph)
    np.save(os.path.join(SAVE_DIR, 'qao_eigenvectors_n0to2_grid120.npy'), eigenvectors_ph)
    print(f"\nSaved reproduced results (photonic_dot-preconditioned run) to {SAVE_DIR}")
    print("  note: these are the photonic_dot emulation of the analog hardware, not "
          "hardware measurements")
