"""Reproduce the QHO smoother error spectrum behind qho_smoother.svg.

The figure shows how much of each eigenmode of the fine operator survives four
smoothing steps, digital versus photonic. The recipe:

  1. Diagonalise the fine operator A of the QHO multigrid hierarchy, eigenvalues
     ascending. A is symmetric, so its eigenvectors are orthonormal and form the
     basis transformation: the spectrum of any error vector e is |V^T e|.
  2. Solve A x = b to rtol=1e-12 for the stored right-hand side `random_b.npy`,
     normalised to unit norm, giving the exact solution.
  3. Start from an initial guess whose error is every eigenvector summed with
     coefficient 1, so the initial spectrum is flat at 1.
  4. Apply the smoother four times and read off the surviving spectrum.
  5. Repeat with the smoother's matrix-vector product swapped for `photonic_dot`.

Richardson smoothing propagates the error as e <- (I - c*A) e, so the digital
curve is analytically |(1 - c*lambda_n)^4| with c the hierarchy's stored
coefficient. That gives an exact expected answer to check against, independent
of this script. The photonic curve has no closed form -- quantization and
detector noise scatter energy across modes.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_ROOT, '..')))
sys.path.insert(0, _ROOT)

import numpy as np

from phoamg.linear_solver.cg import cg
from phoamg.multigrid.multigrid import MultiGrid
from photonic_dot import photonic_dot

MEASUREMENTS_DIR = os.path.normpath(os.path.join(_ROOT, '..', 'reproduce_plots', 'measurements'))
SAVE_DIR = os.path.join(_ROOT, 'measurements')
GRID_DIR = os.path.join(_ROOT, 'mg_load_files', 'qho_plots') + os.sep
RHS_FILE = os.path.join(_ROOT, 'qho_inputs', 'random_b.npy')

NUM_SMOOTHING_STEPS = 4
RTOL = 1e-12
BIT_PRECISION = 8
BIT_PRECISION_ADC = 8
NOISE_STRENGTH = 2**3 / np.sqrt(2)
SEED = 0


def error_spectrum(eigenvectors, error):
    """Decompose an error vector over the eigenbasis and return |c_n|."""
    return np.abs(eigenvectors.T @ error)


def smooth(mg, x0, b):
    """Apply the smoother NUM_SMOOTHING_STEPS times on the finest level."""
    x = x0.copy()
    mg.polynomial(0, x, b, iterations=NUM_SMOOTHING_STEPS)
    return x


if __name__ == '__main__':
    np.random.seed(SEED)

    mg = MultiGrid()
    mg.load(GRID_DIR)
    A = mg.As[0]
    coefficient = mg.coefficients[0]
    print(f"Fine operator {A.shape}, smoother coefficient {coefficient:.9e}")

    # 1. Eigenpairs of the fine operator, ascending. A is symmetric, so eigh
    #    gives an orthonormal basis and sorts the eigenvalues for us.
    eigenvalues, eigenvectors = np.linalg.eigh(A.toarray())
    print(f"Lowest eigenvalues  {np.round(eigenvalues[:5], 6)}")
    print(f"Highest eigenvalues {np.round(eigenvalues[-3:], 4)}")

    # 2. Exact solution of the fine system for the stored right-hand side,
    #    normalised to unit norm.
    b = np.load(RHS_FILE)
    b = b / np.linalg.norm(b)
    x_exact, iterations = cg(A, b, rtol=RTOL, maxiter=10000)
    print(f"Solved A x = b in {iterations} CG iterations, relative residual "
          f"{np.linalg.norm(b - A.dot(x_exact)) / np.linalg.norm(b):.3e}")

    # 3. Initial guess whose error is every eigenvector with coefficient 1.
    initial_error = eigenvectors.sum(axis=1)
    x0 = x_exact - initial_error
    print(f"Initial error spectrum: min {error_spectrum(eigenvectors, initial_error).min():.6f}, "
          f"max {error_spectrum(eigenvectors, initial_error).max():.6f}")

    # 4. Digital (fp64) smoother.
    mg.change_relax_method(False)
    contribution_high = error_spectrum(eigenvectors, x_exact - smooth(mg, x0, b))

    # 5. Photonic smoother: let change_relax_method populate mg.dots and mg.As_low,
    #    then swap the matrix-vector product into that list.
    mg.change_relax_method(True, BIT_PRECISION, NOISE_STRENGTH, None, BIT_PRECISION_ADC)

    def dot_low_precision_photonic(lvl, x):
        return photonic_dot(mg.As_low[lvl], x)

    for lvl in range(len(mg.dots)):
        mg.dots[lvl] = dot_low_precision_photonic

    contribution_low = error_spectrum(eigenvectors, x_exact - smooth(mg, x0, b))

    # --- Checks --------------------------------------------------------------
    analytic = np.abs((1 - coefficient * eigenvalues) ** NUM_SMOOTHING_STEPS)
    print(f"\nDigital vs analytic |(1 - c*lambda)^{NUM_SMOOTHING_STEPS}|: "
          f"max abs difference {np.abs(contribution_high - analytic).max():.3e}")

    cached_high = np.load(os.path.join(MEASUREMENTS_DIR, 'qho_smoother_contribution_fp64.npy'))
    cached_low = np.load(os.path.join(MEASUREMENTS_DIR, 'qho_smoother_contribution_analog.npy'))

    print("\nReproduced vs cached")
    print(f"  digital  max abs difference {np.abs(contribution_high - cached_high).max():.3e}   "
          f"correlation {np.corrcoef(contribution_high, cached_high)[0, 1]:.6f}")
    print(f"  photonic mean {contribution_low.mean():.4f} (cached {cached_low.mean():.4f}), "
          f"range [{contribution_low.min():.4f}, {contribution_low.max():.4f}] "
          f"(cached [{cached_low.min():.4f}, {cached_low.max():.4f}])")
    print("  photonic is a single noisy realisation, so it matches in distribution, "
          "not point by point")

    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(os.path.join(SAVE_DIR, 'qho_smoother_contribution_fp64.npy'), contribution_high)
    # The `_analog` file written here is the photonic_dot *emulation* of the analog
    # hardware, not a hardware measurement; the same name under
    # reproduce_plots/measurements/ is the real device. Same name so the plotting
    # code needs no special case.
    np.save(os.path.join(SAVE_DIR, 'qho_smoother_contribution_analog.npy'), contribution_low)
    print(f"\nSaved reproduced results to {SAVE_DIR}")
    print("  note: qho_smoother_contribution_analog.npy here is the photonic_dot "
          "emulation of the analog hardware, not a hardware measurement")
