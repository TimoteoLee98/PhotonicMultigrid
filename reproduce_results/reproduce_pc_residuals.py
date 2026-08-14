"""Reproduce the parallel-capacitors residual traces behind pc_residual_norms.svg.

Two solves of the same problem, both with the photonic matrix-vector product:

  "Hybrid"  multigrid-preconditioned iterative refinement, residual recorded at each
            outer step
  "Analog"  plain Richardson iteration with no coarse-grid correction, residual
            recorded at every step

phoamg's solve_mixed cannot report a residual history, so this file has a local
`solve_mixed_with_history` that adds one using cg's existing callback hook, rather
than changing the shared library.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_ROOT, '..')))
sys.path.insert(0, _ROOT)

import numpy as np

from phoamg.linear_solver.cg import cg, get_atol_rtol
from mg_utils import load_mg
from photonic_dot import photonic_dot
from pc_problem import build_capacitor_rhs

MEASUREMENTS_DIR = os.path.normpath(os.path.join(_ROOT, '..', 'reproduce_plots', 'measurements'))
SAVE_DIR = os.path.join(_ROOT, 'measurements')

# Both solves below reseed from this before drawing their initial guess, so each one
# reproduces on its own regardless of what ran before it.
SEED = 0


def solve_mixed_with_history(A, b, M, rtol=1e-5, atol=0., maxiter=32, rtol_internal=1e-3):
    bnrm2 = np.linalg.norm(b)
    atol, _ = get_atol_rtol('cg', bnrm2, atol, rtol)

    r_array = []

    def run_inner_cg(rhs, x0=None):
        history = [np.linalg.norm(rhs)]

        def callback(x):
            history.append(np.linalg.norm(rhs - A.dot(x)))

        result, info = cg(A, rhs, x0=x0, rtol=rtol_internal, maxiter=maxiter, M=M, callback=callback)
        r_array.append(history)
        return result, info

    counter = 0
    x, info = run_inner_cg(b)
    counter += info
    r = b - A.dot(x)
    for i in range(40):
        y, info = run_inner_cg(r)
        counter += info
        x += y
        r = b - A.dot(x)
        rS = np.linalg.norm(r)
        if rS < atol:
            break

    return x, [counter, i + 1], rS, r_array


if __name__ == '__main__':
    mg = load_mg(os.path.join(_ROOT, 'mg_load_files', 'parallel_capacitors') + os.sep)
    b = build_capacitor_rhs(Nx=64, Ny=64)

    def dot_photonic(lvl, x):
        return photonic_dot(mg.As[lvl], x)

    print("=== Checkpoint residual array (Hybrid: multigrid + photonic dot) ===")
    mg.change_relax_method(False)
    mg.dots[0] = dot_photonic
    np.random.seed(SEED)
    x0 = np.random.randn(mg.As[0].shape[0])
    r0 = b - mg.As[0].dot(x0)
    M = mg.aspreconditioner(numPre=4, numPost=0, low_precision_residual=True,
                             cycle='V', coarse_solver="cg")
    u, counters, rS, r_array = solve_mixed_with_history(
        mg.As[0], r0, M, rtol=1e-12, rtol_internal=1e-3, maxiter=4)
    # Relative to r0, which is the right-hand side actually solved against here.
    print(f"  counters=[iterations, restarts]={counters}  "
          f"final relative residual={rS / np.linalg.norm(r0):.3e}")
    print(f"  r_array: {len(r_array)} restarts, lengths={[len(h) for h in r_array]}")

    ground_truth = np.load(os.path.join(MEASUREMENTS_DIR, 'pc_hybrid_mpmgp_residual_checkpoints_8x5.npy'))
    print(f"  ground truth shape={ground_truth.shape}, restarts={ground_truth.shape[0]}")
    print(f"  ground truth: mean initial residual={ground_truth[:, 0].mean():.3f}, "
          f"mean final residual={ground_truth[:, -1].mean():.3e}")
    print(f"  reproduced:   mean initial residual={np.mean([h[0] for h in r_array]):.3f}, "
          f"mean final residual={np.mean([h[-1] for h in r_array]):.3e}")

    print("\n=== Pure-Richardson trace (Analog: photonic dot only, no multigrid coarse solve) ===")
    mg.change_relax_method(False)
    mg.dots[0] = dot_photonic
    A0 = mg.As[0]
    np.random.seed(SEED)
    x0 = np.random.randn(A0.shape[0])
    b_new = b - A0.dot(x0)
    e = np.zeros_like(b)
    residual_norms = [np.linalg.norm(b_new - A0.dot(e))]
    for _ in range(250):
        mg.polynomial(0, e, b_new, 1)
        residual_norms.append(np.linalg.norm(b_new - A0.dot(e)))
    residual_norms = np.array(residual_norms)

    analog_ground_truth = np.load(os.path.join(MEASUREMENTS_DIR, 'pc_analog_richardson_residual_251iters.npy'))
    print(f"  reproduced length={len(residual_norms)}  ground truth length={len(analog_ground_truth)}")
    print(f"  reproduced:   start={residual_norms[0]:.3f}  end={residual_norms[-1]:.3e}")
    print(f"  ground truth: start={analog_ground_truth[0]:.3f}  end={analog_ground_truth[-1]:.3e}")
    n_below_1e10_repro = np.sum(residual_norms < 1e-10)
    n_below_1e10_truth = np.sum(analog_ground_truth < 1e-10)
    print(f"  iterations with residual < 1e-10: reproduced={n_below_1e10_repro}  ground truth={n_below_1e10_truth}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    try:
        r_array_to_save = np.array(r_array)
    except ValueError:
        r_array_to_save = np.array(r_array, dtype=object)
        print("  (note: r_array restarts had uneven lengths, saved as an object array)")
    # Both files are the photonic_dot *emulation* of the analog hardware, not hardware
    # measurements; the same names under reproduce_plots/measurements/ are the real
    # device. Same names so the plotting code needs no special case.
    np.save(os.path.join(SAVE_DIR, 'pc_hybrid_mpmgp_residual_checkpoints_8x5.npy'), r_array_to_save)
    np.save(os.path.join(SAVE_DIR, 'pc_analog_richardson_residual_251iters.npy'), residual_norms)
    print(f"\nSaved reproduced results to {SAVE_DIR}")
    print("  note: these are the photonic_dot emulation of the analog hardware, not "
          "hardware measurements")
