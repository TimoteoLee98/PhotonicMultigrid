"""Reproduce the lattice-QCD solver counters behind lqcd_operations.svg.

Three runs, all on the same 10 gauge configurations and the same 10 right-hand sides:

  1. fp64 CG          unpreconditioned conjugate gradient in double precision
  2. quantized 8-bit  'standard' setup phase + 8-bit quantized V-cycle smoother
  3. noisy 8-bit      'mixed' setup phase + 8-bit quantized *and noisy* V-cycle smoother

All three are measured at the lightest (near-critical) mass, the only one this figure
plots. The mass sweep that produces the "Hybrid (our work)" curve of
`lqcd_hybrid_comparison.svg` lives in `reproduce_lqcd_hybrid_comparison.py` instead, on a single gauge
configuration and on the same masses as the iterative-refinement curve it is plotted
against.

Runtime: roughly 5-10 minutes per gauge configuration, so under an hour in total.
Results are rewritten after every configuration, so an interrupted run keeps whatever it
finished.
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_ROOT, '..')))

import numpy as np

from phoamg.linear_solver.cg import cg
from phoamg.multigrid.two_grid import TwoGridQCD

MEASUREMENTS_DIR = os.path.normpath(os.path.join(_ROOT, '..', 'reproduce_plots', 'measurements'))
SAVE_DIR = os.path.join(_ROOT, 'measurements')
GAUGE_FILE = os.path.join(_ROOT, 'lqcd_inputs', 'gauge_links_10configs.npy')

NT = NX = 128
NC, NS = 1, 2

# Lightest (near-critical) bare mass -- the only one lqcd_operations.svg plots.
LIGHTEST_MASS = -0.06283772233983162

# Kept as a one-element list rather than a scalar so the saved noisy array keeps its
# (configuration, mass, right-hand side, counter) shape, which plot_lqcd_counters.py
# indexes as [:, 0].
NOISY_MASSES = [LIGHTEST_MASS]

NUM_RHS = 10

# Every random draw in this pipeline -- photonic noise, setup-phase candidate vectors,
# pyamg's spectral-radius starting vector -- comes from numpy's global RNG. SEED fixes
# the right-hand sides; each gauge configuration is then reseeded with SEED + 1 + its
# index, so a configuration reproduces on its own and is unaffected by what ran before
# it. Without that, changing one configuration's tolerance would alter the number of
# noise draws it makes and shift every later configuration's results.
SEED = 0

RTOL = 1e-12

BIT_PRECISION = 8
BIT_PRECISION_ADC = 8
NOISE_STRENGTH = 2**3 / np.sqrt(2)

MAX_OUTER_QUANTIZED = 20
MAX_OUTER_NOISY = 25

def mgcg_counter(mg, b, max_outer, rtol):
    """Run one preconditioned solve; return its [iterations, outer_iterations] and
    final relative residual norm ||b - A x|| / ||b||."""
    _, counter, residual = mg.mgcg(b, rtol=rtol, numPre=4, numPost=0, single_precision=False,
                                   max_inner=8, max_outer=max_outer)
    return counter, residual / np.linalg.norm(b)


def save(counters_cg, counters_quantized, counters_noisy):
    os.makedirs(SAVE_DIR, exist_ok=True)
    for name, array in [
        ('lqcd_cg_counters_fp64.npy', np.array(counters_cg)),
        ('lqcd_mgcg_counters_quantized8bit.npy', np.array(counters_quantized)),
        ('lqcd_mgcg_counters_noisy8bit.npy', np.array(counters_noisy)),
    ]:
        np.save(os.path.join(SAVE_DIR, name), array)


def compare(tag, reproduced, ground_truth):
    n = len(reproduced)
    sem = reproduced.std(axis=0) / np.sqrt(n)
    print(f"  {tag:24s} reproduced={np.round(reproduced.mean(axis=0), 2)} "
          f"+/- {np.round(sem, 2)}   cached={np.round(ground_truth.mean(axis=0), 2)}")


if __name__ == '__main__':
    gauge_links_array = np.load(GAUGE_FILE)
    print(f"Loaded {len(gauge_links_array)} gauge configurations from {GAUGE_FILE}")
    print(f"Mass: {LIGHTEST_MASS:+.5f} (lightest, near-critical)")

    np.random.seed(SEED)
    bs = (np.random.randn(NUM_RHS, NT * NX * NS)
          + 1j * np.random.randn(NUM_RHS, NT * NX * NS))

    counters_cg = []
    counters_quantized = []
    counters_noisy = []

    for index_config, gauge_links in enumerate(gauge_links_array):
        started = time.time()
        np.random.seed(SEED + 1 + index_config)
        rtol = RTOL
        print(f"\n{'=' * 75}\nGauge configuration {index_config + 1}/{len(gauge_links_array)}"
              f"   seed={SEED + 1 + index_config}", flush=True)

        mg = TwoGridQCD(gauge_links, NT, NX, NC, NS)
        mg.update_m(LIGHTEST_MASS)

        # --- Run 1: unpreconditioned fp64 CG, lightest mass ---------------------
        cg_solutions = [cg(mg.A_high, b, rtol=rtol, maxiter=10000) for b in bs]
        cg_config = [it for _, it in cg_solutions]
        cg_residuals = [np.linalg.norm(b - mg.A_high.dot(x)) / np.linalg.norm(b)
                        for (x, _), b in zip(cg_solutions, bs)]
        counters_cg.append(cg_config)
        print(f"  fp64 CG            iterations={np.mean(cg_config):.2f}  "
              f"final relative residual mean={np.mean(cg_residuals):.3e} "
              f"max={np.max(cg_residuals):.3e}", flush=True)

        # --- Run 2: 8-bit quantized smoother, standard setup, lightest mass ------
        mg.change_relax_method(True, BIT_PRECISION, None, BIT_PRECISION_ADC)
        mg.run_setup_phase(setup_name='standard', initial_relax=True, numRelax=800,
                           initial_precision=BIT_PRECISION, noise_strength=None,
                           bit_precision_adc=BIT_PRECISION_ADC)
        mg.create_operators(mg.B)

        quantized_config = []
        quantized_residuals = []
        for b in bs:
            mg.change_relax_method(True, BIT_PRECISION, None, BIT_PRECISION_ADC)
            counter, residual = mgcg_counter(mg, b, max_outer=MAX_OUTER_QUANTIZED, rtol=rtol)
            quantized_config.append(counter)
            quantized_residuals.append(residual)
        counters_quantized.append(quantized_config)
        print(f"  quantized 8-bit    {np.mean(quantized_config, axis=0)}  "
              f"final relative residual mean={np.mean(quantized_residuals):.3e} "
              f"max={np.max(quantized_residuals):.3e}", flush=True)

        # --- Run 3: 8-bit noisy smoother, mixed setup, lightest mass -------------
        mg.change_relax_method(True, BIT_PRECISION, NOISE_STRENGTH, BIT_PRECISION_ADC)
        mg.run_setup_phase(setup_name='mixed', num5=1, num10=1, num20=40,
                           initial_precision=BIT_PRECISION, noise_strength=NOISE_STRENGTH,
                           bit_precision_adc=BIT_PRECISION_ADC)
        B_noisy = mg.B.copy()

        noisy_config = []
        for m in NOISY_MASSES:
            mg.update_m(m)
            mg.create_operators(B_noisy)
            noisy_mass = []
            noisy_mass_residuals = []
            for b in bs:
                mg.change_relax_method(True, BIT_PRECISION, NOISE_STRENGTH, BIT_PRECISION_ADC)
                counter, residual = mgcg_counter(mg, b, max_outer=MAX_OUTER_NOISY, rtol=rtol)
                noisy_mass.append(counter)
                noisy_mass_residuals.append(residual)
            noisy_config.append(noisy_mass)
            print(f"  noisy 8-bit  m={m:+.5f}  {np.mean(noisy_mass, axis=0)}  "
                  f"final relative residual mean={np.mean(noisy_mass_residuals):.3e} "
                  f"max={np.max(noisy_mass_residuals):.3e}", flush=True)
        counters_noisy.append(noisy_config)

        save(counters_cg, counters_quantized, counters_noisy)
        print(f"  [{(time.time() - started) / 60:.1f} min] saved {len(counters_cg)} "
              f"configuration(s) to {SAVE_DIR}", flush=True)

    # --- Comparison against the cached measurements -----------------------------
    counters_cg = np.array(counters_cg)
    counters_quantized = np.array(counters_quantized)
    counters_noisy = np.array(counters_noisy)

    print(f"\n{'=' * 75}")
    print("Reproduced vs cached, averaged over gauge configurations and right-hand sides")
    print("(fp64 CG is deterministic and must match; the others carry setup and "
          "quantization noise)\n")

    cached_cg = np.load(os.path.join(MEASUREMENTS_DIR, 'lqcd_cg_counters_fp64.npy'))
    cached_quantized = np.load(os.path.join(MEASUREMENTS_DIR, 'lqcd_mgcg_counters_quantized8bit.npy'))
    cached_noisy = np.load(os.path.join(MEASUREMENTS_DIR, 'lqcd_mgcg_counters_noisy8bit.npy'))

    compare('fp64 CG', counters_cg.mean(axis=1), cached_cg.mean(axis=1))
    compare('quantized 8-bit', counters_quantized.mean(axis=1), cached_quantized.mean(axis=1))
    compare('noisy 8-bit', counters_noisy[:, 0].mean(axis=1), cached_noisy[:, 0].mean(axis=1))

    print(f"\nSaved reproduced results to {SAVE_DIR}")
