"""Reproduce both curves of lqcd_hybrid_comparison.svg.

"Hybrid (IR method)" and "Hybrid (our work)" are both measured on one gauge
configuration over the five masses the figure plots. They share the four heavier ones
and differ only in the lightest: IR uses -0.06, MPPMG the near-critical -0.06284 that
reproduce_lqcd_counters.py measures at. That is what plot_lqcd_hybrid_comparison.py assumes -- it plots
the two curves against slightly different x for that last point.

  "Hybrid (our work)"  MPPMG -- 'mixed' setup phase and an 8-bit quantized, noisy
                       V-cycle preconditioner, run to rtol 1e-12. The setup phase is
                       built once at the lightest mass and reused for the other four,
                       so the sweep costs one setup plus the solves.
  "Hybrid (IR method)" iterative refinement, described below.

The MPPMG sweep runs first, so that --include-critical-mass -- which makes the IR sweep
take days -- does not prevent it from being measured.



Iterative refinement (IR): an outer fp64 residual correction wrapped around a fixed
number of inner Richardson sweeps run on the emulated photonic hardware. Per outer
step,

    e = 0
    repeat num_inner times:  e <- e + omega*(r - A_low @ e)   photonic, 8-bit + noise
    x <- x + e
    r  = b - A_high @ x                                       digital, fp64

until ||r|| < 9e-12 * ||b||, with the Richardson relaxation parameter

    omega = 0.25 / (lambda_max + lambda_min)     of A_high.

The two counted quantities are the figure's two hardcoded curves: `digital` is the
number of outer steps (one fp64 matrix-vector product each) and `photonic` is the
number of inner steps, so photonic = digital * num_inner exactly.

`num_inner` is not derived here -- it was hand-tuned per mass by sweeping it and
keeping the value that minimised the total work. The surviving values are the paper's
table, transcribed into IR_SWEEP below along with the condition numbers that table reports.

Everything runs on a single gauge configuration and a single right-hand side.

Runtime: about 15 minutes for the MPPMG sweep, then about 30 minutes for the four IR
masses run by default, the latter dominated by m = 0.1 (51000 photonic steps at ~22 ms
each). The fifth IR mass, m = -0.06, needs 11.5 million photonic steps -- days of
compute -- so by default its published value is filled in instead; pass
--include-critical-mass to actually run it. MPPMG has no such problem and always runs
all five.
"""

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_ROOT, '..')))

import numpy as np
import scipy as sp

from phoamg.linear_solver.cg import get_atol_rtol
from phoamg.multigrid.two_grid import TwoGridQCD

MEASUREMENTS_DIR = os.path.normpath(os.path.join(_ROOT, '..', 'reproduce_plots', 'measurements'))
SAVE_DIR = os.path.join(_ROOT, 'measurements')
GAUGE_FILE = os.path.join(_ROOT, 'lqcd_inputs', 'gauge_links_10configs.npy')

NT = NX = 128
NC, NS = 1, 2

CONFIG_INDEX = 0
NUM_RHS = 10
RHS_INDEX = 0

# Every random draw in this pipeline -- photonic noise, setup-phase candidate vectors,
# pyamg's spectral-radius starting vector -- comes from numpy's global RNG, so seeding
# it before anything runs makes the script give identical results on every run.
SEED = 0

# Richardson relaxation parameter omega = OMEGA_FACTOR / (lambda_max + lambda_min).
OMEGA_FACTOR = 0.25
# Convergence threshold of the outer refinement. Deliberately 9e-12, not the 1e-12 the
# mgcg runs use: it is what produced the published counts.
RTOL = 9e-12
MAX_OUTER = 10**7

BIT_PRECISION = 8
BIT_PRECISION_ADC = 8
NOISE_STRENGTH = 2**3 / np.sqrt(2)

# MPPMG settings, matching reproduce_lqcd_counters.py's noisy run.
MPPMG_RTOL = 1e-12
MPPMG_MAX_OUTER = 60

# Lightest mass of the MPPMG sweep. Deliberately not IR's -0.06: this is the
# near-critical mass reproduce_lqcd_counters.py measures at, so both scripts report the
# same point, and it is where plot_lqcd_hybrid_comparison.py already plots this curve -- its second
# mass list ends at -0.063 while the IR one ends at -0.06. Duplicated from that
# script's LIGHTEST_MASS rather than imported, to keep the two independent.
MPPMG_LIGHTEST_MASS = -0.06283772233983162

# Digital / photonic matrix-vector products spent by the 'mixed' setup phase, per
# candidate vector, times the 8 candidates: 42 double-precision residual updates and
# 5 + 10 + 40*20 = 815 low-precision smoothing steps.
SETUP_HIGH_MIXED = 42 * 8
SETUP_LOW_MIXED = 815 * 8

# Bare mass, condition number, inner Richardson steps, heaviest first -- the five masses
# lqcd_hybrid_comparison.svg plots, transcribed from the paper table in
# lqcd_inputs/ir_richardson_parameters.txt.
IR_SWEEP = (
    (1.0, 22.61, 20),
    (0.5, 65.95, 20),
    (0.25, 190.76, 50),
    (0.1, 637.70, 100),
    (-0.06, 140311.92, 100),
)
CRITICAL_MASS_INDEX = 4

# (digital, photonic) MVMs at m = -0.06, measured once at a cost of about 72 hours of
# compute. Used in place of a fresh run unless --include-critical-mass.
PUBLISHED_CRITICAL_MASS = (115379, 11537900)


def total_mvms(counters):
    """Total MVMs (digital + photonic) per mass.

    Digital  = outer_iterations + 2*iterations + SETUP_HIGH_MIXED
    Photonic = 4*iterations                    + SETUP_LOW_MIXED

    `counters` is (mass, 2) -- [iterations, outer_iterations] -- in IR_SWEEP order.
    """
    iterations, outer = counters.T
    return outer + 6 * iterations + SETUP_HIGH_MIXED + SETUP_LOW_MIXED


def mppmg_sweep(gauge_links):
    """Measure the MPPMG solver at every IR_SWEEP mass; return (mass, 2) counters.

    Built on its own TwoGridQCD so nothing is shared with the IR measurement. The setup
    phase is run once, at the lightest mass -- the hardest one, so the interpolation
    basis it produces is the one the other masses can safely reuse.

    Seeded first, before anything that draws, which is what makes the sweep
    reproducible: the setup phase starts from random candidate vectors, the noisy
    matrix-vector product draws its noise, and pyamg's approximate_spectral_radius
    (called by update_m) draws a starting vector. All three use numpy's global RNG.
    The right-hand side is drawn exactly as run_mass draws IR's -- seed, draw NUM_RHS
    vectors, take RHS_INDEX -- so both curves solve the same system. Drawing (and
    discarding) the other nine keeps every later random draw identical to what it was
    when this sweep used all ten right-hand sides.
    """
    np.random.seed(SEED)
    b = (np.random.randn(NUM_RHS, NT * NX * NS)
         + 1j * np.random.randn(NUM_RHS, NT * NX * NS))[RHS_INDEX]

    mg = TwoGridQCD(gauge_links, NT, NX, NC, NS)
    masses = [m for m, _, _ in IR_SWEEP[:CRITICAL_MASS_INDEX]] + [MPPMG_LIGHTEST_MASS]
    lightest = masses[-1]

    print(f"  building the mixed (noisy) setup phase at m = {lightest:+.5f}", flush=True)
    started = time.time()
    mg.update_m(lightest)
    mg.change_relax_method(True, BIT_PRECISION, NOISE_STRENGTH, BIT_PRECISION_ADC)
    mg.run_setup_phase(setup_name='mixed', num5=1, num10=1, num20=40,
                       initial_precision=BIT_PRECISION, noise_strength=NOISE_STRENGTH,
                       bit_precision_adc=BIT_PRECISION_ADC)
    B_noisy = mg.B.copy()
    print(f"    done [{(time.time() - started) / 60:.1f} min]", flush=True)

    counters = []
    for m in masses:
        mg.update_m(m)
        mg.create_operators(B_noisy)
        mg.change_relax_method(True, BIT_PRECISION, NOISE_STRENGTH, BIT_PRECISION_ADC)
        _, counter, _ = mg.mgcg(b, rtol=MPPMG_RTOL, numPre=4, numPost=0,
                                single_precision=False, max_inner=8,
                                max_outer=MPPMG_MAX_OUTER)
        censored = counter[1] >= MPPMG_MAX_OUTER
        print(f"    m = {m:+9.5f}   iterations {counter[0]:7d}   "
              f"outer {counter[1]:5d}"
              f"{'   CENSORED' if censored else ''}", flush=True)
        counters.append(counter)

    return np.array(counters)


def extreme_eigenvalues(A):
    """Return (lambda_max, lambda_min) of A."""
    lambda_max = sp.sparse.linalg.eigs(A, k=1, which='LM')[0][0]
    lambda_min = sp.sparse.linalg.eigs(A, k=1, which='SM')[0][0]
    return lambda_max, lambda_min


def iterative_refinement(mg, b, num_inner, omega, rtol=RTOL, max_outer=MAX_OUTER,
                         report_every=100):
    """Solve mg.A_high x = b by IR and return (digital_mvms, photonic_mvms, converged).

    The inner sweeps use mg.polynomial, which after change_relax_method(low=True, ...)
    routes its matrix-vector product through the emulated photonic hardware; the outer
    residual uses the exact fp64 operator. Overwrites mg.coefficient with omega.
    """
    mg.coefficient = omega
    atol, _ = get_atol_rtol('cg', np.linalg.norm(b), 0., rtol)

    x = np.zeros_like(b)
    r = b.copy()
    digital = photonic = 0
    started = time.time()

    for _ in range(max_outer):
        e = np.zeros_like(b)
        for _ in range(num_inner):
            mg.polynomial(mg.A_low, e, r)
            photonic += 1
        x += e

        r = b - mg.A_high.dot(x)
        digital += 1
        residual = np.linalg.norm(r)
        if residual < atol:
            return digital, photonic, True
        if digital % report_every == 0:
            print(f"      {digital:7d} outer steps, residual {residual:.3e}, "
                  f"target {atol:.3e}, {(time.time() - started) / 60:.1f} min", flush=True)

    return digital, photonic, False


def run_mass(mg, m, num_inner):
    """Measure one point of the curve. Returns (condition_number, digital, photonic)."""
    mg.update_m(m)
    mg.change_relax_method(True, BIT_PRECISION, NOISE_STRENGTH, BIT_PRECISION_ADC)

    lambda_max, lambda_min = extreme_eigenvalues(mg.A_high)
    condition_number = np.abs(lambda_max / lambda_min)
    omega = OMEGA_FACTOR / (lambda_max + lambda_min)
    print(f"    lambda_max={lambda_max.real:.6f}  lambda_min={lambda_min.real:.6g}  "
          f"condition number={condition_number:.2f}", flush=True)

    # Seeded per mass, so each point is reproducible on its own and independent of
    # which other masses ran. The right-hand side is the first of ten complex Gaussian
    # vectors drawn from SEED.
    np.random.seed(SEED)
    b = (np.random.randn(NUM_RHS, NT * NX * NS)
         + 1j * np.random.randn(NUM_RHS, NT * NX * NS))[RHS_INDEX]

    digital, photonic, converged = iterative_refinement(mg, b, num_inner, omega)
    if not converged:
        raise RuntimeError(f"IR did not converge at m={m} within {MAX_OUTER} outer steps")
    return condition_number, digital, photonic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--include-critical-mass', action='store_true',
                        help="also run m = -0.06 (11.5 million photonic steps, days of "
                             "compute) instead of using its published value")
    args = parser.parse_args()

    gauge_links = np.load(GAUGE_FILE)[CONFIG_INDEX]
    print(f"Gauge configuration {CONFIG_INDEX} of {GAUGE_FILE}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- "Hybrid (our work)": MPPMG across the five plotted masses ---------------
    print(f"\n{'=' * 75}\nMPPMG sweep, rtol {MPPMG_RTOL:g}, single right-hand side "
          f"(matching IR's)", flush=True)
    mppmg_counters = mppmg_sweep(gauge_links)
    mppmg_totals = total_mvms(mppmg_counters)
    np.save(os.path.join(SAVE_DIR, 'lqcd_mppmg_total_mvms_5masses.npy'), mppmg_totals)

    cached_mppmg = np.load(os.path.join(MEASUREMENTS_DIR, 'lqcd_mppmg_total_mvms_5masses.npy'))
    print("\n  total MVMs per mass, heaviest first")
    print(f"    reproduced {np.round(mppmg_totals, 2)}")
    print(f"    cached     {np.round(cached_mppmg, 2)}")

    # --- "Hybrid (IR method)": iterative refinement ------------------------------
    mg = TwoGridQCD(gauge_links, NT, NX, NC, NS)

    condition_numbers = []
    counters = []
    started = time.time()

    for index, (m, table_condition_number, num_inner) in enumerate(IR_SWEEP):
        skipped = index == CRITICAL_MASS_INDEX and not args.include_critical_mass
        print(f"\n{'=' * 75}\n  m = {m:+.2f}   inner Richardson steps = {num_inner}"
              f"{'   [published value, not run]' if skipped else ''}", flush=True)

        if skipped:
            condition_number, (digital, photonic) = table_condition_number, PUBLISHED_CRITICAL_MASS
        else:
            condition_number, digital, photonic = run_mass(mg, m, num_inner)
            print(f"    digital={digital}  photonic={photonic}  "
                  f"[{(time.time() - started) / 60:.1f} min elapsed]", flush=True)

        condition_numbers.append(condition_number)
        counters.append((digital, photonic))

    counters = np.array(counters)
    condition_numbers = np.array(condition_numbers)

    np.save(os.path.join(SAVE_DIR, 'lqcd_ir_counters_5masses.npy'), counters)

    cached = np.load(os.path.join(MEASUREMENTS_DIR, 'lqcd_ir_counters_5masses.npy'))

    print(f"\n{'=' * 75}")
    print("Reproduced vs published. Each mass reseeds the global RNG before solving, so\n"
          "these counters are deterministic -- rerunning gives the same integers, and the\n"
          "MPPMG sweep above does not perturb them.\n")
    print(f"  {'mass':>7}  {'condition number':>28}  {'inner':>5}  "
          f"{'digital MVMs':>22}  {'photonic MVMs':>24}")
    print(f"  {'':>7}  {'reproduced':>13} {'table':>14}  {'':>5}  "
          f"{'reproduced':>10} {'published':>11}  {'reproduced':>11} {'published':>12}")
    for (m, table_condition_number, num_inner), condition_number, (digital, photonic), \
            (cached_digital, cached_photonic) in zip(IR_SWEEP, condition_numbers,
                                                     counters, cached):
        print(f"  {m:+7.2f}  {condition_number:13.2f} {table_condition_number:14.2f}  "
              f"{num_inner:5d}  {digital:10d} {cached_digital:11d}  "
              f"{photonic:11d} {cached_photonic:12d}")

    print(f"\nSaved reproduced results to {SAVE_DIR}")
