# phoamg — Photonic-Hardware-Accelerated Algebraic Multigrid

`phoamg` implements Algebraic MultiGrid (AMG) preconditioning for three problems —
the Quartic Anharmonic Oscillator (QAO) eigenvalue problem, a 2-D Poisson "parallel
capacitors" (PC) problem, and lattice QCD (LQCD) Dirac-operator solves — with support
for emulating a low-precision **photonic hardware** matrix-vector-multiply unit inside
the AMG smoother/preconditioner. Each level's matrix-vector product is a swappable
callable, so a real photonic device can be driven in place of the emulation — which is
how the hardware measurements shipped here were taken.

## Paper

T. Lee, F. Brückerhoff-Plückelmann, J. Dijkstra, J. M. Pawlowski, and W. Pernice,
"Integrated photonic multigrid solver for partial differential equations".

This repository reproduces every figure and result reported in the paper submitted to 
Nature Computational Science: `reproduce_plots/` renders the 10 published figures from
cached measurement data, and `reproduce_results/` regenerates that data from first 
principles (problem definitions, multigrid hierarchies, and a software emulation of the
photonic hardware), verifying it against the cached values.

## Repository structure

```
phoamg/                core library (multigrid, eigensolvers, low-precision emulation)
reproduce_plots/       figure-generation package: 10 plot_*.py scripts + measurements/
                       (small cached input data). plots/ ships empty — the rendered
                       figures appear there when you run the scripts.
reproduce_results/     regenerates the underlying result data from more fundamental
                       inputs (multigrid hierarchies, raw Hamiltonians, photonic-hardware
                       emulation). Its measurements/ and plots/ likewise ship empty and
                       are filled by running the scripts.
reproduce_all.py       runs the full reproduction pipeline (all five steps below) in
                       one command
requirements.txt       pinned Python dependencies
LICENSE                MIT license
```

## Dependencies

| Package | Version tested |
|---|---|
| Python | 3.13.0 |
| numpy | 2.1.3 |
| scipy | 1.14.1 |
| pyamg | 5.2.1 |
| matplotlib | 3.9.2 |

Pinned in `requirements.txt`. No non-standard hardware is required to *run* this
repository's code. Note: some of the "measured" data used as ground truth (e.g.
`*_analog.npy` under `reproduce_plots/measurements/`) was originally collected from a physical photonic hardware
prototype — that hardware is **not** required to run or validate this software;
`reproduce_results/photonic_dot.py` provides a software emulation of it that is used
throughout this repo instead.

### Operating systems tested

Windows 11 (build 10.0.26100). Not yet tested on Linux or macOS (the code has no
OS-specific dependencies, so it is expected to work, but this has not been verified).

## Installation

```
git clone https://github.com/TimoteoLee98/PhotonicMultigrid
cd PhotonicMultigrid
pip install -r requirements.txt
```

There is no separate package build step — `phoamg` is imported directly via `sys.path`
manipulation at the top of each `reproduce_*.py` script, not installed as a pip package.
The `plot_*.py` scripts do not use `phoamg` at all; their `sys.path` line is there to
import `plot_utils`.

**Typical install time on a normal desktop:** under 5 minutes (dominated by downloading
prebuilt wheels for scipy/numpy/pyamg; no compilation required with the pinned
versions on a standard 64-bit Windows/Linux/macOS Python install).

## Demo: reproduce the paper figures from cached data

`reproduce_plots/measurements/` is the small (~240 KB total) cached dataset — 21
`.npy` files — that this demo runs on.

```
python reproduce_plots/run_all_plots.py
```

**Expected output:** 10 figures written to `reproduce_plots/plots/`: `pc_field.svg`,
`pc_operations.svg`, `pc_residual_norms.svg`, `qao_eigenfunction.svg`, `qao_error.svg`,
`qao_operations_final.svg`, `qao_residuals.svg`, `qho_smoother.svg`,
`lqcd_hybrid_comparison.svg`, `lqcd_operations.svg`.

**Expected run time on a normal desktop:** well under a minute — this step only loads
cached arrays and plots them, it does not run any solver.

A single figure can also be regenerated individually, e.g. `python
reproduce_plots/plot_pc_field.py`.

## Reproduction instructions: regenerating the results from scratch

This is the optional-but-included deeper level: instead of plotting cached numbers,
these scripts recompute them from the underlying problem definitions (Hamiltonians,
multigrid hierarchies) and the `photonic_dot` hardware emulation, then verify the
recomputed values agree with the cached/measured ones within statistical error.

```
python reproduce_plots/run_all_plots.py                        # sanity check against cached data, well under a minute
python reproduce_results/run_all_qao_pc_qho.py                  # QAO + PC + QHO reproduction, a few minutes
python reproduce_results/reproduce_lqcd_hybrid_comparison.py    # ~45 minutes
python reproduce_results/reproduce_lqcd_counters.py             # under an hour
python reproduce_results/run_all_plots.py                       # renders figures from the freshly reproduced data, well under a minute
```

Alternatively, `python reproduce_all.py` runs all five steps above sequentially in one
command (same ~1.5-2 hour total runtime; stops at the first failing step since each one
builds on the last). Pass `--include-critical-mass` to forward that flag to
`reproduce_lqcd_hybrid_comparison.py`.

`run_all_qao_pc_qho.py` chains the individual `reproduce_qao_*.py`, `reproduce_pc_*.py`
and `reproduce_qho_smoother.py` scripts; each can also be run on its own (as can
`run_all_qao.py` / `run_all_pc.py` for just one problem). All of them load the QAO, QHO
and PC multigrid hierarchies from `reproduce_results/mg_load_files/`, which ships with the
repository.

`build_mg_hierarchies.py` is not part of the pipeline — nothing runs it, and you do not
need to. It is there to document where `mg_load_files/` came from: it rebuilds those
hierarchies from the problem definitions and writes them to `mg_load_files_GENERATED/`,
a separate directory so that a rebuild can never overwrite the shipped ones, for anyone
who wants to compare the two. A rebuilt finest-level operator is identical, being fully
determined by the problem definition; coarser levels match in shape and sparsity pattern.

Each `reproduce_*.py` script prints a comparison (correlation, relative error, or
mean ± standard error) against the corresponding cached measurement, and saves its
reproduced result into `reproduce_results/measurements/` under the same filename used
in `reproduce_plots/measurements/`.

**Note what that convention implies.** Ten of the 21 cached files are real
photonic-hardware measurement, namely every QAO/PC/QHO quantity on the analog
path:

```
pc_analog_richardson_residual_251iters.npy      qao_error_analog.npy
pc_field_solution_64x64.npy                     qao_eigenvalues_n0to2.npy
pc_hybrid_mpmgp_residual_checkpoints_8x5.npy    qao_eigenvectors_n0to2_grid120.npy
pc_solver_counters_100trials.npy                qao_residual_norm_history_noisy_3modes.npy
qao_solver_counters_box_100trials.npy           qho_smoother_contribution_analog.npy
```

Under `reproduce_results/measurements/` the files carrying those same ten names hold
`photonic_dot` *emulations* of that hardware — same names, so the plotting code needs no
special case. **The filename alone does not tell you which** — for example
`pc_hybrid_mpmgp_residual_checkpoints_8x5.npy` carries no `analog` or `box` marker yet is
a hardware measurement — so rely on the list above rather than on the naming. Nothing in
`reproduce_results/` is a hardware measurement, and every script that writes one of these
ten says so on the line it saves.

The other 11 cached files are software in both places: the five `lqcd_*` files, the five
digital and emulated variants marked `fp64`, `double`, `ideal` or `noisy8bit`, and
`pc_solver_counters_digital_3settings.npy` (the fp64 / ideal-8-bit / noisy-8-bit PC
counters that `pc_operations.svg` plots beside the photonic bar).

This distinction is about provenance, not about conclusions: all of these solves converge
to their target tolerance either way, so an emulated run and the hardware run differ in
noise realisation rather than in outcome, and no figure changes its message.
`reproduce_results/run_all_plots.py` then reuses
the exact same plotting code as `reproduce_plots/` (via a `--base-dir` flag) to render
figures from this freshly reproduced data into `reproduce_results/plots/`.

**Expected run time on a normal desktop:** `run_all_qao_pc_qho.py` takes a few minutes in
total — the slowest single script is `reproduce_qao_counters.py` (400 LOBPCG solves: four
precision settings over the 100 starting blocks in `qao_inputs/X_100.npy`); the rest
complete in well under a minute each.

`reproduce_lqcd_counters.py` solves 128×128 lattice Dirac systems to `rtol=1e-12`, at a
single (near-critical) mass, and takes roughly **5-10 minutes per gauge configuration,
under an hour in total**. It rewrites its output after every configuration, so an
interrupted run keeps whatever it finished.

`reproduce_lqcd_hybrid_comparison.py` takes about **45 minutes** — roughly 15 for the MPPMG
sweep and 30 for the IR sweep, most of the latter spent on the lightest mass IR
actually runs (`m = 0.1`, 51 000 emulated-photonic matrix-vector products). It does *not*
run the near-critical mass `m = -0.06` by default — that single point needs 11.5 million
photonic products, measured at roughly **72 hours** on this machine — and substitutes the
published value for it instead. Pass `--include-critical-mass` to run it for real.

## Notes on reproduction accuracy

The original results were generated without seeding numpy's RNG, so the stochastic
photonic-noise draws and pyamg's setup randomness were not exactly reproducible run to
run. Every script under `reproduce_results/` now seeds all of that from a single `SEED = 0`
declared near the top of the file, so a fresh run reproduces the same numbers every time.
Scripts that need more than one seeding point derive it from that constant rather than
repeating a literal: `reproduce_pc_residuals.py` reseeds before each of its two solves,
and `reproduce_lqcd_counters.py` uses `SEED + 1 + index` per gauge configuration so each
one reproduces on its own, unaffected by what ran before it.

With seeding in place, every reproduced quantity matches its cached counterpart within
statistical error, *except* the comparisons against real photonic-hardware measurements
(the ones loaded from `reproduce_plots/measurements/`, e.g. the PC box counters and the
QAO/QHO `_analog` comparisons): `photonic_dot` is a software emulation of the
noise model, not a calibrated replica of the physical device, so those specific
comparisons show a persistent, documented gap rather than a match. Each script prints
both sides of its comparison as it runs; `reproduce_pc_counters.py` also states its gap
up front in its docstring (36 iterations against the hardware's 49.8). This is expected,
not a bug, and is reported rather than tuned away, along with one further known gap:

- **LQCD near-critical mass (`m = -0.06`).** The `lqcd_hybrid_comparison.svg` "Hybrid
  (IR method)" curve needs 11.5 million emulated photonic matrix-vector products at this
  mass (~72 hours). `reproduce_lqcd_hybrid_comparison.py` substitutes the published value for this
  point by default; pass `--include-critical-mass` to compute it directly. The other
  curve, "Hybrid (our work)" (MPPMG, near-critical mass `-0.06284`), has no such
  problem and always runs all five masses in ~15 minutes.

## License

MIT — see `LICENSE`. All dependencies (numpy, scipy, pyamg, matplotlib) are
permissively licensed (BSD/MIT/PSF); none impose restrictions on this project's
license choice.
