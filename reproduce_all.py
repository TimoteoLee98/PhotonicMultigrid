"""Run the full pipeline described in README.md's "Reproduction instructions", in order:

    python reproduce_plots/run_all_plots.py
    python reproduce_results/run_all_qao_pc_qho.py
    python reproduce_results/reproduce_lqcd_hybrid_comparison.py
    python reproduce_results/reproduce_lqcd_counters.py
    python reproduce_results/run_all_plots.py

Unlike the run_all_plots.py runners (which run independent, unrelated plot scripts and
so continue past a single failure), these steps build on each other -- each reproduce_*
script's output feeds the next -- so this stops at the first failing step instead of
continuing.

Takes roughly 1.5-2 hours in total, dominated by reproduce_lqcd_hybrid_comparison.py
(~45 minutes) and reproduce_lqcd_counters.py (under an hour); the rest complete in a
few minutes combined.
"""

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    os.path.join(_ROOT, 'reproduce_plots', 'run_all_plots.py'),
    os.path.join(_ROOT, 'reproduce_results', 'run_all_qao_pc_qho.py'),
    os.path.join(_ROOT, 'reproduce_results', 'reproduce_lqcd_hybrid_comparison.py'),
    os.path.join(_ROOT, 'reproduce_results', 'reproduce_lqcd_counters.py'),
    os.path.join(_ROOT, 'reproduce_results', 'run_all_plots.py'),
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--include-critical-mass', action='store_true',
                        help="forwarded to reproduce_lqcd_hybrid_comparison.py: also run "
                             "the near-critical mass point (~72 hours) instead of using "
                             "its published value")
    args = parser.parse_args()

    for i, script in enumerate(STEPS, start=1):
        cmd = [sys.executable, script]
        if os.path.basename(script) == 'reproduce_lqcd_hybrid_comparison.py' and args.include_critical_mass:
            cmd.append('--include-critical-mass')

        rel = os.path.relpath(script, _ROOT)
        print(f"\n{'=' * 75}\nStep {i}/{len(STEPS)}: {rel}\n{'=' * 75}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nStep {i}/{len(STEPS)} failed ({rel}); stopping.")
            sys.exit(1)

    print(f"\nAll {len(STEPS)} steps completed successfully.")
