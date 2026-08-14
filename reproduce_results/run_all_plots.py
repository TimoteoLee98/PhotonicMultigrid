"""Render all 10 figures from the freshly reproduced data in this directory.

Reuses reproduce_plots/plot_*.py unchanged, pointing each at this directory with
--base-dir, so there is exactly one copy of the plotting code.
"""

import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_SOURCE = os.path.normpath(os.path.join(_ROOT, '..', 'reproduce_plots'))

scripts = sorted(
    f for f in os.listdir(PLOTS_SOURCE)
    if f.startswith('plot_') and f.endswith('.py') and f != 'plot_utils.py'
)

failed = []
for script in scripts:
    print(f'--- running {script} (reading/writing under reproduce_results/) ---')
    result = subprocess.run(
        [sys.executable, os.path.join(PLOTS_SOURCE, script), '--base-dir', _ROOT])
    if result.returncode != 0:
        failed.append(script)

print()
if failed:
    print(f'{len(scripts) - len(failed)}/{len(scripts)} scripts succeeded. Failed: {failed}')
    sys.exit(1)
else:
    print(f'All {len(scripts)} plot scripts ran successfully. Output written to {os.path.join(_ROOT, "plots")}')
