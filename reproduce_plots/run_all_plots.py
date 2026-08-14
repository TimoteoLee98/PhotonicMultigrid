"""Render all 10 figures by running every plot_*.py in this directory.

Takes no arguments. Reads the cached data in measurements/ and writes to plots/.
The equivalent runner in reproduce_results/ renders the same figures from freshly
reproduced data instead.
"""

import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

scripts = sorted(
    f for f in os.listdir(_ROOT)
    if f.startswith('plot_') and f.endswith('.py') and f != 'plot_utils.py'
)

failed = []
for script in scripts:
    print(f'--- running {script} ---')
    result = subprocess.run([sys.executable, os.path.join(_ROOT, script)])
    if result.returncode != 0:
        failed.append(script)

print()
if failed:
    print(f'{len(scripts) - len(failed)}/{len(scripts)} scripts succeeded. Failed: {failed}')
    sys.exit(1)
else:
    print(f'All {len(scripts)} plot scripts ran successfully. Output written to {os.path.join(_ROOT, "plots")}')
