"""Run every PC (parallel capacitors) reproduce_*.py script.

Each script recomputes one part of the PC results -- the residual-evolution
traces or the 100-trial iteration counters and representative field
solution -- and compares it against the cached measurements in
reproduce_plots/measurements/. Assumes mg_load_files/parallel_capacitors/
already holds the multigrid hierarchy (built once by
build_mg_hierarchies.py).
"""

import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

scripts = sorted(
    f for f in os.listdir(_ROOT)
    if f.startswith('reproduce_pc_') and f.endswith('.py')
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
    print(f'All {len(scripts)} PC scripts ran successfully.')
