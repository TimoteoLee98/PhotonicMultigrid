"""Run every QAO (quartic anharmonic oscillator) reproduce_*.py script.

Each script recomputes one part of the QAO results -- eigenpairs, the
measured-vs-expected error comparison, or the LOBPCG solver iteration
counters -- and compares it against the cached measurements in
reproduce_plots/measurements/. Assumes mg_load_files/qao/ already holds the
multigrid hierarchy (built once by build_mg_hierarchies.py).
"""

import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

scripts = sorted(
    f for f in os.listdir(_ROOT)
    if f.startswith('reproduce_qao_') and f.endswith('.py')
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
    print(f'All {len(scripts)} QAO scripts ran successfully.')
