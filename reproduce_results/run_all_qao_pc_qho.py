"""Run every QAO, PC, and QHO reproduce_*.py script -- the fast part of reproduce_results/.

Together these take a few minutes. QCD is not included -- it takes hours;
see reproduce_lqcd_*.py to run that separately.
"""

import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

scripts = sorted(
    f for f in os.listdir(_ROOT)
    if (f.startswith('reproduce_qao_') or f.startswith('reproduce_pc_')
        or f.startswith('reproduce_qho_'))
    and f.endswith('.py')
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
    print(f'All {len(scripts)} QAO/PC/QHO scripts ran successfully.')
