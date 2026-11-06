"""PyCharm: repair UTF-8 decoding of the figure audit only; no model fitting."""
from pathlib import Path
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
source = Path(__file__).with_name('run_ble_cabin_boost_bayes.py')
out = root/'SCI-UWB/data/results/ble_cabin_boost_bayes_20261105'
env = dict(os.environ, PYTHONIOENCODING='utf-8')
result = subprocess.run([sys.executable, 'E:/codex_skill/nature-figure/scripts/validate_figure.py',
                         str(source), '--json'], capture_output=True, encoding='utf-8', env=env, check=True)
audit = json.loads(result.stdout)
assert audit['summary']['counts']['FAIL'] == 0
(out/'figure_preflight.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print('AUDIT_ONLY_COMPLETE', audit['summary'], flush=True)
