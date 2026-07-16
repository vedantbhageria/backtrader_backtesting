"""Project layout glue.

backend/ holds the server + pipeline, strategies/ holds the trading
strategies, and data/artifact folders (datas/, reports/, report_out/,
vendor/, dashboard.html, saved_configs.json) live at the project ROOT.

Importing this module makes all three directories importable regardless of
where the process was started, so module names stay flat (`import db`,
`import EMAlgoNonLinTest`) and the server's importlib.reload keeps working.
"""
import os
import sys

BACKEND = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BACKEND)
STRATEGIES = os.path.join(ROOT, 'strategies')

for _p in (ROOT, BACKEND, STRATEGIES):
    if _p not in sys.path:
        sys.path.insert(0, _p)
