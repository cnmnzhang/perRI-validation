"""Makes `perri_validation` importable when a script under perri_validation/ is run directly.

Entry-point scripts (perri_validation/scripts/run_*.py) call ensure_importable() before any
`from perri_validation... import ...` so they work both as `python -m perri_validation.scripts.run_dx_incident`
(no bootstrap needed) and as `python perri_validation/scripts/run_dx_incident.py` (direct execution,
where the directory containing `perri_validation/` isn't automatically on sys.path).
"""

import sys
from pathlib import Path


def ensure_importable() -> None:
    package_parent = Path(__file__).resolve().parents[2]
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))
