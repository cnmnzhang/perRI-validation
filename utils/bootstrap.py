"""Makes `perri_validation` importable when a script under  is run directly.

Entry-point scripts (scripts/run_*.py) call ensure_importable() before any
`from .. import ...` so they work both as `python -m scripts.build_dx_incident`
(no bootstrap needed) and as `python scripts/build_dx_incident.py` (direct execution,
where the directory containing `` isn't automatically on sys.path).
"""

import sys
from pathlib import Path


def ensure_importable() -> None:
    package_parent = Path(__file__).resolve().parents[2]
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))
