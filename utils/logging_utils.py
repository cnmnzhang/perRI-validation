"""Consistent console output for 's analysis scripts.

Library functions across `utils/` (clinical/run_clinical.py, clinical/icd.py,
progression/core.py, ...) each print their own progress in a different style --
some tagged, some bare, some raw DataFrame dumps. Rather than reformat those,
`tagged_stdout` wraps stdout for the duration of one analysis so *every* line
printed during it -- including lines from deep inside `utils/` -- gets a
single consistent "[analysis_name] " prefix. That's enough to tell, at a
glance, which of dx_incident / fig3_dx / fig5_iron_infusion / fig4_dx_cases
produced a given line when running
`python -m run_all --analysis all`.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager


class _TaggedStream:
    def __init__(self, tag: str, stream):
        self._prefix = f"[{tag}] "
        self._stream = stream
        self._at_line_start = True

    def write(self, s: str) -> int:
        n = 0
        for line in s.splitlines(keepends=True):
            if self._at_line_start:
                self._stream.write(self._prefix)
            n += self._stream.write(line)
            self._at_line_start = line.endswith("\n")
        return n

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def tagged_stdout(tag: str):
    """Prefix every line written to stdout inside this block with '[tag] '."""
    original = sys.stdout
    sys.stdout = _TaggedStream(tag, original)
    try:
        yield
    finally:
        sys.stdout = original


@contextmanager
def timed_step(tag: str, description: str):
    """Print a start/done banner (with elapsed seconds) around a block of work."""
    print(f"[{tag}] {description}...")
    t0 = time.time()
    try:
        yield
    finally:
        print(f"[{tag}] {description} done ({time.time() - t0:.1f}s)")
