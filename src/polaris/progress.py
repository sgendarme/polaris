from __future__ import annotations

import os
import sys
import time

_T0 = time.perf_counter()
_ENABLED = os.environ.get("POLARIS_QUIET", "") not in {"1", "true", "True"}


def enabled() -> bool:
    return _ENABLED


def log(msg: str) -> None:
    if not _ENABLED:
        return
    elapsed = time.perf_counter() - _T0
    print(f"[{elapsed:7.1f}s] {msg}", file=sys.stderr, flush=True)
