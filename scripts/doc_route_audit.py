#!/usr/bin/env python3
"""List registered FastAPI routes for cross-check with docs/DOC_INVENTORY.md.

Run from repo root:
  python scripts/doc_route_audit.py

Compare output to the HTTP/WS table in docs/DOC_INVENTORY.md after adding routes.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.app import app  # noqa: E402

try:
    from starlette.routing import WebSocketRoute
except ImportError:
    WebSocketRoute = ()  # type: ignore[misc, assignment]


def main() -> None:
    rows: list[tuple[str, str, str]] = []
    for r in app.routes:
        methods = getattr(r, "methods", None) or set()
        path = getattr(r, "path", "") or ""
        name = getattr(r, "name", "") or ""
        if "HEAD" in methods:
            methods = methods - {"HEAD"}
        if not path:
            continue
        if WebSocketRoute and isinstance(r, WebSocketRoute):
            rows.append(("WS", path, name))
        elif methods:
            for m in sorted(methods):
                rows.append((m, path, name))
        else:
            rows.append(("?", path, name))

    rows.sort(key=lambda x: (x[1], x[0]))
    print(f"{'METHOD':<8} {'PATH':<55} NAME")
    for method, path, name in rows:
        print(f"{method:<8} {path:<55} {name}")


if __name__ == "__main__":
    main()
