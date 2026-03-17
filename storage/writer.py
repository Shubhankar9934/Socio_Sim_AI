"""
Append-only JSONL writer for streaming survey responses to disk.

Each survey session gets its own ``.jsonl`` file under the configured
output directory.  Writes are buffered in memory and flushed to disk
when the buffer reaches ``buffer_limit`` records, on explicit ``flush()``,
or when the writer is used as a context manager and exits.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


class JSONLWriter:
    """Thread-safe buffered append-only JSONL writer.

    Parameters
    ----------
    base_dir : path to the output directory (created if absent).
               Defaults to the ``jsonl_output_dir`` setting.
    buffer_limit : flush to disk once a session buffer reaches this many
                   lines.  Defaults to the ``jsonl_buffer_size`` setting.
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        buffer_limit: Optional[int] = None,
        flush_interval: Optional[float] = None,
        max_file_size_mb: Optional[float] = None,
        max_file_age_hours: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.jsonl_output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_limit = buffer_limit or settings.jsonl_buffer_size
        self._flush_interval: float = flush_interval if flush_interval is not None else settings.jsonl_flush_interval
        self._max_file_bytes: int = int((max_file_size_mb if max_file_size_mb is not None else settings.jsonl_max_file_size_mb) * 1_048_576)
        self._max_file_age_secs: float = (max_file_age_hours if max_file_age_hours is not None else settings.jsonl_max_file_age_hours) * 3600
        self._last_flush_time: float = time.monotonic()
        self._lock = threading.Lock()
        self._buffers: Dict[str, List[str]] = {}
        self._rotation_counters: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "JSONLWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.flush()

    def __del__(self) -> None:
        try:
            self.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _safe_name(self, session_id: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)

    def _session_path(self, session_id: str) -> Path:
        safe = self._safe_name(session_id)
        counter = self._rotation_counters.get(session_id, 0)
        if counter == 0:
            return self.base_dir / f"{safe}.jsonl"
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.base_dir / f"{safe}_{date_str}_{counter:02d}.jsonl"

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _should_time_flush(self) -> bool:
        """Check whether the time-based flush interval has elapsed."""
        return (time.monotonic() - self._last_flush_time) > self._flush_interval

    def write_response(
        self,
        session_id: str,
        round_idx: int,
        response: Dict[str, Any],
    ) -> None:
        """Buffer one response record; auto-flushes on size or time."""
        record = {"round": round_idx, **response}
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            buf = self._buffers.setdefault(session_id, [])
            buf.append(line)
            if len(buf) >= self._buffer_limit or self._should_time_flush():
                self._flush_session(session_id)
                self._last_flush_time = time.monotonic()

    def write_round(
        self,
        session_id: str,
        round_idx: int,
        question: str,
        question_id: str,
        responses: list[Dict[str, Any]],
    ) -> None:
        """Buffer all responses from a single round."""
        lines: List[str] = []
        for r in responses:
            record = {
                "round": round_idx,
                "question": question,
                "question_id": question_id,
                **r,
            }
            lines.append(json.dumps(record, default=str, ensure_ascii=False))

        with self._lock:
            buf = self._buffers.setdefault(session_id, [])
            buf.extend(lines)
            if len(buf) >= self._buffer_limit or self._should_time_flush():
                self._flush_session(session_id)
                self._last_flush_time = time.monotonic()

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------

    def flush(self, session_id: Optional[str] = None) -> None:
        """Flush buffered lines to disk.

        If *session_id* is given, flush only that session; otherwise flush all.
        """
        with self._lock:
            if session_id is not None:
                self._flush_session(session_id)
            else:
                for sid in list(self._buffers):
                    self._flush_session(sid)

    def _should_rotate(self, path: Path) -> bool:
        """True if the current file should be rotated (size or age)."""
        if not path.exists():
            return False
        st = path.stat()
        if st.st_size >= self._max_file_bytes:
            return True
        age_secs = time.time() - st.st_mtime
        return age_secs >= self._max_file_age_secs

    def _flush_session(self, session_id: str) -> None:
        """Write buffered lines for *session_id* to disk.  Caller holds lock.

        Rotates to a new file when the current one exceeds
        ``_max_file_bytes`` or ``_max_file_age_secs``.
        """
        lines = self._buffers.pop(session_id, [])
        if not lines:
            return
        path = self._session_path(session_id)
        if self._should_rotate(path):
            self._rotation_counters[session_id] = self._rotation_counters.get(session_id, 0) + 1
            path = self._session_path(session_id)
            logger.info("Rotated JSONL file for session %s -> %s", session_id, path.name)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_session(self, session_id: str) -> list[Dict[str, Any]]:
        """Read all records from a session file (for debugging / export).

        Flushes any pending buffer first so the read is up-to-date.
        """
        self.flush(session_id)
        path = self._session_path(session_id)
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if raw_line:
                    records.append(json.loads(raw_line))
        return records
