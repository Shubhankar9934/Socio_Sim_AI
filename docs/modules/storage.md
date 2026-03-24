# Storage Module

Buffered JSONL streaming for survey responses: append-only per-session files with configurable buffer size, time-based flush, and file rotation by size and age.

## writer.py

**Purpose**: Thread-safe buffered writer that appends survey response records to `.jsonl` files under a configurable output directory. Flushes when buffer reaches limit, on explicit flush(), or on context-manager exit. Supports file rotation by max size and max age.

### Class: JSONLWriter

**Parameters (from config.settings when not overridden)**: base_dir (jsonl_output_dir), buffer_limit (jsonl_buffer_size), flush_interval (jsonl_flush_interval), max_file_size_mb, max_file_age_hours.

| Method | Description | How |
|--------|-------------|-----|
| `__init__(base_dir, buffer_limit, flush_interval, max_file_size_mb, max_file_age_hours)` | Initialize. | Resolve paths and limits from settings; create base_dir; _buffers: session_id → list of JSON lines; _rotation_counters for rotated filenames. |
| `__enter__` / `__exit__` | Context manager. | __exit__ calls flush(). |
| `__del__` | Destructor. | flush() on delete. |
| `_safe_name(session_id)` | Safe filename segment. | Alphanumeric, dash, underscore only. |
| `_session_path(session_id)` | Current file path for session. | base_dir / "{safe}.jsonl" or "{safe}_{date}_{counter}.jsonl" when rotated. |
| `_should_time_flush()` | Whether flush interval elapsed. | time.monotonic() - _last_flush_time > _flush_interval. |
| `write_response(session_id, round_idx, response)` | Buffer one response. | record = {round, **response}; append json.dumps(record) to session buffer; if len(buf) >= buffer_limit or _should_time_flush, _flush_session(session_id). |
| `write_round(session_id, round_idx, question, question_id, responses)` | Buffer a full round. | For each response build record with round, question, question_id; extend buffer; same flush condition. |
| `flush(session_id)` | Flush to disk. | If session_id given, _flush_session(session_id); else flush all sessions. |
| `_should_rotate(path)` | Whether to rotate file. | True if path exists and (st_size >= _max_file_bytes or age >= _max_file_age_secs). |
| `_flush_session(session_id)` | Write buffer to file (caller holds lock). | Pop buffer; if _should_rotate increment _rotation_counters and recompute path; append lines to path with newline. |
| `read_session(session_id)` | Read all records from session file. | flush(session_id); read _session_path line-by-line; json.loads each line; return list of dicts. |

The `storage` package currently exposes **`writer.py` only** (no separate `__init__.py` file in the repo); import `JSONLWriter` from `storage.writer`.
