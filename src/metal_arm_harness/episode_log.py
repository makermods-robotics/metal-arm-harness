"""Append-only JSONL episode log. `EpisodeLog(None)` is a no-op sink."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class EpisodeLog:
    def __init__(self, directory: str | Path | None):
        self._file = None
        if directory is not None:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.path: Path | None = path / f"episode-{stamp}.jsonl"
            self._file = self.path.open("a", encoding="utf-8")
        else:
            self.path = None

    def event(self, kind: str, **data: Any) -> None:
        if self._file is None:
            return
        record = {"t": round(time.time(), 3), "event": kind, **data}
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
