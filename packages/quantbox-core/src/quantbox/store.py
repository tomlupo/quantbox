from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd

class FileArtifactStore:

    def __init__(self, root: str, run_id: str):
        self._run_id = run_id
        self.root = Path(root) / run_id
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def run_id(self) -> str:
        return self._run_id

    def get_path(self, name: str) -> str:
        return str(self.root / name)

    def put_parquet(self, name: str, df: pd.DataFrame) -> str:
        path = self.root / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return str(path)

    def put_json(self, name: str, obj: Dict[str, Any]) -> str:
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        return str(path)

    def append_event(self, line: str) -> str:
        path = self.root / "events.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return str(path)

