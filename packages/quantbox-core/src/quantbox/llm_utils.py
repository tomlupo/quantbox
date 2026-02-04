from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
from jsonschema import validate as js_validate, ValidationError

def infer_table_schema(df: pd.DataFrame) -> Dict[str, Any]:
    # Lightweight, stable schema representation
    cols = []
    for c in df.columns:
        dt = str(df[c].dtype)
        if dt.startswith("int"):
            t = "integer"
        elif dt.startswith("float"):
            t = "number"
        elif dt.startswith("bool"):
            t = "boolean"
        else:
            t = "string"
        cols.append({"name": str(c), "type": t})
    return {"columns": cols}

def load_schema(schema_path: str | Path) -> Dict[str, Any]:
    p = Path(schema_path)
    return json.loads(p.read_text(encoding="utf-8"))

def validate_table(df: pd.DataFrame, schema: Dict[str, Any]) -> List[str]:
    # Compare inferred schema to declared const columns, if present.
    warnings: List[str] = []
    inferred = infer_table_schema(df)
    try:
        js_validate(inferred, schema)
    except ValidationError as e:
        warnings.append(f"schema_validation_failed: {e.message}")
    return warnings

def event_line(event: str, **payload: Any) -> str:
    obj = {"event": event, **payload}
    return json.dumps(obj, ensure_ascii=False)
