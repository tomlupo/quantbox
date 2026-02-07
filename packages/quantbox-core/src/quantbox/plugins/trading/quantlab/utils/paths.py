# utils/paths.py
from pathlib import Path
import sys, os

def get_project_root(markers=("pyproject.toml", ".git")) -> Path:
    start = Path(__file__).resolve()
    for parent in (start, *start.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    return start.parent

PROJECT_ROOT = get_project_root()

# Ensure sys.path includes root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
