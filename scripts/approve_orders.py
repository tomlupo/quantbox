from __future__ import annotations
import json
from pathlib import Path
import argparse
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True, help="Path to artifacts/<run_id>/")
ap.add_argument("--who", default="human")
ap.add_argument("--note", default="approved")
args = ap.parse_args()

run_dir = Path(args.run_dir)
od = run_dir / "orders_digest.json"
if not od.exists():
    raise SystemExit("orders_digest.json not found in run dir")

digest = json.loads(od.read_text(encoding="utf-8"))["orders_digest"]
out_dir = Path("approvals")
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"{digest}.json"

payload = {
    "approved": True,
    "orders_digest": digest,
    "who": args.who,
    "when": datetime.now(timezone.utc).isoformat(),
    "note": args.note,
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote approval:", out)
