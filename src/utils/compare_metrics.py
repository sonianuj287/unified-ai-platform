import json
import sys
from pathlib import Path

metrics_file = Path(sys.argv[1])

# Simple rule: higher accuracy or lower loss = improved
if not metrics_file.exists():
    print("false")
    sys.exit(0)

with open(metrics_file) as f:
    new_metrics = json.load(f)

old_file = Path("metrics/last_eval.json")
if not old_file.exists():
    print("true")
    sys.exit(0)

with open(old_file) as f:
    old_metrics = json.load(f)

new_loss = new_metrics.get("loss", 1e9)
old_loss = old_metrics.get("loss", 1e9)

if new_loss < old_loss:
    Path("metrics/last_eval.json").write_text(json.dumps(new_metrics, indent=2))
    print("true")
else:
    print("false")
