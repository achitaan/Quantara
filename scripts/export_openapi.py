import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from quantara.api import create_app
from quantara.db import Store

target = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"
target.parent.mkdir(exist_ok=True)
app = create_app(store=Store("sqlite://"))
target.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
app.state.jobs.close()
print(target)
