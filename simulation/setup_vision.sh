#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
python3 -m venv --system-site-packages .venv-jerry
.venv-jerry/bin/python -m pip install 'ultralytics==8.3.228' 'matplotlib>=3.10'
.venv-jerry/bin/python - <<'PY'
from pathlib import Path
import hashlib
import urllib.request

model=Path('models/yolo/yolo11n.pt')
model.parent.mkdir(parents=True,exist_ok=True)
expected='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
if not model.exists():
    temporary=model.with_suffix('.pt.part')
    urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',temporary)
    temporary.replace(model)
if hashlib.sha256(model.read_bytes()).hexdigest()!=expected:
    raise SystemExit('YOLO11n checkpoint checksum differs from the expected official release')
Path('.cache/yolo').mkdir(parents=True,exist_ok=True)
print('Pretrained YOLO11n is ready:',model)
PY
