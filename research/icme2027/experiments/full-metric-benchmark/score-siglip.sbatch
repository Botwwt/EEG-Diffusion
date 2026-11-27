#!/bin/bash
set -euo pipefail

study="$1"
anchor="$SLURM_SUBMIT_DIR"
source <(awk '/^umask 077/{exit} {print}' "$anchor/pair-loss-r80-20261127/run-pair-loss-r80.sbatch")
umask 077
export PYTHONDONTWRITEBYTECODE=1
work="$(mktemp -d /tmp/eeg-siglip-eval.XXXXXXXX)"
trap 'rm -rf -- "$work"' EXIT
"$py" - "$study/packages" "$work" <<'PY'
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile

packages, target = map(Path, sys.argv[1:])
wheel_hashes = {}
for line in (packages / "runtime.sha256").read_text().splitlines():
    digest, name = line.split()
    if Path(name).name != name or not name.endswith(".whl") or name in wheel_hashes:
        raise ValueError("Invalid package lock entry")
    wheel_hashes[name] = digest
if set(wheel_hashes) != {path.name for path in packages.glob("*.whl")}:
    raise ValueError("Package lock does not cover all dependencies")
for name, digest in wheel_hashes.items():
    payload = (packages / name).read_bytes()
    if sha256(payload).hexdigest() != digest:
        raise ValueError(f"Different bytes for {name}")
    with ZipFile(BytesIO(payload)) as source:
        for entry in source.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe member in a registered dependency wheel")
            source.extract(entry, target)
PY
export PYTHONPATH="$work:$PYTHONPATH"
export HF_HUB_OFFLINE=1
printf -v subject 'sub-%02d' "$SLURM_ARRAY_TASK_ID"
methods=(brain-hive atm nice eegnet ridge m1a-d2)
for budget in r4 r80; do
  archive="$anchor/uniform_reconstruction_v1/exports/render/official-$budget/$subject.rgb.tar"
  "$py" -u "$study/code/paper_independent_naming.py" \
    --model "$study/model" --assets "$study/naming-assets" --archive "$archive" \
    --output "$study/siglip-results/official-$budget-$subject.json" \
    --subjects "$SLURM_ARRAY_TASK_ID" --arms "${methods[@]}" \
    --official-budget "$budget" --uniform-archive --device cuda
done
