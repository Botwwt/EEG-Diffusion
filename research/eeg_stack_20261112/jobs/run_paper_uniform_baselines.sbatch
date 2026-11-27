#!/bin/bash
set -euo pipefail
source "$SLURM_SUBMIT_DIR/environment"
run="$RAD_ROOT/uniform_reconstruction_v1"
export EEG_RUNS_ROOT="$RAD_ROOT/fits/shared"
export PYTHONPATH="$RAD_ROOT/ablation_code:$RAD_ROOT/code:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
if [[ -s "$run/runtime-base-v1.tar" ]]; then
  cache="/tmp/eeg-uniform-wheels-v1-$(id -u)"
  mkdir -m 700 -p "$cache"
  (
    flock 9
    if [[ ! -s "$cache/ready" ]]; then
      tar -xf "$run/runtime-base-v1.tar" -C "$cache"
      export PYTHONHOME="$cache/base"
      export LD_LIBRARY_PATH="$cache/base/lib:${LD_LIBRARY_PATH:-}"
      export PYTHONPATH=
      "$cache/base/bin/python3.11" -u - "$ARCHIVE/package_cache" "$cache" <<'PY'
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from sys import argv
from zipfile import ZipFile

root, cache = map(Path, argv[1:])
wheels = []
for group in ("wheels_v235", "wheels_v239", "wheels_v243"):
    wheels.extend((w, "site-packages") for w in sorted((root / group).glob("*.whl")))
wheels.extend((w, "site-packages") for w in (root / "wheels_v255").glob("torchvision-0.18.0+cu121-*.whl"))
wheels.extend((w, "backbone_runtime_v323") for w in (root / "wheels").glob("einops-0.8.1-*.whl"))
wheels.extend((w, "evaluation_runtime_v275") for w in sorted((root / "evaluation_wheels_v274").glob("*.whl")))
assert len(wheels) >= 50 and sum(w.name.startswith("torch-") for w, _ in wheels) == 1
assert sum(w.name.startswith("torchvision-") for w, _ in wheels) == 1
for wheel, destination in wheels:
    print("Staging", wheel.name, flush=True)
    with ZipFile(wheel) as zipped:
        for entry in zipped.infolist():
            if entry.is_dir():
                continue
            path = PurePosixPath(entry.filename)
            assert not path.is_absolute() and ".." not in path.parts
            if path.parts[0].endswith(".data"):
                if len(path.parts) < 3 or path.parts[1] not in {"purelib", "platlib"}:
                    continue
                path = PurePosixPath(*path.parts[2:])
            target = cache / destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(entry) as source, target.open("wb") as output:
                copyfileobj(source, output)
PY
      printf 'runtime-wheels-v1\n' > "$cache/ready"
    fi
  ) 9>"$cache/lock"
  export PYTHONHOME="$cache/base"
  export LD_LIBRARY_PATH="$cache/base/lib:${LD_LIBRARY_PATH:-}"
  packages="$cache/backbone_runtime_v323:$cache/evaluation_runtime_v275:$cache/site-packages"
  if [[ -s "$run/runtime-source-v1.tar" ]]; then
    mirror="$cache/benchmark"
    mkdir -p "$mirror"
    (
      flock 9
      if [[ ! -s "$mirror/ready" ]]; then
        tar -xf "$run/runtime-source-v1.tar" -C "$mirror"
        for entry in data models layouts fits images expected-inputs.json input-integrity.json; do
          ln -sfn "$RAD_ROOT/$entry" "$mirror/$entry"
        done
        printf 'runtime-source-v1\n' > "$mirror/ready"
      fi
    ) 9>"$mirror/lock"
    export RAD_ROOT="$mirror"
    export EEG_RUNS_ROOT="$RAD_ROOT/fits/shared"
    export PYTHONPATH="$packages:$RAD_ROOT/uniform_reconstruction_v1:$RAD_ROOT/ablation_code:$RAD_ROOT/code:$RAD_ROOT/code/Brain-HIVE"
    cd "$cache"
  else
    export PYTHONPATH="$packages:$run:$RAD_ROOT/ablation_code:$RAD_ROOT/code:$RAD_ROOT/code/Brain-HIVE"
  fi
  py="$cache/base/bin/python3.11"
fi
driver="$run/paper_uniform_baselines.py"
if [[ -n "${cache:-}" ]]; then
  cached_driver=$(mktemp "$cache/driver.XXXXXXXX.py")
  cp "$driver" "$cached_driver"
  driver="$cached_driver"
fi
methods=(brain-hive atm nice eegnet ridge)
publish() {
  local work=$1 kind=$2 budget=$3 subject=$4
  shift 4
  local target="$run/exports/$kind/$budget"
  mkdir -p "$target"
  tar -C "$work" -cf "$target/sub-$subject.rgb.tar.partial" "$@"
  mv "$target/sub-$subject.rgb.tar.partial" "$target/sub-$subject.rgb.tar"
  local list="$work/receipts.txt"
  for folder in "$@"; do
    for file in render.json scores.json observations.npz inception_official.json inception_official.npz; do
      printf '%s/%s\n' "$folder" "$file"
    done
    if [[ "$kind" == render && "$subject" == 08 ]]; then
      for file in $(seq -f '%05g.png' 0 5); do printf '%s/%s\n' "$folder" "$file"; done
    fi
  done > "$list"
  tar -C "$work" -cf "$target/sub-$subject.scores.tar.partial" -T "$list"
  mv "$target/sub-$subject.scores.tar.partial" "$target/sub-$subject.scores.tar"
}
case "$1" in
  prepare) exec "$py" -u "$driver" --run "$run" prepare ;;
  fit) exec "$py" -u "$driver" --run "$run" fit "${methods[$SLURM_ARRAY_TASK_ID]}" ;;
  render)
    subject="$SLURM_ARRAY_TASK_ID"
    budget="$2"
    shift 2
    if (($#)); then methods=("$@"); else methods+=(m1a-d2); fi
    subject_id=$(printf '%02d' "$subject")
    work=$(mktemp -d "/tmp/eeg-uniform-render-$subject_id.XXXXXXXX")
    "$py" -u "$driver" --run "$run" render --subject "$subject" --budgets "$budget" --methods "${methods[@]}" --image-root "$work/images"
    folders=()
    relative=()
    for method in "${methods[@]}"; do
      name="images/$method/$budget/sub-$subject_id"
      out="$work/$name"
      if [[ ! -s "$out/scores.json" ]]; then "$py" -u "$RAD_ROOT/code/paper_score.py" "$out" official; fi
      folders+=("$out")
      relative+=("$name")
    done
    "$py" -u "$RAD_ROOT/code/paper_inception_delta.py" "${folders[@]}" --root "$work/images" --prep "$RAD_ROOT/data/prep" --batch-size 16 --cpu-threads 4
    publish "$work" render "$budget" "$subject_id" "${relative[@]}"
    ;;
  d2-no-layout)
    subject="$SLURM_ARRAY_TASK_ID"
    subject_id=$(printf '%02d' "$subject")
    work=$(mktemp -d "/tmp/eeg-uniform-ablation-$subject_id.XXXXXXXX")
    name="ablation/shared/d2-no-layout/official-r4/sub-$subject_id"
    out="$work/$name"
    "$py" -u "$RAD_ROOT/d2_stage_reference/code/paper_render.py" --subject "$subject" --arms d2-no-layout --budgets official-r4 --output-root "$work/ablation"
    if [[ ! -s "$out/scores.json" ]]; then "$py" -u "$RAD_ROOT/code/paper_score.py" "$out" official; fi
    "$py" -u "$RAD_ROOT/code/paper_inception_delta.py" "$out" --root "$work/ablation" --prep "$RAD_ROOT/data/prep" --batch-size 16 --cpu-threads 4
    publish "$work" layout official-r4 "$subject_id" "$name"
    ;;
  stage-cogcap)
    mkdir -p "$run/cogcap"
    exec tar -xf "$run/cognitioncapturer-scorer.tar" -C "$run/cogcap"
    ;;
  score-cogcap)
    subject="$SLURM_ARRAY_TASK_ID"
    out="$run/cogcap/$(printf 'cogcap-sub%02d' "$subject")"
    if [[ ! -s "$out/scores.json" ]]; then "$py" -u "$RAD_ROOT/code/paper_score.py" "$out" official; fi
    exec "$py" -u "$run/paper_inception_delta.py" "$out" --root "$run/cogcap" --prep "$RAD_ROOT/data/prep" --batch-size 16 --cpu-threads 4
    ;;
  probe)
    exec "$py" -u -c 'import torch, torchvision, diffusers; from paper_uniform_baselines import BASELINES, JointEncoder; x=torch.zeros(2,17,250,device="cuda"); participant=torch.zeros(2,device="cuda",dtype=torch.long); print(torch.__version__, torchvision.__version__, diffusers.__version__, torch.cuda.get_device_name(), flush=True); [print(method, JointEncoder(method).cuda().eval()(x,participant).shape, flush=True) for method in BASELINES]'
    ;;
  *) exit 64 ;;
esac
