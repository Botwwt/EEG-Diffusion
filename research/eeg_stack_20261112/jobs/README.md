# Job scripts (as run for this folder's experiments)

Shell wrappers that queued the experiments on 8-GPU nodes without a scheduler (tmux sessions, one or two jobs per
card). They are kept for the record; none of them is needed to reproduce the results (the commands in `../README.md`
are enough). They assume the working layout used here:

```text
$RAD_ROOT/code/rad/     = ../code/        (scripts)
$RAD_ROOT/code/jobs/    = this folder
$RAD_ROOT/env/          Python virtual environment (env.sh)
$RAD_ROOT/data, models, runs, logs, tmp, cache
```

Server paths were replaced by `$RAD_ROOT`. A helper that kept an idle card reserved between jobs was removed, together
with the lines that started it; the queue runners are otherwise unchanged. GPU indices inside the scripts are the ones
that were free at the time.

`downloads/`: the URL lists `dl.sh` / `dl_enc.sh` read (target folders relative to `$RAD_ROOT`).

| script | purpose |
|---|---|
| `chain_g3.sh` | wait for the g2 image queue to finish, then run the g3 queue (rung.sh) |
| `check40.sh` | quick image check of the joint encoder without the head: calibration 0.5 + start image (compare with t0 ..._cal0.5_low) |
| `deg.sh` | degraded-image targets (deg.py) and the subject-8 encoders trained on them (negative result) |
| `dl.sh` | download data, embeddings and models with aria2c from the lists in `downloads/` (copy them to `$RAD_ROOT/tmp/`; hf-mirror.com can be replaced by huggingface.co); the per-image list `aria_images.txt` is not included — it is the file listing of the `Haitao999/things-eeg` Image_set folder |
| `dl_enc.sh` | download the two OpenCLIP image towers used by the candidate embedder |
| `env.sh` | create the uv virtual environment (Python 3.10) and install the packages, plus OpenAI CLIP for the metrics |
| `envr.sh` | `source envr.sh <gpu>`: the cache / offline environment of the workers and `$P` = the project Python (used for the 2026-11-13 evening jobs) |
| `g12.sh` | start images for subject 8 (low.py), then an image queue on one card |
| `joinq.sh` | joinq.sh <queue> <session prefix> <gpu>...: when session <prefix><gpu> has ended and the card is idle, start a pw.sh worker there |
| `launch05.sh` | joint-encoder epoch screen plus an amplification image queue |
| `low13.sh` | start images for subjects 1 and 3, then an image queue |
| `lowgen.sh` | lowgen.sh <start-image tag> "<gpus>": wait for runs/low/sub-XX/<tag>_n0.8.npy for all subjects, then head + start image |
| `more_subs.sh` | download + prepare the remaining subjects one at a time; the raw 63-channel files are removed after preparation |
| `ms_screen.sh` | wait for the sweep on GPU 4, then screen joint-encoder variants (condition level only; seconds to minutes each) |
| `ms_screen2.sh` | ms_screen2.sh <gpu> <arg string>...: run train_ms.py variants one after another |
| `newenc.sh` | newenc.sh <gpu> <subs...>: for the joint encoder ms_e10_ft10in, train the co-author's head, apply it, generate 3 arms |
| `ordchain.sh` | ordchain.sh <gpu> <sub> [<sub> ...]: per subject, train the ordinary head, apply it to the test conditions, then generate |
| `ordw.sh` | ordw.sh <gpu>: take subjects one at a time (shared lock dir): train the ordinary head (first without gradient |
| `ordwork.sh` | ordwork.sh <gpu> <train/wait:SESSION> <sub> [<sub> ...] |
| `postpool.sh` | postpool.sh <head name> <gpu>...: wait for the shared head, apply it to all ten subjects, generate four arms per subject |
| `postpool2.sh` | postpool2.sh <head name> <gpu>...: wait for the shared head, apply it, generate A, B, A', B' (all best of 4) |
| `pw.sh` | pw.sh <gpu> <abs queue file> [script=gen.py]: pop one line at a time (flock) and run it; several workers may share a queue |
| `rg.sh` | rg.sh <gpu> <abs queue file> [parallel=2] [script=gen.py]: each line = arguments of the script. |
| `rq.sh` | rq.sh <gpu> <abs queue file>: training queue (lines = train.py arguments), then low.py for the subjects listed after it. |
| `rung.sh` | rung.sh <abs queue file>: each line = arguments of gen.py; runs on GPU 5 next to the training queue. |
| `rungp.sh` | rungp.sh <abs queue file> [parallel=2]: each line = arguments of gen.py; runs P jobs at a time on GPU 5. |
| `runq.sh` | runq.sh <queue file>: run each line as "python train.py <line>" on GPU 5. |
| `runq2.sh` | runq2.sh <abs queue file>: training queue that can share GPU 5 with an image-generation queue. |
| `runq3.sh` | runq3.sh <abs queue file>: training queue on GPU 5. |
| `st05.sh` | st05.sh "<subs>" "<gpus>": low-level operating point (img2img strength 0.5) with and without best of 4, one job per card at a time |
| `sweep.sh` | sweep.sh "<subs>" "<gpus>": queue the semantic sweep on top of head + low-level start, round-robin over idle cards |
| `sweep2.sh` | sweep2.sh "<subs>" "<gpus>": CLIP / SSIM levers on the joint encoder + per-subject head, one job per card at a time |
