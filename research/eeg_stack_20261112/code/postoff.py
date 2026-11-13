"""Score post-processing variants offline from a run saved with gen.py --dump 1.

python postoff.py <run name> <specs ';'-separated, as gen.py --post>  -> adds/updates info["post_off"][spec] in that run's metrics.json
"""
import json
import sys

import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur

from common import RUNS, write_json
from metrics import load_gt, score

name, specs = sys.argv[1], sys.argv[2].split(";")
out = RUNS / "img" / name
info = json.loads((out / "metrics.json").read_text())
rec = torch.from_numpy(np.load(out / "rec_u8.npy")).float() / 255
K = rec.shape[0]
f = info["args"]["init_file"]
init = torch.nn.functional.interpolate(torch.from_numpy(np.load(RUNS / "low" / f)).float(), size=512, mode="bicubic").clamp(0, 1)
_, gt = load_gt()


def lowpass(x, sg):
    ks = int(6 * sg) | 1
    return torch.cat([gaussian_blur(x[i:i + 50].cuda(), [ks, ks], [sg, sg]).cpu() for i in range(0, len(x), 50)])


po = info.setdefault("post_off", {})
for spec in specs:
    out_ = torch.empty_like(rec)
    if spec.startswith("m"):
        sg, beta = (float(v) for v in spec.split(",")[1:])
        hf = torch.stack([rec[k] - lowpass(rec[k], sg) for k in range(K)]) if sg > 0 else rec
        hm = hf.mean(0)
        for k in range(K):
            out_[k] = (rec[k] - (1 - beta) * (hf[k] - hm)).clamp(0, 1)
    else:
        w, sg, beta = (float(v) for v in spec.split(","))
        ls = lowpass(init, sg)
        for k in range(K):
            lg = lowpass(rec[k], sg)
            out_[k] = ((1 - w) * lg + w * ls + beta * (rec[k] - lg)).clamp(0, 1)
    out_ = (out_ * 255).round() / 255
    ps = score(out_, gt)
    po[spec] = {k: float(np.mean([s_[k] for s_ in ps])) for k in ps[0]}
    print(f"{name} post_off {spec} | " + " ".join(f"{k} {v:.4f}" for k, v in po[spec].items()), flush=True)
write_json(out / "metrics.json", info)
