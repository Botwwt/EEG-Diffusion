"""Text prompts from EEG by retrieval over the 1654 *training* concepts (disjoint from the 200 test concepts).

python prompt.py <cond .npy rel. to runs/cond> <out name> [k=1] [template]
  -> runs/prompt/<out name>.json : 200 prompts, e.g. "a photo of a dog" (k=1) or "a photo of a dog or a wolf" (k=2)
Concept prototype = mean of the normalised fused targets F of its 10 training images. Similarity = cosine between the
test condition and each prototype, both centred by their own mean (label-free). No test label or test image is used.
"""
import json
import sys

import numpy as np
import torch

from common import IMG, PREP, RUNS

nz = torch.nn.functional.normalize
cond = torch.from_numpy(np.load(RUNS / "cond" / sys.argv[1])).float()
name, k = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1
tmpl = sys.argv[4] if len(sys.argv) > 4 else "a photo of {}"
concepts = [p.name.split("_", 1)[1].replace("_", " ") for p in sorted((IMG / "training_images").iterdir())]
cid = torch.from_numpy(np.load(PREP / "train_concept.npy")).long()
F = nz(torch.from_numpy(np.load(PREP / "F_train.npy")).float(), dim=1)
proto = torch.zeros(len(concepts), F.shape[1]).index_add_(0, cid, F)
proto = nz(proto - proto.mean(0), dim=1)
q = nz(nz(cond, dim=1) - nz(cond, dim=1).mean(0), dim=1)
top = (q @ proto.T).topk(k, dim=1).indices
art = lambda w: ("an " if w[0] in "aeiou" else "a ") + w  # noqa: E731
prompts = [tmpl.format(" or ".join(art(concepts[j]) for j in row)) for row in top.tolist()]
(RUNS / "prompt").mkdir(parents=True, exist_ok=True)
(RUNS / "prompt" / f"{name}.json").write_text(json.dumps(prompts, indent=0))
test = [p.name.split("_", 1)[1].replace("_", " ") for p in sorted((IMG / "test_images").iterdir())]
ids = json.loads((PREP / "test_ids.json").read_text())["ids"]
print(name, "| examples (true test concept -> prompt):")
for i in range(0, 200, 20):
    print(f"   {ids[i]:<28} -> {prompts[i]}")
