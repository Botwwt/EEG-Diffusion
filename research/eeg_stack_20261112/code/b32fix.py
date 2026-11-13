"""Linear correction for the small-CLIP branch of the candidate embedder (fenc.py): the published Brain-HIVE
"CLIP-ViT-B-32" embeddings are a fixed linear function of the downloaded laion/CLIP-ViT-B-32-laion2B-s34B-b79K output.
Fitted by least squares on every 4th training image (about 4,100); the last 600 of them are held out for the check.
Test images are only used to report the cosine after the fit (the matrix is refitted on training images alone).

python b32fix.py   -> data/prep/b32_fix.pt  [512, 512]   (run after prep.py targets)
"""
import json

import torch
from PIL import Image
from torchvision import transforms as T
from transformers import CLIPVisionModelWithProjection

from common import EMB_KEYS, IMG, PREP, ROOT

nz = torch.nn.functional.normalize
k = EMB_KEYS[1]
m = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / k), torch_dtype=torch.float16).cuda().eval()
cn = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
tf = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(512), T.ToTensor()])


def emb(paths):
    out, ok, buf = [], [], []

    def flush():
        if buf:
            x = cn(torch.nn.functional.interpolate(torch.stack(buf).cuda(), size=224, mode="bicubic", antialias=True).clamp(0, 1)).half()
            with torch.no_grad():
                out.append(m(pixel_values=x).image_embeds.float().cpu())
            buf.clear()
    for j, p in enumerate(paths):
        try:
            buf.append(tf(Image.open(p).convert("RGB")))
            ok.append(j)
        except OSError:
            print("skipped unreadable image", p.name)
        if len(buf) == 100:
            flush()
    flush()
    return torch.cat(out), ok


tr_ids = json.loads((PREP / "train_ids.json").read_text())["ids"]
files = {f.stem: f for f in (IMG / "training_images").glob("*/*.jpg")}
sel = list(range(0, 16540, 4))
e, ok = emb([files[tr_ids[i]] for i in sel])
e = e.double()
r = torch.load(PREP / "emb_train.pt", weights_only=True)[k][[sel[j] for j in ok]].double()
n = len(e) - 600
W = torch.linalg.lstsq(e[:n], r[:n]).solution
p = e[n:] @ W
print(len(e), "images | held-out training cosine after the fit", float((nz(p, dim=1) * nz(r[n:], dim=1)).sum(1).mean()))
W = torch.linalg.lstsq(e, r).solution
te_ids = json.loads((PREP / "test_ids.json").read_text())["ids"]
tfiles = {f.stem: f for f in (IMG / "test_images").glob("*/*.jpg")}
et, _ = emb([tfiles[i] for i in te_ids])
rt = torch.load(PREP / "emb_test.pt", weights_only=True)[k].double()
print("test cosine after the fit", float((nz(et.double() @ W, dim=1) * nz(rt, dim=1)).sum(1).mean()))
torch.save(W.float(), PREP / "b32_fix.pt")
