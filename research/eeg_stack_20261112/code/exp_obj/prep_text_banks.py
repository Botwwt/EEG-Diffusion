import os, re, json, time, torch
import torch.nn.functional as F
import torchvision.models as tvM
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModelWithProjection

t0 = time.time()
RAD = os.environ.get("RAD_ROOT", "/path/to/rad")
OUT = f"{RAD}/runs/obj_bench/text_banks"
os.makedirs(OUT, exist_ok=True)
dev = "cuda:0"

# 1. Load THINGS train/test concept names (STRICTLY separate: train1654 for training ONLY, test200+train for EVAL ONLY!)
test_raw = json.load(open(f"{RAD}/data/prep/test_ids.json"))
train_raw = json.load(open(f"{RAD}/data/prep/train_ids.json"))
def path_to_concept(p):
    folder = p.split("/")[1]
    name = folder.split("_", 1)[1] if "_" in folder else folder
    return re.sub(r"\d+$", "", name).replace("_", " ").strip()

test_names = [path_to_concept(p) for p in test_raw["paths"]]
train_names_per_img = [path_to_concept(p) for p in train_raw["paths"]]
uniq_train = []
seen_tr = set()
for c in train_names_per_img:
    if c not in seen_tr:
        seen_tr.add(c); uniq_train.append(c)
tr_name2id = {c: i for i, c in enumerate(uniq_train)}
tr_img2cid = torch.tensor([tr_name2id[c] for c in train_names_per_img], dtype=torch.long)

# Eval-only 1823 vocabulary (200 test + 1623 train_only)
seen_all = set(test_names)
train_only_eval = [c for c in uniq_train if c not in seen_all]
eval_1823 = test_names + train_only_eval

# 2. Build ZERO-LEAKAGE CV-OpenVocab (OpenImages + LVIS + ImageNet-1K — 100% independent of THINGS!)
cv_raw = json.load(open(f"{RAD}/data/prep/cv_openvocab.json"))
in1k_cats = [c.lower().strip() for c in tvM.Inception_V3_Weights.DEFAULT.meta["categories"]]
cv_open = sorted(set(cv_raw["openimages_lvis"]) | set(in1k_cats))
cv_lvis_in1k = sorted(set(cv_raw["lvis"]) | set(in1k_cats))

# Report overlap stats purely for transparent documentation in paper (NOT used to filter!)
tset = set(test_names)
print(f"THINGS: test={len(test_names)}, uniq_train={len(uniq_train)}, eval_1823={len(eval_1823)}")
print(f"CV-OpenVocab (OpenImages+LVIS+IN1K): {len(cv_open)} concepts (exact-string overlap with 200 test: {len(tset & set(cv_open))}/200)")
print(f"CV-Compact (LVIS+IN1K): {len(cv_lvis_in1k)} concepts (exact-string overlap with 200 test: {len(tset & set(cv_lvis_in1k))}/200)", flush=True)

# 3. Canonical 27 THINGS high-level categories (Hebart et al. 2019, for eval + optional train-only supervision)
CAT27 = [
    "animal", "bird", "body part", "clothing", "clothing accessory",
    "container", "dessert", "drink", "electronic device", "food",
    "fruit", "furniture", "home decor", "insect", "kitchen appliance",
    "kitchen tool", "medical equipment", "musical instrument", "office supply", "part of car",
    "plant", "sports equipment", "tool", "toy", "vegetable", "vehicle", "weapon"
]

TEMPLATES = [
    "a photo of a {}",
    "a clear photograph of a {}",
    "a close-up photo of a {}",
    "an image of a {}",
]

tok2 = CLIPTokenizer.from_pretrained(f"{RAD}/models/sdxl-turbo/tokenizer_2")

def encode_h14_multi(names, model):
    vecs = None
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for tmpl in TEMPLATES:
            prompts = [tmpl.format(n) for n in names]
            parts = []
            for i in range(0, len(prompts), 512):
                inp = tok2(prompts[i:i+512], padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(dev)
                out_t = model.text_model(input_ids=inp["input_ids"])
                eot = inp["input_ids"].argmax(dim=-1)
                pooled = out_t.last_hidden_state[torch.arange(eot.shape[0], device=dev), eot]
                proj = model.text_projection(pooled)
                parts.append(F.normalize(proj.float(), dim=-1).cpu())
            v = torch.cat(parts, dim=0)
            vecs = v if vecs is None else vecs + v
    return F.normalize(vecs, dim=-1)

def encode_bigg_multi(names, model):
    vecs = None
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for tmpl in TEMPLATES:
            prompts = [tmpl.format(n) for n in names]
            parts = []
            for i in range(0, len(prompts), 512):
                inp = tok2(prompts[i:i+512], padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(dev)
                proj = model(input_ids=inp["input_ids"]).text_embeds
                parts.append(F.normalize(proj.float(), dim=-1).cpu())
            v = torch.cat(parts, dim=0)
            vecs = v if vecs is None else vecs + v
    return F.normalize(vecs, dim=-1)

print("Encoding with LAION CLIP-ViT-H-14 on GPU...", flush=True)
h14 = CLIPModel.from_pretrained(f"{RAD}/models/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.float16).to(dev).eval()
h14_train = encode_h14_multi(uniq_train, h14)
h14_eval1823 = encode_h14_multi(eval_1823, h14)
h14_cv_open = encode_h14_multi(cv_open, h14)
h14_cv_comp = encode_h14_multi(cv_lvis_in1k, h14)
h14_cat27 = encode_h14_multi(CAT27, h14)
del h14; torch.cuda.empty_cache()
print(f"  H14 done in {time.time()-t0:.1f}s", flush=True)

print("Encoding with SDXL OpenCLIP-bigG/14 on GPU...", flush=True)
bigg = CLIPTextModelWithProjection.from_pretrained(
    f"{RAD}/models/sdxl-turbo/text_encoder_2", variant="fp16", torch_dtype=torch.float16
).to(dev).eval()
bigg_train = encode_bigg_multi(uniq_train, bigg)
bigg_eval1823 = encode_bigg_multi(eval_1823, bigg)
bigg_cv_open = encode_bigg_multi(cv_open, bigg)
bigg_cv_comp = encode_bigg_multi(cv_lvis_in1k, bigg)
bigg_cat27 = encode_bigg_multi(CAT27, bigg)
del bigg; torch.cuda.empty_cache()
print(f"  bigG done in {time.time()-t0:.1f}s", flush=True)

torch.save({
    "uniq_train": uniq_train,
    "tr_img2cid": tr_img2cid,
    "test_names": test_names,
    "eval_1823": eval_1823,
    "cv_open": cv_open,
    "cv_lvis_in1k": cv_lvis_in1k,
    "cat27": CAT27,
    "h14_train": h14_train,
    "h14_eval1823": h14_eval1823,
    "h14_cv_open": h14_cv_open,
    "h14_cv_comp": h14_cv_comp,
    "h14_cat27": h14_cat27,
    "bigg_train": bigg_train,
    "bigg_eval1823": bigg_eval1823,
    "bigg_cv_open": bigg_cv_open,
    "bigg_cv_comp": bigg_cv_comp,
    "bigg_cat27": bigg_cat27,
}, f"{OUT}/text_banks_laion.pt")
print(f"Saved {OUT}/text_banks_laion.pt in {time.time()-t0:.1f}s!", flush=True)
