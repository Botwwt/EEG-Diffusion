"""Math/ID invariants only; passing does not establish reconstruction quality."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,'/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import torch
import torch.nn.functional as F
from locked_metrics import fixed_gallery_metrics, VERSION

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
indices=protocol['image_rows']['T'][:64]
x=torch.load('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2/train_image_embeds.pt',weights_only=True,mmap=True)[indices].clone()
ids=[f'fixture-stimulus-{i}' for i in range(len(x))]
self_result=fixed_gallery_metrics(x,x,ids,ids,batch_size=7)
assert torch.equal(self_result['two_afc'],torch.ones(64))
scores=F.normalize(x.float(),dim=-1) @ F.normalize(x.float(),dim=-1).T
diagonal=scores.diag()
legacy=((scores<diagonal[:,None]).sum(1)+.5*((scores==diagonal[:,None]).sum(1)-1))/63
assert torch.equal(self_result['two_afc'],legacy)
# Multiple subjects repeat stimuli and exceed the number of gallery columns.
queries=torch.cat([x,x.flip(0)]);query_ids=ids+ids[::-1]
multi=fixed_gallery_metrics(queries,x,query_ids,ids,batch_size=11)
assert len(multi['two_afc'])==128 and torch.equal(multi['two_afc'],torch.ones(128))
permutation=torch.arange(64).roll(17)
reordered=fixed_gallery_metrics(queries,x[permutation],query_ids,[ids[i] for i in permutation],batch_size=13)
assert torch.equal(multi['two_afc'],reordered['two_afc'])
uniform=[]
for shift in range(64):
    shuffled=ids[shift:]+ids[:shift]
    result=fixed_gallery_metrics(x,x,shuffled,ids,batch_size=16)
    uniform.append(result['two_afc'].mean())
assert abs(float(torch.stack(uniform).mean())-.5)<1e-6
ties=fixed_gallery_metrics(torch.ones(3,8),torch.ones(4,8),['a','b','c'],['a','b','c','d'])
assert torch.equal(ties['two_afc'],torch.full((3,),.5))
receipt=dict(status='MATH_AND_ID_INVARIANTS_PASSED',metric_version=VERSION,
    source_sha256=hashlib.sha256((ROOT/'locked_metrics.py').read_bytes()).hexdigest(),
    actual_fixture='First64 existing T image CLIP features, used only for metric math; not DINO extraction or generated-image evidence',
    original_self_2afc=1.,all_cyclic_label_shifts_mean=float(torch.stack(uniform).mean()),
    repeated_subject_query_rows=128,unique_gallery_images=64,all_ties_2afc=.5,
    diagonal_legacy_formula_equivalent=True,gallery_ordering_and_batching_invariant=True,
    new_formal_image_evaluations=0)
(ROOT/'audit/locked_metrics_math_checks.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
