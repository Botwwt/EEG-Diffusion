"""One prespecified Dfit->Dcheck scale diagnostic, never an image candidate.

Tests whether the observed amplitude mismatch survives native projection's LN,
and whether adapted/token cosine is pairing-dependent rather than common bias.
"""
import csv
import json
import os
import random
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_native_models_v1 import *


@torch.inference_mode()
def main():
    torch.set_num_threads(2);started=time.monotonic()
    out=WORK/'audit/scale_and_pairing_CPU_v1.json';assert not out.exists()
    assert (WORK/'audit/condition_geometry_CPU_v1.json').exists()
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    manifest={int(x['array_row']):x for x in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if x['dataset_split']=='train'}
    dfit=[i for i in protocol['image_rows']['D'] if manifest[i]['concept_id'] in set(protocol['d_fit_concepts'])]
    dcheck=[i for i in protocol['image_rows']['D'] if manifest[i]['concept_id'] in set(protocol['d_check_concepts'])]
    assert len(dfit)==820 and len(dcheck)==830 and set(dfit).isdisjoint(dcheck)
    models=prior_models('cpu');a=models['fusion_encoder'].adapter;p=models['proj']
    target=torch.load(WORK/'data/fused_train_targets.pt',weights_only=True,map_location='cpu')
    records=[]
    for subject in protocol['subjects']:
        state=torch.load(WORK/'runs/full_source_dropout_v2'/subject/'conditions.pt',weights_only=True,map_location='cpu')
        lookup={int(i):j for j,i in enumerate(state['array_rows'])}
        fit_pred=state['condition'][[lookup[i] for i in dfit]]
        fit_image=target[dfit]
        # Exactly one scalar per subject. No strength grid or tuning on Dcheck.
        scale=float(a(fit_image).norm(dim=-1).mean()/a(fit_pred).norm(dim=-1).mean())
        actual=state['condition'][[lookup[i] for i in dcheck]];image=target[dcheck]
        pa,ta=a(actual),a(image);native,truth,scaled=p(pa),p(ta),p(pa*scale)
        permutation=list(range(len(dcheck)));rng=random.Random(1729)
        for i in range(len(permutation)-1,0,-1):
            j=rng.randrange(i);permutation[i],permutation[j]=permutation[j],permutation[i]
        mean=lambda x:float(x.mean())
        row=dict(subject=subject,Dfit_queries=820,Dcheck_queries=830,scale_from_Dfit_only=scale,
            matched_adapted_cosine=mean(F.cosine_similarity(pa,ta)),
            deranged_adapted_cosine=mean(F.cosine_similarity(pa[permutation],ta)),
            matched_token_cosine=mean(F.cosine_similarity(native,truth,dim=-1)),
            deranged_token_cosine=mean(F.cosine_similarity(native[permutation],truth,dim=-1)),
            scaled_token_cosine=mean(F.cosine_similarity(scaled,truth,dim=-1)),
            native_to_scaled_token_cosine=mean(F.cosine_similarity(native,scaled,dim=-1)),
            native_to_scaled_token_RMS=mean((native-scaled).square().mean((-1,-2)).sqrt()))
        assert all(__import__('math').isfinite(v) for v in row.values() if isinstance(v,float))
        records.append(row)
    keys=[k for k,v in records[0].items() if isinstance(v,float)]
    write(out,dict(status='ONE_FIXED_SCALE_AND_DCHECK_PAIRING_DIAGNOSTIC_COMPLETE',
        per_subject=records,mean={k:sum(r[k] for r in records)/len(records) for k in keys},
        seconds=time.monotonic()-started,trained_candidates=0,generated_images=0,
        V_or_benchmark_targets_read=False,scale_selection_grid=False,
        limits=['One scalar is a diagnostic of magnitude, not a complete conditional error model.',
                'A/P cosine includes a common component; compare matched versus deranged.',
                'Any image improvement would still need a separately registered matched full image comparison.']))


if __name__=='__main__':main()
