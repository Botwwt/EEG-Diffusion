"""Full allowed T/OOF and D condition geometry; never reads V/test targets.

This is a deployment-interface audit, not a new trained candidate or an image
metric. A/P are exactly the frozen public native modules, evaluated in FP32.
"""
import csv
import json
import os
import random
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_native_models_v1 import *


def stats(x):
    x=x.float()
    assert x.ndim==1 and torch.isfinite(x).all()
    return dict(mean=float(x.mean()),std=float(x.std()),
                quantiles_05_50_95=torch.quantile(x,torch.tensor([.05,.5,.95])).tolist())


@torch.inference_mode()
def geometry(predicted,target,adapter,projector):
    result={key:[] for key in ['raw_cosine','adapted_cosine','token_cosine',
        'raw_RMS_error','adapted_RMS_error','token_RMS_error',
        'predicted_raw_norm','target_raw_norm','predicted_adapted_norm','target_adapted_norm']}
    for p,t in zip(predicted.split(256),target.split(256)):
        pa,ta=adapter(p),adapter(t)
        pp,tp=projector(pa),projector(ta)
        values=dict(raw_cosine=F.cosine_similarity(p,t),adapted_cosine=F.cosine_similarity(pa,ta),
            token_cosine=F.cosine_similarity(pp,tp,dim=-1).mean(-1),
            raw_RMS_error=(p-t).square().mean(-1).sqrt(),
            adapted_RMS_error=(pa-ta).square().mean(-1).sqrt(),
            token_RMS_error=(pp-tp).square().mean((-1,-2)).sqrt(),
            predicted_raw_norm=p.norm(dim=-1),target_raw_norm=t.norm(dim=-1),
            predicted_adapted_norm=pa.norm(dim=-1),target_adapted_norm=ta.norm(dim=-1))
        for key,value in values.items():
            assert torch.isfinite(value).all()
            result[key].append(value)
    return {key:torch.cat(values) for key,values in result.items()}


def main():
    torch.set_num_threads(2)
    started=time.monotonic()
    output=WORK/'audit/condition_geometry_CPU_v1.json'
    assert not output.exists(),'Keep the original audit'
    assert (WORK/'audit/all40_parent_training_CPU_verification_v1.json').exists()
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    image_rows={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
    train=protocol['image_rows']['T'];development=protocol['image_rows']['D'];forbidden=set(protocol['image_rows']['V'])
    dfit_concepts={str(c) for c in protocol['d_fit_concepts']}
    dcheck_concepts={str(c) for c in protocol['d_check_concepts']}
    dfit=[i for i in development if str(image_rows[i]['concept_id']) in dfit_concepts]
    dcheck=[i for i in development if str(image_rows[i]['concept_id']) in dcheck_concepts]
    assert len(train)==13230 and len(dfit)==820 and len(dcheck)==830
    assert set(dfit).isdisjoint(dcheck) and set(dfit+dcheck)==set(development)
    assert set(train+dfit+dcheck).isdisjoint(forbidden)
    target_path=WORK/'data/fused_train_targets.pt'
    contract=json.loads((WORK/'audit/fused_target_contract.json').read_text())
    assert sha(target_path)==contract['target_sha256']
    targets=torch.load(target_path,weights_only=True,map_location='cpu')
    assert targets.shape==(16540,1024)
    models=prior_models('cpu');adapter=models['fusion_encoder'].adapter;projector=models['proj']
    proof=json.loads((WORK/'audit/all40_parent_training_CPU_verification_v1.json').read_text())
    verified={(p['role'],p['subject']):p for p in proof['models']}
    summaries=[];rows_by_train_id={i:p for p,i in enumerate(train)}
    for subject in sorted({p['subject'] for p in proof['models']}):
        native=torch.load(WORK/'runs/full_source_dropout_v2'/subject/'conditions.pt',weights_only=True,map_location='cpu')
        assert native['model_sha256']==verified['full',subject]['checkpoint_sha256']
        assert sha(WORK/'runs/full_source_dropout_v2'/subject/'conditions.pt')==verified['full',subject]['condition_cache_sha256']
        native_lookup={int(row):p for p,row in enumerate(native['array_rows'])}
        oof=torch.empty(len(train),1024);coverage=torch.zeros(len(train),dtype=torch.int16)
        sources=[]
        for fold in [0,1]:
            role='oof'+str(fold);directory=WORK/'runs'/(role+'_source_dropout_v2')
            reg=json.loads((directory/'registration.json').read_text())
            data=torch.load(directory/subject/'conditions.pt',weights_only=True,map_location='cpu')
            assert data['model_sha256']==verified[role,subject]['checkpoint_sha256']
            assert sha(directory/subject/'conditions.pt')==verified[role,subject]['condition_cache_sha256']
            trained_concepts={image_rows[i]['concept_id'] for i in reg['training_array_rows']}
            held_concepts={image_rows[int(i)]['concept_id'] for i in data['array_rows']}
            assert trained_concepts.isdisjoint(held_concepts)
            positions=[rows_by_train_id[int(i)] for i in data['array_rows']]
            oof[positions]=data['condition'];coverage[positions]+=1
            sources.append(dict(role=role,model_sha256=data['model_sha256'],held_rows=len(positions),concept_disjoint=True))
        assert torch.all(coverage==1)
        for split,ids,predicted in [
            ('T_OOF',train,oof),('T_in_fit_diagnostic',train,native['condition'][[native_lookup[i] for i in train]]),
            ('Dfit',dfit,native['condition'][[native_lookup[i] for i in dfit]]),
            ('Dcheck',dcheck,native['condition'][[native_lookup[i] for i in dcheck]])]:
            assert set(ids).isdisjoint(forbidden)
            target=targets[ids]
            values=geometry(predicted,target,adapter,projector)
            permutation=list(range(len(ids)));rng=random.Random(1729)
            for i in range(len(permutation)-1,0,-1):
                j=rng.randrange(i);permutation[i],permutation[j]=permutation[j],permutation[i]
            assert all(i!=j for i,j in enumerate(permutation))
            wrong=F.cosine_similarity(predicted[permutation],target)
            summary=dict(subject=subject,split=split,queries=len(ids),statistics={k:stats(v) for k,v in values.items()},
                         deranged_raw_cosine=stats(wrong),raw_pairing_advantage=float((values['raw_cosine']-wrong).mean()))
            summaries.append(summary)
            save_tensor(WORK/'data/condition_geometry_v1'/f'{subject}_{split}.pt',dict(
                subject=subject,split=split,array_rows=torch.tensor(ids),values=values,
                source_conditions='Real complete concept-OOF' if split=='T_OOF' else 'Frozen full-T parent',
                oof_sources=sources if split=='T_OOF' else [],sampled_new_images=False))
        print(json.dumps(dict(stage='CPU_NATIVE_GEOMETRY',subject=subject,completed_splits=4,seconds=time.monotonic()-started)),flush=True)
    aggregate={}
    for split in ['T_OOF','T_in_fit_diagnostic','Dfit','Dcheck']:
        selected=[s for s in summaries if s['split']==split]
        assert len(selected)==10
        aggregate[split]={k:sum(s['statistics'][k]['mean'] for s in selected)/10 for k in selected[0]['statistics']}
        aggregate[split]['deranged_raw_cosine']=sum(s['deranged_raw_cosine']['mean'] for s in selected)/10
        aggregate[split]['queries']=sum(s['queries'] for s in selected)
    write(output,dict(status='ALL10_FULL_ALLOWED_T_OOF_AND_D_NATIVE_GEOMETRY_AUDITED_CPU',
        aggregate=aggregate,per_subject=summaries,seconds=time.monotonic()-started,
        excluded_V_rows=len(forbidden),benchmark_or_test_targets_read=False,
        extra_EEG_preprocessing=False,diagnostic_precision='FP32 A/P; cached F was eval BF16 stored FP32',
        normal_inference_target_access=False,trained_new_candidates=0,
        limits=['Cosine and norm are condition geometry, not image correctness.',
                'OOF encoders see half T; full parent sees all T, so risk calibration must check deployment shift.',
                'A train/generalization gap alone is not proof of insufficient capacity or its cause.',
                'Predictive conditional error is not physiological signal quality or SNR.']))
    print('FULL_ALLOWED_NATIVE_GEOMETRY_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
