"""D40 full-denominator image audit and pre-fixed eight-arm DINO readout."""
import argparse,csv,datetime,json,time
from pathlib import Path
import numpy as np
import brainhive_spatial_images_20261028_v1 as U

def audit_images(path,stage):
    started=time.monotonic();execution,base,queries,prepared=U.prepared(path);U.stage_gate(path,stage)
    import sys
    sys.path.insert(0,'/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages')
    from PIL import Image
    folder=U.EVAL/f'screen{stage}';folder.mkdir(exist_ok=True);assert not (folder/'image_audit.json').exists()
    bank=np.load(U.checked(prepared['conditions']),allow_pickle=False);assert bank.shape==(400,8,1024)
    originals=[json.loads(x) for x in U.checked(base['baseline_manifest']).read_text().splitlines()]
    old={r['query_id']:r for r in originals if r['arm']=='parent'};assert len(old)==400
    target_rows=U.read(U.checked(prepared['target_rows']));assert len(target_rows)==400
    inodes=set();pixels_seen=set();duplicates=constants=0;records=[]
    for i,q in enumerate(queries[:stage]):
        assert target_rows[i]['target_index']==i and all(q[k]==target_rows[i][k] for k in U.IDENTITY)
        U.checked(target_rows[i]['image'])
        for a,arm in enumerate(U.ARMS):
            record=U.EVAL/'images/records'/f'{i:03d}_{arm}.json';r=U.read(record)
            assert all(r[k]==q[k] for k in U.IDENTITY) and r['query_index']==i and r['arm']==arm and r['target_index']==i
            assert r['execution']['sha256']==U.sha(path) and r['prepared']['sha256']==U.sha(U.EVAL/'prepared/completion.json')
            assert r['condition_sha256']==U.byte_sha(bank[i,a]) and r['condition_FP16_sha256']==U.byte_sha(bank[i,a].astype(np.float16))
            for k in ['initial_UNet_input_sha256','initial_noise_latents_sha256','ancestral_noise_observations']:assert r[k]==old[q['query_id']][k]
            assert r['UNet_NFE']==4 and r['native_IP_K_calls']==r['native_IP_V_calls']==280 and len(r['ancestral_noise_observations'])==4
            assert r['scheduler_step_signature_unmodified'] and r['normal_inference_target_access'] is False and r['one_image_per_query']
            e,bg=arm.split('__');assert r['encoder_checkpoint']==prepared['models'][e]
            assert r['ordinary_checkpoint']==(prepared['ordinary_checkpoint'] if bg=='ordinary' else None)
            assert r['target_feature_binding']['sha256']==base['target_features']['sha256']
            image=Path(r['generated_path']);assert image.resolve().is_relative_to((U.EVAL/'images/pixels').resolve()) and U.sha(image)==r['generated_sha256']
            st=image.stat();inode=(st.st_dev,st.st_ino);assert inode not in inodes;inodes.add(inode)
            with Image.open(image) as im:assert im.size==(512,512);pixels=np.asarray(im.convert('RGB')).copy()
            digest=U.byte_sha(pixels);assert digest==r['generated_pixel_sha256']
            duplicates+=int(digest in pixels_seen);pixels_seen.add(digest)
            constants+=int((pixels.astype(np.float32).std(axis=(0,1))<1e-5).all());records.append(r)
    assert len(records)==stage*8
    U.write(folder/'audited_rows.json',records);U.evaluation_budget()
    v=dict(status='D40_ALL_EIGHT_ARM_PIXELS_CONDITIONS_TARGETS_AND_NOISE_CPU_AUDITED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      execution=U.ref(path),stage=stage,images=len(records),rows=U.ref(folder/'audited_rows.json'),target_rows=prepared['target_rows'],
      unique_image_inodes=len(inodes),duplicate_pixel_images_reported_not_excluded=duplicates,constant_images_reported_not_excluded=constants,
      condition_FP16_SHAs_independently_recomputed=True,token_tensor_SHAs_recorded_not_independently_recomputed=True,
      full_denominator_retained=True,seconds=time.monotonic()-started)
    U.write(folder/'image_audit.json',v);print(json.dumps(v),flush=True)

def readout(features,target,queries,stage):
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
    assert U.sha(S.__file__)=='bc9610b97501cdf078789f913b0520b023c0079801e24c37093cbac406189009'
    metrics={arm:S.metric_arrays(features[:,a],target,queries[:stage],queries) for a,arm in enumerate(U.ARMS)}
    contrasts={name:S.compare(metrics[a],metrics[b],queries[:stage]) for name,(a,b) in U.contrast_pairs().items()}
    interaction={k:(metrics['measurement__ordinary'][k]-metrics['original__ordinary'][k])-(metrics['measurement__native'][k]-metrics['original__native'][k]) for k in S.METRICS}
    contrasts['ordinary_interaction']=S.compare(interaction,{k:np.zeros(stage) for k in S.METRICS},queries[:stage])
    return metrics,contrasts,U.decision(contrasts,stage)

def score(path,stage):
    started=time.monotonic();execution,base,queries,prepared=U.prepared(path);U.stage_gate(path,stage)
    fdpath=U.EVAL/f'screen{stage}/features/completion.json';fd=U.read(fdpath)
    assert fd['status']=='D40_ALL_EIGHT_ARM_GENERATED_AND_ALL400_REAL_TARGET_DINO_BOUND' and fd['execution']['sha256']==U.sha(path)
    features=np.load(U.checked(fd['features']),allow_pickle=False);target=np.load(U.checked(fd['fresh_targets']),allow_pickle=False)
    assert features.shape==(stage,8,768) and target.shape==(400,768) and np.isfinite(features).all() and np.isfinite(target).all()
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
    selfcheck=S.metric_arrays(target,target,queries,queries);assert np.all(selfcheck['DINO2AFC']==1)
    metrics,contrasts,decision=readout(features,target,queries,stage)
    folder=U.EVAL/f'screen{stage}/score';assert not folder.exists();folder.mkdir()
    table=[dict(**q,arm=arm,**{k:float(v[i]) for k,v in metrics[arm].items()}) for i,q in enumerate(queries[:stage]) for arm in U.ARMS]
    with (folder/'per_query.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    U.write(folder/'all_per_query_metrics.json',{a:{k:v.tolist() for k,v in m.items()} for a,m in metrics.items()})
    mechanism={b:all(contrasts['measurement_vs_'+e+'__'+b]['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and contrasts['measurement_vs_'+e+'__'+b]['within_concept10_top1']['difference']>=0 for e in ['marginal','permuted_measurement']) for b in U.BACKGROUNDS}
    v=dict(status='D40_ALL_EIGHT_FIXED_IMAGE_ARMS_SCORED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution=U.ref(path),stage=stage,gallery=400,
      feature_completion=U.ref(fdpath),metric_arrays=U.ref(folder/'all_per_query_metrics.json'),table=U.ref(folder/'per_query.csv'),
      means={a:{k:float(v.mean()) for k,v in m.items()} for a,m in metrics.items()},fixed11_contrasts=contrasts,decision=decision,
      mechanism_direction_support_by_background=mechanism if stage==400 else None,interaction_descriptive_not_independence_proof=True,
      bootstrap=dict(draws=1000,seed=1729,unit='wholeconcept',intervals_descriptive=True),original_target_cache_replayed_all400=True,
      all_eight_arms_and_full_denominators_retained=True,no_control_winner_substitution=True,no_official_or_V_unlocked=True,seconds=time.monotonic()-started)
    U.write(folder/'completion.json',v);print(json.dumps(v),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution',type=Path,required=True);p.add_argument('--stage',type=int,choices=[80,400],required=True)
    p.add_argument('--mode',choices=['audit_images','score'],required=True);a=p.parse_args();audit_images(a.execution,a.stage) if a.mode=='audit_images' else score(a.execution,a.stage)
