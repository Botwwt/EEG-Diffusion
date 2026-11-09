"""Image/identity CPU audit and unchanged fixed-gallery DINO decision for D37."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import time
import numpy as np
import brainhive_temporal_image_screen_20261028_v1 as U


def audit_images(spec_path,stage):
    start=time.monotonic();spec,prepared,queries,_=U.prepared(spec_path);U.stage_gate(spec_path,stage)
    import sys
    sys.path[:0]=['/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages','/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages']
    from PIL import Image
    output=U.B/f'screen{stage}';output.mkdir(exist_ok=True);dest=output/'image_audit.json';assert not dest.exists()
    old=[json.loads(x) for x in U.check(spec['baseline_manifest']).read_text().splitlines()];base={(r['query_id'],r['arm']):r for r in old}
    condition=np.load(U.check(prepared['files']['ordinary_conditions400']),allow_pickle=False)
    direction=np.load(U.check(prepared['files']['temporal_tokens400']),allow_pickle=False)
    donor=np.load(U.check(prepared['files']['same_concept_donor400']),allow_pickle=False)
    scaling=np.load(U.check(prepared['files']['wrong_RMS_scale400']),allow_pickle=False)
    records=[];inodes=set();spatial_constant=0
    for i,query in enumerate(queries[:stage]):
        for arm in U.ARMS:
            path=U.B/'screen_images_v1/records'/(f'{i:03d}_{arm}.json');row=U.read(path);reference=base[(query['query_id'],'ordinary')]
            assert all(row[k]==query[k] for k in U.IDENTITY) and row['query_index']==i and row['arm']==arm
            assert row['spec']['sha256']==U.sha(spec_path) and row['condition_sha256']==U.byte_sha(condition[i])==reference['condition_sha256']
            assert row['tokens_sha256']==reference['tokens_sha256'] and row['initial_UNet_input_sha256']==reference['initial_UNet_input_sha256']
            assert row['initial_noise_latents_sha256']==reference['initial_noise_latents_sha256'] and row['ancestral_noise_observations']==reference['ancestral_noise_observations']
            assert row['UNet_NFE']==4 and row['native_IP_K_calls']==row['native_IP_V_calls']==280
            assert row['additional_temporal_K_calls']==row['additional_temporal_V_calls']==280 and row['active_IP_layers']==70 and row['total_active_IP_calls']==280
            j=int(donor[i]) if arm=='same_concept_wrong' else i
            d=direction[j:j+1].copy()
            if arm=='same_concept_wrong':d*=scaling[i:i+1,:,None,None]
            assert row['temporal_direction_sha256']==U.byte_sha(d) and row['temporal_EEG_donor_index']==j and row['temporal_EEG_donor_query_id']==queries[j]['query_id']
            if arm!='flat_residual':assert row['native_marginal_max_error']<2e-6
            image_path=Path(row['generated_path']);assert image_path.resolve().is_relative_to((U.B/'screen_images_v1/pixels').resolve())
            assert U.sha(image_path)==row['generated_sha256'];stat=image_path.stat();inode=(stat.st_dev,stat.st_ino);assert inode not in inodes;inodes.add(inode)
            with Image.open(image_path) as image:
                assert image.size==(512,512);pixels=np.asarray(image.convert('RGB')).copy()
            assert U.byte_sha(pixels)==row['generated_pixel_sha256']
            if (pixels.astype(np.float32).std(axis=(0,1))<1e-5).all():spatial_constant+=1
            records.append(row)
    assert len(records)==stage*3
    U.write(output/'audited_image_rows.json',records)
    value=dict(status='D37_ALL_STAGE_IMAGE_PIXELS_AND_INPUTS_CPU_AUDITED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,
        images=len(records),spec=U.ref(spec_path),prepared=U.ref(U.B/'prepared_image_v1/completion.json'),rows=U.ref(output/'audited_image_rows.json'),
        unique_image_inodes=len(inodes),all_query_condition_temporal_donor_tokens_initial_and_ancestral_noise_SHAs_verified=True,
        constant_images_reported_without_exclusion=spatial_constant,full_denominator_preserved=True,target_images_decoded=0,seconds=time.monotonic()-start)
    U.write(dest,value);print(json.dumps(value),flush=True)


def score(spec_path,stage):
    start=time.monotonic();spec,prepared,queries,_=U.prepared(spec_path);U.stage_gate(spec_path,stage)
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
    assert U.sha(S.__file__)=='bc9610b97501cdf078789f913b0520b023c0079801e24c37093cbac406189009'
    output=U.B/f'screen{stage}/score';assert not output.exists();output.mkdir()
    features=U.read(U.B/f'screen{stage}/features/completion.json');assert features['status']=='D37_ALL_THREE_IMAGE_ARMS_DINO_CACHED' and features['stage']==stage
    candidate=np.load(U.check(features['features']),allow_pickle=False);assert candidate.shape==(stage,3,768)
    target=np.load(U.check(spec['target_features']),allow_pickle=False);baseline=np.load(U.check(spec['baseline_features']),allow_pickle=False)
    assert target.shape==(400,768) and baseline.shape==(400,3,768)
    selfcheck=S.metric_arrays(target,target,queries,queries);assert np.all(selfcheck['DINO2AFC']==1) and np.allclose(selfcheck['target_cosine'],1,atol=2e-5)
    names=list(U.BASE_ARMS)+list(U.ARMS);metrics={}
    for a,arm in enumerate(U.BASE_ARMS):metrics[arm]=S.metric_arrays(baseline[:stage,a],target,queries[:stage],queries)
    for a,arm in enumerate(U.ARMS):metrics[arm]=S.metric_arrays(candidate[:,a],target,queries[:stage],queries)
    paired={arm:S.compare(metrics[arm],metrics['ordinary'],queries[:stage]) for arm in U.ARMS}
    controls={arm:S.compare(metrics['conditional'],metrics[arm],queries[:stage]) for arm in ['flat_residual','same_concept_wrong']}
    action=S.decision(paired['conditional'],str(stage))
    mechanism_direction=all(v['DINO2AFC']['difference']>0 and v['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 for v in controls.values())
    table=[]
    for i,query in enumerate(queries[:stage]):
        for arm in names:table.append({**query,'arm':arm,**{k:float(v[i]) for k,v in metrics[arm].items()}})
    with (output/'per_query.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    U.write(output/'all_per_query_metrics.json',metrics_to_lists(metrics))
    value=dict(status=f'D37_ALL_THREE_FIXED{stage}_IMAGE_ARMS_SCORED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,queries=stage,gallery_targets=400,
        spec=U.ref(spec_path),source=U.ref(__file__),frozen_metric_source=U.ref(S.__file__),feature_completion=U.ref(U.B/f'screen{stage}/features/completion.json'),
        target_features=spec['target_features'],baseline_features=spec['baseline_features'],rows=U.ref(output/'per_query.csv'),metric_arrays=U.ref(output/'all_per_query_metrics.json'),
        means={arm:{k:float(v.mean()) for k,v in m.items()} for arm,m in metrics.items()},paired_against_old_ordinary=paired,primary_minus_matched_controls=controls,
        primary_decision=action,mechanism_direction_supported_in_this_development_panel=mechanism_direction,
        all_control_results_reported=True,control_winner_not_substituted=True,multiple_candidate_optional_stopping_not_nominal_confirmation=True,
        bootstrap=dict(draws=1000,seed=1729,unit='whole concepts with all10 stimuli; unchanged400 target gallery',descriptive_only=True),
        target_selfcheck={k:float(v.mean()) for k,v in selfcheck.items()},new_model_fits=0,new_GPU_jobs=0,seconds=time.monotonic()-start)
    U.write(output/'completion.json',value)
    print(json.dumps(dict(status=value['status'],stage=stage,means=value['means'],primary_decision=action,controls=controls)),flush=True)


def metrics_to_lists(metrics):return {arm:{k:v.tolist() for k,v in m.items()} for arm,m in metrics.items()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--stage',type=int,choices=[80,400],required=True);p.add_argument('--mode',choices=['audit_images','score'],required=True);a=p.parse_args()
    audit_images(a.spec,a.stage) if a.mode=='audit_images' else score(a.spec,a.stage)
