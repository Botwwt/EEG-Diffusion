"""Real-input preflight and all400 image readout for a frozen-output probe."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import brainhive_relative_fixed_output_contract_20261103_v1 as K


def main(args):
    started=time.monotonic();ctx=K.setup(args.policy);s=ctx['s'];r=Path(s['output_root'])
    before=K.resources(ctx,starting=args.mode=='prepare');a=K.assets(ctx);rel=ctx['relative']
    sys.path.insert(0,rel['repository_root'])
    import numpy as np
    import torch
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    if args.mode=='prepare':
        out=r/'CPU_preflight';assert not out.exists();out.mkdir()
        from scipy.special import erf
        from brainhive_bounded_component_v1 import BoundedConditionAdapter
        cache=torch.load(K.checked(a['inputs']['native_cache']),weights_only=True,map_location='cpu',mmap=True)
        parent=torch.load(K.checked(a['inputs']['native_checkpoint']),weights_only=True,map_location='cpu')['model']
        pos={int(v):i for i,v in enumerate(cache['array_rows'])}
        native=cache['condition'][[pos[v['array_row']] for v in a['queries']]].numpy().copy()
        old=np.load(K.checked(a['d09']['check_conditions']),allow_pickle=False)
        assert old.shape==(400,3,1024) and np.array_equal(native,old[:,0])
        gamma=parent['block.layers.0.norm.weight'].numpy().copy()
        beta=parent['block.layers.0.norm.bias'].numpy().copy()
        u=(native.astype(np.float64)-beta.astype(np.float64))/gamma.astype(np.float64)
        norm=np.linalg.norm(u,axis=1,keepdims=True)
        np.save(r/'native400.npy',native,allow_pickle=False)
        np.savez(r/'real_affine.npz',gamma=gamma,beta=beta)
        references={};expected=[];reports={};rr=Path(rel['output_root']);cached_candidates={}
        previous_feature_done=K.read(rr/'images/features_000_080_80/completion.json')
        for arm in K.ARMS:
            done=K.read(rr/arm/'completion.json');state=torch.load(K.checked(done['component']),weights_only=True,map_location='cpu')
            assert state['arm']==arm and state['seed']==1729 and state['updates']==24807
            assert state['seen_queries']==396900 and state['initial_checkpoint']==rel['starting_F']
            assert state['total_historical_updates']==49614 and state['total_historical_exposures']==793800
            params={k:v.numpy().astype(np.float64) for k,v in state['model'].items()}
            assert sum(v.size for v in params.values())==525568
            h=u@params['trunk.0.weight'].T+params['trunk.0.bias']
            h=.5*h*(1+erf(h/np.sqrt(2)))
            d=h@params['direction.weight'].T+params['direction.bias']
            d-=d.mean(1,keepdims=True);d-=u*(d*u).sum(1,keepdims=True)/norm**2
            delta=.25*norm*d/np.sqrt(norm**2+(d*d).sum(1,keepdims=True))
            scale=norm/np.linalg.norm(u+delta,axis=1,keepdims=True)
            value=native.astype(np.float64)+gamma*((scale-1)*u+scale*delta)
            model=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False)
            model.load_state_dict(state['model'],strict=True);model.radius=.25
            with torch.inference_mode():
                actual=model(torch.from_numpy(native),torch.from_numpy(gamma[None].copy()).expand(400,-1),
                             torch.from_numpy(beta[None].copy()).expand(400,-1),gate_mode='fixed')[0].numpy()
            scaled=float(np.max(abs(actual-value)/(1+abs(value))));assert scaled<2e-6
            np.savez(out/f'{arm}_model_arrays.npz',**{k:v.numpy() for k,v in state['model'].items()})
            expected.append(value)
            references[arm]=dict(component=done['component'],old_selected_strength=state['calibration']['selected_strength'],
                                 fixed_new_diagnostic_strength=1,radius=.25,new_updates=0)
            reports[arm]=dict(NumPy64_CPU32_scaled_max_error=scaled,old_calibration_retained=True,
                fixed_raw1_nonzero_changed_queries=int(np.any(actual!=native,axis=1).sum()))
            if state['calibration']['selected_strength']==1:
                old_conditions=rr/'images'/f'conditions400_{arm}.npy'
                old_features=previous_feature_done['features'][arm];K.checked(old_features)
                cached_candidates[arm]=dict(conditions=K.ref(old_conditions),features80=old_features,
                    records_root=str(rr/'images/records'),only_reuse_if_all400_FP32_values_bit_identical=True)
        assert references[K.ARMS[0]]['old_selected_strength']==0
        assert references[K.ARMS[1]]['old_selected_strength']==.75
        assert all(references[v]['old_selected_strength']==1 for v in K.ARMS[2:])
        np.save(r/'expected_raw1_conditions64.npy',np.stack(expected,axis=1),allow_pickle=False)
        # Pin every existing starting-F image identity before any new image.
        froot=Path(rel['starting_F_features400']['path']).parent;frows=[]
        for i,v in enumerate(a['queries']):
            p=froot/'records'/f'{i:03d}_F_only.json';row=K.read(p)
            assert row['query_id']==v['query_id'] and row['component']==rel['starting_F']
            assert row['selected_strength']==1 and row['UNet_NFE']==4
            assert K.ref(row['generated_path'])['sha256']==row['generated_sha256']
            frows.append(row)
        K.atomic(r/'starting_F_records.json',frows)
        K.atomic(r/'queries.json',a['queries']);K.atomic(r/'component_references.json',references)
        K.atomic(r/'cached_candidates.json',cached_candidates)
        source=dict(status='PASS_REAL_FROZEN4_RAW1_HEADS_ALL400_NUMPY64_AND_BASELINE_IDENTITIES',
            policy=K.ref(args.policy),queries=400,arms=list(K.ARMS),
            observed_nonzero_raw1_changed_queries=sum(v['fixed_raw1_nonzero_changed_queries'] for v in reports.values()),
            reports=reports,expected=K.ref(r/'expected_raw1_conditions64.npy'),native=K.ref(r/'native400.npy'),
            affine=K.ref(r/'real_affine.npz'),models=references,original_calibrated_method_and_STOP_unchanged=True,
            all400_starting_F_image_identity_and_SHA=True,new_training_updates=0,new_GPU_jobs=0,
            resources=K.resources(ctx),seconds=time.monotonic()-started)
        K.atomic(out/'completion.json',source);print(json.dumps(source));return
    assert args.mode=='score'
    from PIL import Image
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
    import score_low_level as L
    condition=K.read(r/'condition_completion.json')
    values={arm:np.load(K.checked(condition['conditions'][arm]),allow_pickle=False) for arm in K.ARMS}
    base={arm:[a['old_records'][3*i+j] for i in range(400)] for j,arm in enumerate(K.BASELINES[:3])}
    base['starting_F']=K.read(r/'starting_F_records.json')
    records={**base,**{arm:[K.read(r/'records'/f'{i:03d}_{arm}.json') for i in range(400)] for arm in K.ARMS}}
    seen_paths=set();seen_inodes=set();pixels={};all_rows=[]
    noise_keys=('initial_UNet_input_sha256','initial_noise_latents_sha256','ancestral_noise_observations')
    for arm,rows in records.items():
        for i,row in enumerate(rows):
            assert row['query_id']==a['queries'][i]['query_id']
            assert all(row[k]==base['parent'][i][k] for k in noise_keys)
            path=Path(row['generated_path']);actual=K.ref(path)
            assert actual['sha256']==row['generated_sha256'] and str(path) not in seen_paths
            inode=(path.stat().st_dev,path.stat().st_ino);assert inode not in seen_inodes
            seen_paths.add(str(path));seen_inodes.add(inode)
            with Image.open(path) as im:
                data=np.asarray(im.convert('RGB'));assert data.shape==(512,512,3)
            pixel_sha=hashlib.sha256(data.tobytes()).hexdigest();assert pixel_sha==row['generated_pixel_sha256']
            pixels.setdefault(pixel_sha,[]).append([arm,i])
            if arm in K.ARMS:
                assert row['policy']==K.ref(args.policy) and row['fixed_strength']==1 and row['normal_inference_target_access'] is False
                assert row['condition_sha256']==hashlib.sha256(values[arm][i:i+1].tobytes()).hexdigest()
                assert row['condition_FP16_sha256']==hashlib.sha256(values[arm][i:i+1].astype(np.float16).tobytes()).hexdigest()
            all_rows.append(dict(method=arm,query_index=i,**row))
    K.atomic(r/'audited_all3200_records.json',all_rows)
    features=K.read(r/'features_completion.json')['features']
    target=np.load(K.checked(a['d09']['check_target_features']),allow_pickle=False)
    bfeatures=np.load(K.checked(a['d09']['check_generated_features']),allow_pickle=False)
    feature_arrays={arm:bfeatures[:,j] for j,arm in enumerate(K.BASELINES[:3])}
    feature_arrays['starting_F']=np.load(K.checked(rel['starting_F_features400']),allow_pickle=False)
    for arm in K.ARMS:feature_arrays[arm]=np.load(K.checked(features[arm]),allow_pickle=False)
    metrics={arm:S.metric_arrays(v.astype(np.float64),target.astype(np.float64),a['queries'],a['queries']) for arm,v in feature_arrays.items()}
    # Independently derive every rank from float64 normalized score matrices.
    t=target.astype(np.float64);t/=np.linalg.norm(t,axis=1,keepdims=True)
    matrices=[];maximum=0.
    for arm,x in feature_arrays.items():
        x=x.astype(np.float64);x/=np.linalg.norm(x,axis=1,keepdims=True);score=x@t.T;matrices.append(score)
        for i,qv in enumerate(a['queries']):
            row=score[i];truth=row[i];near=[j for j,v in enumerate(a['queries']) if v['concept_id']==qv['concept_id']]
            assert len(near)==10
            rebuilt=(truth,(np.sum(row<truth)+.5*(np.sum(row==truth)-1))/399,float(row.argmax()==i),
                     float(near[int(row[near].argmax())]==i),(np.sum(row[near]<truth)+.5*(np.sum(row[near]==truth)-1))/9)
            for k,v in zip(K.METRICS[:5],rebuilt):maximum=max(maximum,abs(v-metrics[arm][k][i]))
    assert maximum<1e-12
    for arm in records:
        for key in K.METRICS[5:]:metrics[arm][key]=np.empty(400,np.float64)
    for i,meta in enumerate(a['metadata']):
        with zipfile.ZipFile(meta['image_zip']) as archive:raw=archive.read(meta['image_member'])
        assert hashlib.sha256(raw).hexdigest()==meta['image_sha256'] and len(raw)==int(meta['image_bytes'])
        trgb,ty=L.pixels(io.BytesIO(raw));tm=L.moments(ty)
        for arm,rows in records.items():
            rgb,y=L.pixels(rows[i]['generated_path'])
            metrics[arm]['PixCorr_RGB'][i]=L.pixcorr(rgb,trgb)
            metrics[arm]['SSIM_Y'][i]=L.ssim_y(y,ty,tm)
            metrics[arm]['mean_RGB_absolute_error'][i]=float(abs(rgb.mean((0,1))-trgb.mean((0,1))).mean())
        if i%40==39:
            assert time.monotonic()-started<800
            K.atomic(r/'score_progress.json',dict(complete_queries=i+1,seconds=time.monotonic()-started))
    comparisons=K.fixed_pairs(metrics,a['queries'])
    # Independent bootstrap implementation uses count weights instead of gathered means.
    groups=[[i for i,v in enumerate(a['queries']) if v['concept_id']==c] for c in sorted({v['concept_id'] for v in a['queries']})]
    draws=np.random.default_rng(1729).integers(0,40,size=(1000,40))
    weights=np.array([np.bincount(d,minlength=40) for d in draws])/40
    maximum_bootstrap=0.
    for other in (*K.BASELINES,*K.ARMS[1:]):
        for key in K.METRICS:
            delta=metrics[K.ARMS[0]][key]-metrics[other][key]
            cm=np.array([delta[g].mean() for g in groups]);ci=np.quantile(weights@cm,[.025,.975])
            saved=comparisons[K.ARMS[0]+'-'+other][key]
            maximum_bootstrap=max(maximum_bootstrap,float(abs(ci-saved['descriptive_concept_bootstrap95']).max()))
    assert maximum_bootstrap<1e-12
    raw={arm+'__'+key:v for arm,fields in metrics.items() for key,v in fields.items()}
    for other in (*K.BASELINES,*K.ARMS[1:]):
        for key in K.METRICS:raw[K.ARMS[0]+'-'+other+'__'+key]=metrics[K.ARMS[0]][key]-metrics[other][key]
    np.savez_compressed(r/'all400_metrics_and_differences.npz',**raw)
    np.save(r/'all8_actual_float64_score_matrices.npy',np.stack(matrices,axis=1),allow_pickle=False)
    main=comparisons[K.ARMS[0]+'-ordinary']
    summary=dict(status='COMPLETE_FROZEN_RAW1_ALL400_ALL8METHODS_ALL8METRICS',policy=K.ref(args.policy),
        queries=400,gallery=400,methods=list(feature_arrays),metrics=list(K.METRICS),
        means={a:{k:float(v.mean()) for k,v in fields.items()} for a,fields in metrics.items()},fixed_comparisons=comparisons,
        raw=K.ref(r/'all400_metrics_and_differences.npz'),score_matrices=K.ref(r/'all8_actual_float64_score_matrices.npy'),
        all3200_images_actual_identity_SHA_pixels_noise_audited=True,
        unique_paths=len(seen_paths),unique_inodes=len(seen_inodes),exact_pixel_duplicate_groups=[v for v in pixels.values() if len(v)>1],
        all1600_new_protocol_conditions_checked=True,independent_all16000_semantic_values_max_error=maximum,
        independent_all56_intervals_max_error=maximum_bootstrap,
        interpretation=dict(full400_diagnostic_complete=True,old_calibrated_STOP_unchanged=True,
            semantic_effect_vs_ordinary=main['DINO2AFC'],stimulus_effect_vs_ordinary=main['within_concept10_two_afc'],
            all_secondary_positive_and_adverse_metrics_retained=True,posthoc_development_not_independent_confirmation=True,
            no_automatic_new_fit=True,no_new_second_method_or_SOTA_established=True),
        low_level_source=K.ref(Path(rel['repository_root'])/'score_low_level.py'),seconds=time.monotonic()-started,
        resources=K.resources(ctx),new_training_updates=0,NEW_contacted=False)
    K.atomic(r/'readout400.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=('prepare','score'),required=True)
    main(ap.parse_args())
