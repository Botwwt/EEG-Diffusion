"""D14 immutable conditions, native noise/pixels, CPU sample replay and logs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_evidence_guided_prior_20261027_v1 as V
from analyze_BrainHIVE_visual_support_CPU_20261027_v1 import F_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completion',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); start=time.monotonic(); torch.set_num_threads(2)
    assert not torch.cuda.is_initialized() and not args.output.exists(); args.output.mkdir(parents=True)
    done=U.read(args.completion); stage=done['stage']; assert done['status']==f'COMPLETE_D14_STAGE{stage}'
    fit=U.read(U.checked(done['inputs'])); assert fit['updates']==1300 and fit['visual_exposures']==1323000 and fit['full_passes']==10
    inputs=U.read(U.checked(fit['D09_inputs'])); panel=U.read(U.checked(fit['panel']))
    lock_path=U.checked(done['condition_lock']); lock=U.read(lock_path)
    bank=np.load(U.checked(lock['conditions']),allow_pickle=False); assert bank.shape==(400,3,1024)
    initial,seeds=V.OLD.query_noise(panel['Dcheck'])
    assert seeds==lock['conditional_noise_seeds'] and np.array_equal(initial.numpy(),np.load(U.checked(lock['conditional_initial_noise'])))
    permutation=V.OLD.OLD.control_permutation(panel['Dcheck']); assert permutation.tolist()==lock['shuffled_EEG_query_permutation']
    assert lock['correct_and_wrong_EEG_same_initial_condition_noise'] is True
    assert all(lock[k] is False for k in ('query_target_F_features_used','query_target_images_used','target_rank_or_DINO_used_for_choices'))
    records=[json.loads(x) for x in U.checked(done['manifest']).read_text().splitlines()]
    baseline=[json.loads(x) for x in U.checked(fit['baseline_manifest']).read_text().splitlines()]
    assert len(records)==3*stage and len(baseline)==1200
    paths=set(); inodes=set(); files=set(); pixels=set(); constant=0
    def image_check(row):
        path=Path(row['generated_path']); st=path.stat()
        assert path.resolve() not in paths and (st.st_dev,st.st_ino) not in inodes
        paths.add(path.resolve()); inodes.add((st.st_dev,st.st_ino))
        assert U.sha(path)==row['generated_sha256']; files.add(row['generated_sha256'])
        with Image.open(path) as image:
            assert image.size==(512,512) and image.mode=='RGB'
            array=np.asarray(image); h=hashlib.sha256(array.tobytes()).hexdigest(); assert h==row['generated_pixel_sha256']; pixels.add(h)
            return int(np.all(array==array[0,0]))
    for ordinal,row in enumerate(records):
        i,j=divmod(ordinal,3)
        for k,value in panel['Dcheck'][i].items(): assert row[k]==value
        assert row['arm']==V.ARMS[j] and row['query_index']==i
        assert row['EEG_source_query_index']==(i if j==0 else int(permutation[i]) if j==1 else None)
        assert row['conditional_noise_seed']==seeds[i]
        assert row['conditional_initial_noise_sha256']==hashlib.sha256(initial[i:i+1].numpy().tobytes()).hexdigest()
        assert row['UNet_NFE']==row['per_query_policy_NFE']==4 and row['small_flow_velocity_evaluations']==32 and row['conditional_prior_samples']==1
        assert row['native_IP_K_calls']==row['native_IP_V_calls']==280
        assert row['normal_inference_target_access'] is False and row['scheduler_step_signature_unmodified'] is True
        assert row['extra_UNet_evaluations']==0
        for k in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'): assert row[k]==baseline[3*i][k]
        raw=np.ascontiguousarray(bank[i,j:j+1])
        assert hashlib.sha256(raw.tobytes()).hexdigest()==row['condition_sha256']
        assert hashlib.sha256(raw.astype(np.float16).tobytes()).hexdigest()==row['condition_FP16_sha256']
        assert lock_path.stat().st_mtime_ns < Path(row['generated_path']).stat().st_mtime_ns
        constant+=image_check(row)
    parities=U.read(U.checked(done['pixel_parity']))
    assert parities['status']=='BOTH_T_NATIVE_AND_FULL_DC_PARENT_EXACT_BYTES_PIXELS_PASSED'
    for row,expected in zip(parities['rows'],[inputs['parity_reference'],baseline[0]]):
        for key in ('generated_sha256','generated_pixel_sha256'): assert row['image'][key]==expected[key]
        for key in ('condition_sha256','condition_FP16_sha256','tokens_sha256'): assert row['proof'][key]==expected[key]
        for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256'): assert row['observations'][key]==expected[key]
        if 'ancestral_noise_observations' in expected: assert row['observations']['ancestral_noise_observations']==expected['ancestral_noise_observations']
        else: assert row['name']=='T_native'
        image_check(row['image'])
    generated=np.load(U.checked(done['generated_features']),allow_pickle=False)
    target_features=np.load(U.checked(fit['target_features']),allow_pickle=False)
    base_features=np.load(U.checked(fit['baseline_generated_features']),allow_pickle=False)
    assert V.score(generated,target_features,base_features,panel['Dcheck'][:stage],panel['Dcheck'],stage)==done['readout']
    if stage==400:
        previous=U.read(U.checked(done['previous']))
        assert records[:240]==[json.loads(x) for x in U.checked(previous['manifest']).read_text().splitlines()]
        assert np.array_equal(generated[:80],np.load(U.checked(previous['generated_features'])))
    # Independent FP32 CPU replay of each concept's first stimulus, both EEG arms.
    # Original full bank used BF16 GPU arithmetic. This is a numerical engineering check.
    cache=torch.load(U.checked(inputs['native_cache']),weights_only=True,map_location='cpu',mmap=True)
    lookup={int(row):i for i,row in enumerate(cache['array_rows'])}
    c=cache['condition'][[lookup[q['array_row']] for q in panel['Dcheck']]].clone()
    ckpt=torch.load(U.checked(lock['model']),weights_only=True,map_location='cpu',mmap=True)
    assert ckpt['epoch']==10 and ckpt['updates']==1300 and ckpt['visual_exposures']==1323000
    import brainhive_noisy_EEG_evidence_20261027_v1 as E
    model=V.Marginal().eval(); model.load_state_dict(ckpt['model'],strict=True)
    frozen=torch.load(U.checked(lock['frozen_D13_head']),weights_only=True,map_location='cpu',mmap=True)
    head=E.Evidence().eval().requires_grad_(False); head.load_state_dict(frozen['models']['c_only'],strict=True)
    selected=np.arange(0,stage,10); idx=torch.as_tensor(selected); subjects=torch.zeros(len(idx),dtype=torch.long)
    values=[]
    for arm in V.ARMS:
        context=c[torch.as_tensor(permutation[selected])] if arm=='guided_shuffled_c' else c[idx]
        output,_,_=V.sample(model,head if arm!='marginal_prior' else None,initial[idx],context,subjects,
            ckpt['mean'],ckpt['scale'],ckpt['F_weight'],ckpt['F_bias'])
        values.append(output)
    cpu=torch.stack(values,1).numpy(); expected=bank[selected]
    relative=np.linalg.norm(cpu-expected,axis=-1)/np.linalg.norm(expected,axis=-1)
    cosine=(cpu*expected).sum(-1)/(np.linalg.norm(cpu,axis=-1)*np.linalg.norm(expected,axis=-1))
    numeric=dict(queries=len(idx),arms=3,indices=selected.tolist(),maximum_relative_L2=float(relative.max()),
        minimum_cosine=float(cosine.min()),tolerance=dict(maximum_relative_L2=.03,minimum_cosine=.999),
        all400_conditions_numerically_replayed=False,CPU_dtype='FP32',original_GPU_dtype='BF16 velocity/FP32 integration',
        selection='first registered stimulus in each concept, both arms; no score sorting')
    U.atomic(args.output/'CPU_fixed_sample_replay.json',numeric)
    assert (relative<=.03).all() and (cosine>=.999).all(),'Independent selected sample replay unresolved; preserve all model/images and do not alter them'
    # Post-outcome F targets diagnose stimulus loss; never change the locked samples.
    contract=U.read('/root/brainhive_reproduction_20261020_v1/data/component_input_contract_v1.json')
    raw_F=torch.load(U.checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
    targets=raw_F[[q['array_row'] for q in panel['Dcheck']]].numpy()
    geometry={'native_parent':F_metrics(c.numpy(),targets,stage)}
    for j,arm in enumerate(V.ARMS): geometry[arm]=F_metrics(bank[:,j],targets,stage)
    logs=[json.loads(x) for x in (Path(lock['model']['path']).parent/'training_log.jsonl').read_text().splitlines()]
    passes=[x for x in logs if x['stage']=='COMPLETE_VISUAL_T_PASS']; held=[x for x in logs if x['stage']=='FIXED_DFIT_VELOCITY_LOG_ONLY']
    assert len(passes)==10 and len(held)==11 and [x['epoch'] for x in passes]==list(range(1,11))
    raw=np.load(U.checked(lock['raw_preprojection_u']),allow_pickle=False)
    geometry['raw_preprojection_u']={arm:dict(norm_mean=float(np.linalg.norm(raw[:stage,j],axis=-1).mean()),
        feature_mean_abs=float(np.abs(raw[:stage,j].mean(-1)).mean())) for j,arm in enumerate(V.ARMS)}
    plot=None
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4)); ax.plot([x['epoch'] for x in passes],[x['train_velocity_MSE'] for x in passes],label='T training')
        ax.plot([x['epoch'] for x in held],[x['velocity_MSE'] for x in held],label='Dfit marginal (logging only)')
        ax.set(xlabel='Completed fixed visual T passes',ylabel='Fixed velocity MSE'); ax.legend(); fig.tight_layout()
        path=args.output/'actual_training_and_held_loss.png'; fig.savefig(path,dpi=160); plt.close(fig); plot=U.ref(path)
    except ImportError: pass
    metadata=U.read(U.checked(inputs['target_metadata']))['Dcheck']
    canvas=Image.new('RGB',(7*144,34+8*164),'white'); draw=ImageDraw.Draw(canvas)
    for j,label in enumerate(('target','parent','ordinary','risk','guided c','wrong c','marginal')): draw.text((144*j+3,6),label,fill='black')
    with ExitStack() as stack:
        archives={}
        for line,i in enumerate(range(0,80,10)):
            item=metadata[i]
            if item['image_zip'] not in archives: archives[item['image_zip']]=stack.enter_context(zipfile.ZipFile(item['image_zip']))
            payload=archives[item['image_zip']].read(item['image_member']); assert hashlib.sha256(payload).hexdigest()==item['image_sha256']
            with Image.open(io.BytesIO(payload)) as image: target=image.convert('RGB')
            for j,row in enumerate([None]+baseline[i*3:i*3+3]+records[i*3:i*3+3]):
                if row is None: image=target
                else:
                    with Image.open(row['generated_path']) as opened: image=opened.convert('RGB')
                y=34+164*line; canvas.paste(ImageOps.contain(image,(140,140)),(144*j+2,y))
                if j==0: draw.text((2,y+141),item['stimulus_id'][:23],fill='black')
    sheet=args.output/'fixed_first_stimulus_8concepts.png'; canvas.save(sheet)
    diagnostic=np.load(U.checked(lock['all32_step_diagnostics']),allow_pickle=False)
    gradients=np.load(U.checked(lock['trajectory_guidance']),allow_pickle=False)
    states=np.load(U.checked(lock['trajectory_states']),allow_pickle=False)
    assert diagnostic.shape==(400,3,32,4) and gradients.shape==states.shape==(400,3,4,1024)
    assert np.isfinite(diagnostic).all() and np.isfinite(gradients).all() and np.isfinite(states).all()
    guide_stats={}
    for j,arm in enumerate(V.ARMS[:2]):
        r,h,g,v=diagnostic[:,j].transpose(2,0,1)
        ratio=g/np.maximum(v,1e-8)
        guide_stats[arm]=dict(logit_mean_by_step=r.mean(0).tolist(),unscaled_h_mean_by_step=h.mean(0).tolist(),
            guide_norm_mean_by_step=g.mean(0).tolist(),prior_norm_mean_by_step=v.mean(0).tolist(),
            guide_to_prior_norm_ratio_median=float(np.median(ratio)),guide_to_prior_norm_ratio_max=float(ratio.max()),
            t0_logit_exact_zero=bool(np.count_nonzero(r[:,0])==0))
    assert np.count_nonzero(gradients[:,2])==0 and np.count_nonzero(diagnostic[:,2])==0
    # Same generated state, change c only. This is sensitivity, not calibration.
    logits=np.empty((400,3,4,2),np.float32)
    with torch.no_grad():
        for j in range(3):
            for k,time_index in enumerate((4,12,20,28)):
                x=torch.from_numpy(states[:,j,k].copy()); times=torch.full((400,),time_index/32)
                for m,context in enumerate((c,c[torch.as_tensor(permutation)])):
                    logits[:,j,k,m]=head(x,times,context,None,torch.zeros(400,dtype=torch.long),'c_only').numpy()
    U.save_array(args.output/'same_state_correct_vs_shuffled_c_logits.npy',logits)
    normalized=bank/(np.linalg.norm(bank,axis=-1,keepdims=True)+1e-12)
    changes={name:float((1-(normalized[:stage,a]*normalized[:stage,b]).sum(-1)).mean())
        for name,a,b in [('correct_vs_shuffled_c',0,1),('correct_vs_marginal',0,2)]}
    U.atomic(args.output/'actual_guidance_diagnostic.json',dict(status='D14_ALL400_ALL32_FROZEN_LOGIT_AND_GUIDANCE_VALUES_FINITE',
        arms=guide_stats,all400_conditions_and_diagnostics=lock['all32_step_diagnostics'],
        same_state_correct_minus_shuffled_logit_by_arm={arm:(logits[:,j,:,0]-logits[:,j,:,1]).mean(0).tolist() for j,arm in enumerate(V.ARMS)},
        normalized_F_changes_same_initial_noise=changes,
        no_generated_state_probability_calibration_claim=True,correct_and_shuffled_c_are_registered_interventions=True))
    summary=dict(status='COMPLETE_D14_SOURCE_CONDITION_NOISE_PIXEL_READOUT_AND_FIXED_SAMPLE_CPU_AUDIT',
        source=U.ref(__file__),completion=U.ref(args.completion),stage=stage,query_count=stage,new_candidate_images=3*stage,
        unique_paths=len(paths),unique_inodes=len(inodes),unique_file_hashes=len(files),unique_pixel_hashes=len(pixels),constant_research_images=constant,
        all400_conditional_noise_seeds_and_arrays_exact=True,all_generated_rows_original_image_noise_and_query_identity_verified=True,
        all_generated_rows_raw_conditions_bitwise_verified=True,readout_recomputed_exactly=True,selected_sample_CPU_replay=numeric,
        both_native_parities_exact=True,legacy_T_ancestral_noise_comparison_available=False,
        all10_visual_training_passes_logged=passes,all11_Dfit_loss_diagnostics=held,F_diagnostics=geometry,actual_guidance=guide_stats,
        F_metrics_are_not_image_DINO=True,full_generation_and_model_frozen_before_postresult_F_analysis=True,
        no_model_or_samples_modified=True,training_and_held_plot=plot,contact_sheet=U.ref(sheet),
        GPU_used=False,seconds=time.monotonic()-start,independent_confirmation=False,SOTA_established=False,
        complete_human_object_accuracy_measured=False,perceptual_noninferiority_established=False)
    U.atomic(args.output/'summary.json',summary); print(json.dumps(dict(status=summary['status'],stage=stage,
        CPU_sample_replay=numeric,seconds=summary['seconds'],F_diagnostics=geometry)))


if __name__=='__main__': main()
