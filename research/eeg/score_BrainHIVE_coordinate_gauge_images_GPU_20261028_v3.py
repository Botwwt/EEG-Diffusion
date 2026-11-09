"""D24 score-only recovery: all1200 existing images retained, zero UNet calls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_coordinate_gauge_image_readout_20261028_v1 as V

def main():
    p=argparse.ArgumentParser()
    for k in ('spec','input-completion','native-registration','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args();assert a.max_seconds==240 and not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    try:
        spec=U.read(a.spec)
        for item in spec['source_pins']:U.checked(item)
        fit=U.read(a.input_completion);lock=U.read(U.checked(fit['condition_lock']));assert lock['ordinary_applied'] is False and lock['Dcheck_target_F_or_images_used'] is False
        failed=Path(spec['failed_render_directory']);failure=U.read(failed/'failure.json');assert "np.array_equal(generated[:80], old_features)" in failure['traceback'] and failure['active']=='DINO_NEW_IMAGES_AND_FROZEN_READOUT'
        prefix=U.read(U.checked(spec['prefix240']));old80=U.read(U.checked(spec['original80']));assert prefix['scientific_scores_read'] is False
        rows=[json.loads(s) for s in U.checked(spec['complete_render_manifest']).read_text().splitlines()];prefixrows=[json.loads(s) for s in U.checked(prefix['manifest']).read_text().splitlines()]
        features=np.load(U.checked(prefix['generated_features']),allow_pickle=False);assert features.shape==(240,3,768) and len(prefixrows)==len(features)*3==720 and len(rows)==1200 and rows[:720]==prefixrows
        assert np.array_equal(features[:80],np.load(U.checked(old80['generated_features']),allow_pickle=False))
        d09=U.read(U.checked(spec['D09_completion']));inputs=U.read(U.checked(d09['inputs']));panel=U.read(U.checked(inputs['panel']));baseline_rows=[json.loads(s) for s in U.checked(d09['check_manifest']).read_text().splitlines()]
        bank=np.load(U.checked(lock['conditions']),allow_pickle=False);assert bank.shape==(400,3,1024) and lock['query_ids']==[q['query_id'] for q in panel['Dcheck']]
        assert fit['model']==lock['model']==spec['fixed_final_model']
        for ordinal,row in enumerate(rows):
            i,j=divmod(ordinal,3);q=panel['Dcheck'][i]
            for k,v in q.items():assert row[k]==v
            assert row['arm']==V.ARMS[j] and row['query_index']==i and row['frozen_encoder_model']==fit['model'] and row['ordinary_applied'] is False
            assert U.sha(row['generated_path'])==row['generated_sha256']
            raw=np.ascontiguousarray(bank[i,j:j+1]);assert hashlib.sha256(raw.tobytes()).hexdigest()==row['condition_sha256'] and hashlib.sha256(raw.astype(np.float16).tobytes()).hexdigest()==row['condition_FP16_sha256']
            for k in ['initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations']:assert row[k]==baseline_rows[3*i][k]
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(C.read(a.native_registration))
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;cfg=C.read(a.native_registration);document=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=240,maximum_output_bytes=64<<20,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(document);budget=D.Budget(document)
        def check():assert not stop[0] and time.monotonic()-start<210;budget.check()
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);check()
        import image_evaluation as E
        assert U.sha(E.DINO_WEIGHT)==E.DINO_SHA256==U.S.DINO_SHA256
        model=E.dino_model();pieces=[];paths=[Path(v['generated_path']) for v in rows[720:]];assert len(paths)==480
        for lo in range(0,len(paths),32):check();pieces.append(E.dino_features(paths[lo:lo+32],model,batch_size=32,real_images=False).float().numpy())
        generated=np.concatenate([features,np.concatenate(pieces).reshape(160,3,768)])
        assert generated.shape==(400,3,768) and np.array_equal(generated[:len(features)],features) and np.isfinite(generated).all()
        U.save_array(a.output/'generated_features.npy',generated)
        target=np.load(U.checked(d09['check_target_features']),allow_pickle=False);baseline=np.load(U.checked(d09['check_generated_features']),allow_pickle=False)
        readout=V.score(generated,target,baseline,panel['Dcheck'],panel['Dcheck'],400)
        for arm in U.ARMS:assert readout['means'][arm]==d09['readout']['means'][arm]
        U.atomic(a.output/'readout.json',readout)
        values={arm:U.S.metric_arrays(baseline[:,j],target,panel['Dcheck'],panel['Dcheck']) for j,arm in enumerate(U.ARMS)}|{arm:U.S.metric_arrays(generated[:,j],target,panel['Dcheck'],panel['Dcheck']) for j,arm in enumerate(V.ARMS)}
        fields=['query_id','concept_id','arm']+list(next(iter(values.values())))
        with (a.output/'per_query.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i,q in enumerate(panel['Dcheck']):
                for arm,v in values.items():writer.writerow(dict(query_id=q['query_id'],concept_id=q['concept_id'],arm=arm,**{k:float(x[i]) for k,x in v.items()}))
        components=[U.read(p) for p in sorted((failed/'loader').glob('component_*.json'))]
        assert sorted(v['component'] for v in components)==['AutoencoderKL','CLIPTextModel','CLIPTextModelWithProjection','UNet2DConditionModel'] and all(v['all_tensors_exact'] for v in components)
        loader_proof=dict(status='ALL_FOUR_NATIVE_COMPONENTS_EXACT_CONVERTED_TENSOR_LOAD_VERIFIED',components=components,loading_uses_existing_official_conversion=True,frozen_FP16_weights_changed=False,metadata_only_observer=True,recovered_from_saved_original_render_loader_records=True)
        for item in spec['source_pins']:U.checked(item)
        check();done=dict(status='COMPLETE_D24_STAGE400',schema=V.SCHEMA,stage=400,inputs=U.ref(a.input_completion),spec=U.ref(a.spec),condition_lock=fit['condition_lock'],source=U.ref(__file__),wrapper=wrapper,
            manifest=spec['complete_render_manifest'],generated_features=U.ref(a.output/'generated_features.npy'),previous=spec['original80'],execution_prefix=spec['prefix240'],pixel_parity=U.ref(failed/'pixel_parity.json'),host=inputs['expected_host'],loader_proof=loader_proof,readout=readout,
            new_research_images=0,reused_research_images=1200,total_study_new_research_images=1200,new_UNet_evaluations=0,new_DINO_encoded_existing_images=480,reused_prefix_features_queries=240,engineering_parity_images=0,
            score_only_recovery=True,original_render_failure=U.ref(failed/'failure.json'),original_D_resumed=False,ordinary_applied=False,normal_inference_target_access=False,additional_inference_parameters=0,normal_inference_individual_trials=False,normal_inference_other_subject_EEG=False,EEG_or_generator_weights_updated_during_generation=False,
            seconds=time.monotonic()-start,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes)
        U.atomic(a.output/'completion.json',done);print(json.dumps(dict(status=done['status'],seconds=done['seconds'],decision=readout['decision'],new_images=0,new_UNet_calls=0)))
    except BaseException as e:
        U.atomic(a.output/'failure.json',dict(status='D24_SCORE_ONLY_RECOVERY_FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start,new_images=0,new_UNet_calls=0));raise

if __name__=='__main__':main()
