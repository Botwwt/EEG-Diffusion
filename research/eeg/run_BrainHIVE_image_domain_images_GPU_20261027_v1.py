"""D19 locked postprocessing latents decoded without any new UNet sampling."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,signal,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_image_domain_readout_20261027_v1 as V
import fit_BrainHIVE_image_domain_GPU_20261027_v2 as FIT
import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N

def main():
    p=argparse.ArgumentParser()
    for key in ('fit-completion','image-spec','native-registration','previous','output'):p.add_argument('--'+key,type=Path)
    p.add_argument('--stage',type=int,choices=(80,400),required=True);p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();invoked=time.monotonic();assert not a.output.exists();a.output.mkdir(parents=True);assert (a.stage==400)==bool(a.previous)
    stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));active='INPUTS';new_rows=[]
    def log(value):
        value=value|dict(seconds=time.monotonic()-invoked)
        with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(value)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(value),flush=True)
    try:
        fit=U.read(a.fit_completion);assert fit['status']=='D19_THREE_FIXED_IMAGE_DOMAIN_HEADS_AND_ALL400_LOCKED_ENDPOINT_EDITS_COMPLETE'
        spec=U.read(a.image_spec);assert spec['schema']==V.SCHEMA
        for item in spec['source_pins']:U.checked(item)
        lock_path=U.checked(fit['edit_lock']);lock_sha=U.sha(lock_path);lock=U.read(lock_path);assert tuple(lock['arms'])==V.ARMS and lock['Dcheck_target_pixels_F_or_scores_used'] is False
        assert fit['model']==spec['fixed_model']==lock['model'];U.checked(fit['model'])
        bank=np.load(U.checked(lock['latents']),allow_pickle=False,mmap_mode='r');assert bank.shape==(400,4,4,64,64) and np.isfinite(bank).all()
        prepared=U.read(U.checked(fit['prepared_inputs']));mapping=U.read(U.checked(prepared['mapping']));source=mapping['Dcheck400']
        d09=U.read(U.checked(spec['D09_completion']));inputs=U.read(U.checked(d09['inputs']));panel=U.read(U.checked(inputs['panel']));queries=panel['Dcheck']
        assert lock['query_ids']==[q['query_id'] for q in queries] and len(source)==400
        baseline=np.load(U.checked(d09['check_generated_features']));target=np.load(U.checked(d09['check_target_features']));old_rows=[json.loads(line) for line in U.checked(d09['check_manifest']).read_text().splitlines()]
        prefix=[];old_features=None
        if a.previous:
            previous=U.read(a.previous);assert previous['status']=='COMPLETE_D19_STAGE80' and previous['readout']['decision']['action']=='EXPAND400'
            assert previous['fit']==U.ref(a.fit_completion) and previous['edit_lock']==fit['edit_lock']
            prefix=[json.loads(line) for line in U.checked(previous['manifest']).read_text().splitlines()];assert len(prefix)==320
            old_features=np.load(U.checked(previous['generated_features']));assert old_features.shape==(80,4,768)
            for i,row in enumerate(prefix):assert row['query_id']==queries[i//4]['query_id'] and row['arm']==V.ARMS[i%4] and U.sha(row['generated_path'])==row['generated_sha256']
        import brainhive_paired_study_contract_20261025_v1 as C
        config=C.read(a.native_registration);C.validate(config);C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;document=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=a.max_seconds,maximum_output_bytes=1024<<20,maximum_extra_CPU_bytes=10<<30,
            maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
        wrapper=D.verify_wrapper(document);budget=D.Budget(document)
        def check():assert not stopped[0] and time.monotonic()-invoked<a.max_seconds-30,'D19 finite image budget';budget.check()
        torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(1729);torch.cuda.reset_peak_memory_stats();torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;C.resources(config,training_start=True)
        active='NATIVE_VAE_LOAD';pipe,loader,unet_calls,vae_proof=FIT.load_vae(a.output,inputs['expected_host']);check()
        parity=U.read(U.checked(fit['VAE_parity']));pz=torch.from_numpy(np.load(U.checked(parity['latent']))).cuda()
        saved=N.save_image(a.output/'parity_common_VAE_decode.webp',FIT.decode(pipe,pz)[0])
        for key in ('generated_sha256','generated_pixel_sha256'):assert saved[key]==parity['image'][key]
        U.atomic(a.output/'pixel_parity.json',dict(status='EXACT_FIXED_FP32_COMMON_VAE_DECODE_BYTES_AND_PIXELS_PASSED',image=saved,reference=fit['VAE_parity']))
        U.atomic(a.output/'provenance.json',dict(schema=V.SCHEMA,source=U.ref(__file__),fit=U.ref(a.fit_completion),spec=U.ref(a.image_spec),edit_lock=fit['edit_lock'],wrapper=wrapper,
            cached_parent_sources_reused=True,new_UNet_sampling=False,new_noise_draws=False,normal_inference_target_access=False,original_D_resumed=False))
        active='FOUR_LOCKED_ENDPOINT_LATENT_ARMS_DECODE';records=list(prefix);manifest=a.output/'inference_manifest.jsonl';decode_started=time.monotonic()
        with manifest.open('x') as stream,ThreadPoolExecutor(max_workers=1) as executor:
            for row in prefix:stream.write(json.dumps(row)+'\n')
            for i in range(0 if a.stage==80 else 80,a.stage):
                check();q=queries[i];original=old_rows[3*i]
                for key,value in q.items():assert original[key]==source[i][key]==value
                assert original['generated_sha256']==source[i]['generated']['sha256'];U.checked(source[i]['generated'])
                z=torch.from_numpy(np.array(bank[i],copy=True)).cuda();images=FIT.decode(pipe,z);assert len(images)==4
                futures=[]
                for j,arm in enumerate(V.ARMS):
                    latent=np.ascontiguousarray(bank[i,j:j+1]);path=a.output/arm/f'{i:03d}_{q["array_row"]:05d}.webp'
                    row=q|dict(arm=arm,query_index=i,source_image=source[i]['generated'],source_pixel_sha256=source[i]['generated_pixel_sha256'],
                        source_initial_noise_sha256=original['initial_noise_latents_sha256'],source_initial_UNet_input_sha256=original['initial_UNet_input_sha256'],
                        source_ancestral_noises=original['ancestral_noise_observations'],latent_sha256=hashlib.sha256(latent.tobytes()).hexdigest(),edit_model=fit['model'],
                        endpoint_radius=.1 if arm!='vae_roundtrip' else 0.,small_CNN_gradient_calls=8 if arm!='vae_roundtrip' else 0,
                        original_source_UNet_NFE=4,new_UNet_NFE=0,additional_VAE_encode_images=1,additional_VAE_decode_images=1,
                        no_new_sampling_noise=True,normal_inference_only_current_query_EEG_and_generated_image=True,normal_inference_target_access=False)
                    futures.append((executor.submit(N.save_image,path,images[j]),row))
                for future,row in futures:
                    row=row|future.result();records.append(row);new_rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                if (i+1)%20==0:log(dict(stage=active,queries=i+1,expected=a.stage,new_images=len(new_rows)))
        decode_seconds=time.monotonic()-decode_started;assert len(records)==4*a.stage and records[:len(prefix)]==prefix and unet_calls[0]==0
        active='FROZEN_DINO_AND_READOUT';import image_evaluation as E
        assert U.sha(E.DINO_WEIGHT)==E.DINO_SHA256==U.S.DINO_SHA256;dino=E.dino_model();pieces=[];paths=[Path(row['generated_path']) for row in new_rows]
        for lo in range(0,len(paths),32):check();pieces.append(E.dino_features(paths[lo:lo+32],dino,batch_size=32,real_images=False).float().numpy())
        generated=np.concatenate(pieces).reshape(len(new_rows)//4,4,768)
        if old_features is not None:generated=np.concatenate([old_features,generated]);assert np.array_equal(generated[:80],old_features)
        U.save_array(a.output/'generated_features.npy',generated);readout=V.score(generated,target,baseline,queries[:a.stage],queries,a.stage);U.atomic(a.output/'readout.json',readout)
        perquery={arm:U.S.metric_arrays(baseline[:a.stage,j],target,queries[:a.stage],queries) for j,arm in enumerate(U.ARMS)}
        perquery|={arm:U.S.metric_arrays(generated[:,j],target,queries[:a.stage],queries) for j,arm in enumerate(V.ARMS)}
        with (a.output/'per_query.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['query_id','concept_id','arm']+list(U.S.METRICS));writer.writeheader()
            for i,q in enumerate(queries[:a.stage]):
                for arm,value in perquery.items():writer.writerow(dict(query_id=q['query_id'],concept_id=q['concept_id'],arm=arm,**{key:float(v[i]) for key,v in value.items()}))
        for ref in spec['source_pins']:U.checked(ref)
        assert U.sha(lock_path)==lock_sha and unet_calls[0]==0;check()
        complete=dict(status=f'COMPLETE_D19_STAGE{a.stage}',schema=V.SCHEMA,source=U.ref(__file__),spec=U.ref(a.image_spec),fit=U.ref(a.fit_completion),edit_lock=fit['edit_lock'],
            stage=a.stage,manifest=U.ref(manifest),generated_features=U.ref(a.output/'generated_features.npy'),pixel_parity=U.ref(a.output/'pixel_parity.json'),previous=U.ref(a.previous) if a.previous else None,
            readout=readout,loader_proof=loader,VAE=vae_proof,wrapper=wrapper,new_research_images=len(new_rows),reused_research_images=len(prefix),new_parity_images=1,
            original_source_images_reused=320 if a.stage==400 else 80,source_sampling_UNet_NFE_per_query=4,new_UNet_forward_calls=0,new_initial_or_ancestral_noise_draws=0,
            normal_inference_extra_VAE_encodes=1,normal_inference_extra_VAE_decodes=1,normal_inference_small_CNN_gradient_calls=8,
            cached_latent_decoding_excludes_already_charged_encoding_and_edit_costs=True,edit_costs_in_fit_completion=U.ref(a.fit_completion),decode_and_save_seconds=decode_seconds,
            original_D_resumed=False,old_EEG_prior_UNet_or_VAE_weights_updated=False,normal_inference_target_access=False,
            normal_inference_only_current_query_EEG_and_generated_image=True,exact_diffusion_score_or_posterior=False,independent_confirmation=False,SOTA_established=False,
            seconds=time.monotonic()-invoked,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes)
        U.atomic(a.output/'completion.json',complete);log(dict(stage='COMPLETE',decision=readout['decision']));return 0
    except BaseException as error:
        U.atomic(a.output/'failure.json',dict(status='D19_INCOMPLETE_NO_AUTOMATIC_RETRY',active=active,new_research_images=len(new_rows),error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-invoked));raise

if __name__=='__main__':raise SystemExit(main())
