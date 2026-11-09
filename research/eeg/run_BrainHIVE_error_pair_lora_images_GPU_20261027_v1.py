"""D20 genuine four-step generation; 80 then fixed 160-query continuation chunks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,signal,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_error_pair_lora_20261027_v2 as V
import brainhive_error_pair_lora_readout_20261027_v1 as M
import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N

def main():
    p=argparse.ArgumentParser()
    for key in ('spec','native-registration','previous','output'):p.add_argument('--'+key,type=Path)
    p.add_argument('--end',type=int,choices=(80,240,400),required=True);p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    invoked=time.monotonic();assert not a.output.exists() and (a.end==80)==(not a.previous);a.output.mkdir(parents=True)
    active='INPUTS';stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));newrows=[]
    def log(x):
        x=x|{'seconds':time.monotonic()-invoked}
        with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(x),flush=True)
    try:
        spec=U.read(a.spec);assert spec['schema']==M.SCHEMA
        for item in spec['source_pins']:U.checked(item)
        fixed=U.read(U.checked(spec['fit_spec']));assert fixed['image_policy']['primary_reference']=='ordinary'
        completed={arm:U.read(U.checked(spec['fits'][arm])) for arm in V.ARMS}
        for arm,x in completed.items():assert x['arm']==arm and x['coverage_all'] and x['updates']==200 and x['fixed_final_checkpoint'] and x['original_weights_unchanged'];U.checked(x['model'])
        assert len({x['initial_state_sha256'] for x in completed.values()})==1
        d09=U.read(U.checked(spec['D09_completion']));inputs=U.read(U.checked(d09['inputs']));panel=U.read(U.checked(inputs['panel']));queries=panel['Dcheck'];assert len(queries)==400
        conditions=np.load(U.checked(d09['check_conditions']),allow_pickle=False);assert conditions.shape==(400,3,1024)
        source_rows=[json.loads(x) for x in U.checked(d09['check_manifest']).read_text().splitlines()]
        baseline=np.load(U.checked(d09['check_generated_features']),allow_pickle=False);target=np.load(U.checked(d09['check_target_features']),allow_pickle=False)
        prefix=[];previous_features=None;start=0
        if a.previous:
            prev=U.read(a.previous);assert prev['spec']==U.ref(a.spec) and prev['end']==a.end-160
            if prev['end']==80:assert prev['readout']['decision']['action']=='EXPAND400'
            start=prev['end'];prefix=[json.loads(x) for x in U.checked(prev['manifest']).read_text().splitlines()];previous_features=np.load(U.checked(prev['generated_features']),allow_pickle=False)
            assert len(prefix)==3*start and previous_features.shape==(start,3,768)
            for row in prefix:assert U.sha(row['generated_path'])==row['generated_sha256']
        import brainhive_paired_study_contract_20261025_v1 as C
        cfg=C.read(a.native_registration);C.validate(cfg);C.setup_runtime(cfg)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=a.max_seconds,maximum_output_bytes=1<<30,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():assert not stopped[0] and time.monotonic()-invoked<a.max_seconds-35,'finite D20 chunk';budget.check()
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);C.resources(cfg,training_start=True)
        from brainhive_component_generator_v1 import load_native
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        folder=a.output/'loader';folder.mkdir();observer=ConvertedWeightAudit(torch,folder);observer.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:observer.restore()
        loader=observer.complete();assert all(x['host']==host for x in completed.values());layers=V.install(pipe.unet)
        for p in V.new_parameters(layers):p.requires_grad_(False)
        states={arm:torch.load(U.checked(x['model']),weights_only=False,map_location='cpu')['model'] for arm,x in completed.items()}
        # Preserve the old one-query FP16 A/P GEMM path for exact parity;
        # batching these400 projections can change rounding and sampled pixels.
        with torch.no_grad():tokens=torch.cat([projector(adapter(torch.from_numpy(np.array(conditions[i:i+1,1],copy=True)).cuda().half())).detach() for i in range(400)])
        for i in range(400):assert hashlib.sha256(tokens[i:i+1].cpu().contiguous().numpy().tobytes()).hexdigest()==source_rows[3*i+1]['tokens_sha256']
        V.enabled(layers,False);img,obs=N.sample_observed(pipe,prompts,tokens[:1],queries[0]['image_seed'],torch);parity=N.save_image(a.output/'old_ordinary_pixel_parity.webp',img);old=source_rows[1]
        assert old['arm']=='ordinary'
        for k in ['generated_sha256','generated_pixel_sha256']:assert parity[k]==old[k]
        U.atomic(a.output/'pixel_parity.json',dict(status='D20_DISABLED_LOW_RANK_NATIVE_OLD_ORDINARY_EXACT_PIXEL_AND_NOISE_REPLAY',image=parity,old_reference=old,observed=obs))
        active='ACTUAL_FOUR_STEP_SAMPLING_ALL_THREE_FIXED_MODELS';rows=list(prefix);manifest=a.output/'inference_manifest.jsonl'
        with manifest.open('x') as stream,ThreadPoolExecutor(max_workers=1) as executor:
            for row in prefix:stream.write(json.dumps(row)+'\n')
            for i in range(start,a.end):
                check();q=queries[i];reference=source_rows[3*i+1]
                for k,v in q.items():assert reference[k]==v
                assert hashlib.sha256(np.ascontiguousarray(conditions[i:i+1,1]).tobytes()).hexdigest()==reference['condition_sha256']
                futures=[]
                for arm in V.ARMS:
                    V.load_state(layers,states[arm]);V.enabled(layers,True)
                    image,proof=N.sample_observed(pipe,prompts,tokens[i:i+1],q['image_seed'],torch)
                    for key in ['initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations']:assert proof[key]==reference[key]
                    row=q|dict(arm=arm,query_index=i,condition_sha256=reference['condition_sha256'],tokens_sha256=hashlib.sha256(tokens[i:i+1].cpu().contiguous().numpy().tobytes()).hexdigest(),model=completed[arm]['model'],ordinary_checkpoint=fixed['ordinary_checkpoint'],normal_inference_target_access=False,normal_inference_input='current query EEG through frozen native encoder and old ordinary; new fixed native IP low-rank weights',extra_UNet_NFE=0,new_UNet_NFE=4,**proof)
                    futures.append((executor.submit(N.save_image,a.output/arm/f'{i:03d}_{q["array_row"]:05d}.webp',image),row))
                for future,row in futures:
                    row|=future.result();rows.append(row);newrows.append(row);stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                if (i+1)%10==0:log(dict(stage=active,queries=i+1,end=a.end,new_images=len(newrows)))
        assert len(rows)==3*a.end and rows[:len(prefix)]==prefix
        active='FROZEN_DINO_FEATURES_AND_FIXED_POLICY';import image_evaluation as E
        assert U.sha(E.DINO_WEIGHT)==E.DINO_SHA256==U.S.DINO_SHA256;dino=E.dino_model();paths=[Path(r['generated_path']) for r in newrows];pieces=[]
        for lo in range(0,len(paths),32):check();pieces.append(E.dino_features(paths[lo:lo+32],dino,batch_size=32,real_images=False).float().numpy())
        features=np.concatenate(pieces).reshape(-1,3,768)
        if previous_features is not None:features=np.concatenate([previous_features,features]);assert np.array_equal(features[:start],previous_features)
        U.save_array(a.output/'generated_features.npy',features)
        readout=M.score(features,target,baseline,queries[:a.end],queries,a.end) if a.end in (80,400) else None
        if readout is not None:
            U.atomic(a.output/'readout.json',readout);arrays={arm:U.S.metric_arrays(baseline[:a.end,j],target,queries[:a.end],queries) for j,arm in enumerate(U.ARMS)}
            arrays|={arm:U.S.metric_arrays(features[:,j],target,queries[:a.end],queries) for j,arm in enumerate(V.ARMS)}
            with (a.output/'per_query.csv').open('x',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['query_id','concept_id','arm']+list(U.S.METRICS));w.writeheader()
                for i,q in enumerate(queries[:a.end]):
                    for arm,values in arrays.items():w.writerow({'query_id':q['query_id'],'concept_id':q['concept_id'],'arm':arm}|{k:float(v[i]) for k,v in values.items()})
        U.atomic(a.output/'completion.json',dict(status='COMPLETE_D20_FIXED_NATIVE_LORA_IMAGES_CHUNK',schema=M.SCHEMA,spec=U.ref(a.spec),source=U.ref(__file__),wrapper=wrapper,start=start,end=a.end,manifest=U.ref(manifest),generated_features=U.ref(a.output/'generated_features.npy'),previous=U.ref(a.previous) if a.previous else None,readout=readout,
            fits=spec['fits'],D09_completion=spec['D09_completion'],pixel_parity=U.ref(a.output/'pixel_parity.json'),loader=loader,host=host,new_research_images=len(newrows),reused_research_images=len(prefix),reused_strong_reference_images=3*a.end,new_parity_images=1,
            new_UNet_NFE=4*(len(newrows)+1),per_query_inference_UNet_NFE=4,extra_sampler_steps=0,all_arms_and_queries_reported=True,original80_prefix_preserved=True,normal_inference_target_access=False,formal_D_resumed=False,
            same_NFE_not_identical_compute=True,exact_Turbo_likelihood_or_posterior=False,seconds=time.monotonic()-invoked,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=torch.cuda.max_memory_allocated(),SOTA_established=False))
        log({'stage':'COMPLETE_FIXED_CHUNK','end':a.end,'decision':readout['decision'] if readout is not None else 'CONTINUE_FIXED_REMAINDER_NO_INTERIM_SCORE_GATE'})
    except BaseException as e:
        U.atomic(a.output/'failure.json',dict(status='FAILED_D20_IMAGE_CHUNK_PREFIX_RETAINED',stage=active,error=repr(e),new_images=len(newrows),seconds=time.monotonic()-invoked,formal_D_resumed=False));traceback.print_exc();raise

if __name__=='__main__':main()
