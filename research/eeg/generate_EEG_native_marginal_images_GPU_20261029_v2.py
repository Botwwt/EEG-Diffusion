"""New-route fixed-final EEG-only images and frozen DINO features.

No teacher image, VQ target, target features or target score is read by this
producer. The same-concept donor arm is an intervention control, not inference.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,gc,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def check(x):p=Path(x['path']);assert ref(p)==x;return p
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def seed(stimulus,kind):return int.from_bytes(hashlib.sha256(f'1729|EEG-native-marginal-v2|{kind}|sub-01|{stimulus}|R4'.encode()).digest()[:8],'little')%(2**63-1)
def digest(state):
    import torch
    h=hashlib.sha256()
    for key,value in state.items():
        t=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def run(sp,lo,hi,job):
    s=json.loads(Path(sp).read_bytes());assert s['status']=='REGISTERED_NATIVE_MARGINAL_FIXED_FINAL_IMAGES_v2'
    assert 0<=lo<hi<=400 and lo%20==0 and hi==lo+20
    if lo>=80:
        expansion=json.loads(check(s['stage80_decision']).read_bytes());assert expansion['decision']=='EXPAND400'
    for item in s['source_pins']:check(item)
    # Fail absent fixed finals/CPU audit before CUDA or output creation.
    done=json.loads(check(s['fit_completion']).read_bytes());audit=json.loads(check(s['fit_CPU_audit']).read_bytes())
    assert done['status']=='FULL_T_NATIVE_MARGINAL_FIXED_FINAL_FIT_COMPLETE' and done['epoch']==10 and done['EEG_updates']==2070
    assert audit['status']=='PASS_FULL_T_FIXED_FINAL_EEG_POSTERIOR_OPTIMIZER_COVERAGE_AND_RNG_CPU' and audit['final_model']==done['final_model']
    fit_spec=json.loads(check(done['spec']).read_bytes());assert fit_spec['arm']==s['fitted_arm'] and fit_spec['recipe']['tokens']==5
    root=Path(s['output_root']);root.mkdir(exist_ok=True);out=root/f'chunk_{lo:04d}_{hi:04d}';out.mkdir();started=time.monotonic();base=peak();phase='WRAPPER_AND_RAW_EEG'
    try:
        expected='run_gpu_job_v2.py' if s['host']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'
        assert expected in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode().replace('\0',' ')
        receipt=Path(s['repository_root'])/'audit'/(job+'_job.json')
        for _ in range(30):
            if receipt.is_file():break
            time.sleep(.1)
        rr=json.loads(receipt.read_bytes());assert rr['pid']==os.getpid() and rr['status']=='RUNNING'
        assert float(rr['max_seconds'] if s['host']=='old' else rr['maximum_seconds'])==900
        assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        sys.path.insert(0,fit_spec['training_code']);import AR_anchored_training_contract_CPU_20261029_v1 as C
        a,_=C.load_admission(fit_spec['training_admission']);C.validate_sources(a);C.verify_protected_files(a);sys.path.insert(0,a['vendor'])
        import numpy as np
        import torch
        from PIL import Image
        from EEG_generative_backend_loader_20261029_v1 import build_and_load
        from eeg_labram_pretrained_20261028_v1 import build_encoder
        from eeg_labram_interface_20261028_v1 import CacheNativeEEGTransform
        from EEG_native_conditional_variational_core_20261029_v1 import NativeConditionDecoder,EEGGaussianPrior,native_moments,reparameterize
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        panel=json.loads(check(s['panel']).read_bytes());queries=panel['queries'];assert len(queries)==400
        physical=json.loads(check(s['physical_queries']).read_bytes());assert len(physical['Dcheck'])==400
        for p,q in zip(queries,physical['Dcheck']):assert all(p[k]==q[k] for k in ('stimulus_id','array_row','concept_id'))
        raw=np.load(check(s['Dcheck_mean63']),mmap_mode='r',allow_pickle=False);assert raw.shape==(400,63,250)
        stats=json.loads(check(fit_spec['full_EEG']['normalization']).read_bytes());assert stats['training_positions']==list(range(13230))
        transform=CacheNativeEEGTransform(tuple(stats['channel_names']),np.array(stats['center']),np.array(stats['scale']))
        x=transform(np.array(raw),channel_names=tuple(stats['channel_names']))
        groups={}
        for i,q in enumerate(queries):groups.setdefault(q['concept_id'],[]).append(i)
        assert len(groups)==40 and {len(g) for g in groups.values()}=={10}
        donor={i:g[(j+1)%10] for g in groups.values() for j,i in enumerate(g)}
        arms=['coupled','donor'] if s['host']=='old' else ['visual','marginal']
        assert s['arms']==arms
        def guard():
            assert time.monotonic()-started<800 and peak()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
            assert shutil.disk_usage(out).free>2<<30
        phase='PUBLIC_NATIVE_PRIOR_AND_FIXED_FINAL_EEG'
        torch.cuda.reset_peak_memory_stats();vae,d,backend=build_and_load(a['author_source'],a['vendor'],check(a['public_VQ']),check(a['public_VAR']),seed=1729)
        public=torch.load(check(a['public_VAR']),weights_only=True,map_location='cpu',mmap=True);classes=public['class_emb.weight'].clone();del public
        before=digest({k:v for k,v in d.state_dict().items() if not k.startswith('embed_proj.')});vq_before=digest(vae.state_dict())
        native=NativeConditionDecoder(d,classes).cuda().eval();vae.cuda().eval().requires_grad_(False);d.rng=torch.Generator(device='cuda')
        center,scale=native_moments(classes);encoder=build_encoder(a['author_source'],a['vendor'],seed=1729);p=EEGGaussianPrior(encoder,center,scale)
        final=torch.load(check(done['final_model']),weights_only=True,map_location='cpu');assert final['epoch']==10 and final['updates']==2070 and final['arm']==s['fitted_arm'] and final['spec_sha256']==done['spec']['sha256']
        p.load_state_dict(final['model'],strict=True);del final;p.cuda().eval().requires_grad_(False)
        from models.labram import get_input_chans
        positions=get_input_chans([name.upper() for name in stats['channel_names']])
        with torch.inference_mode():
            predicted=[p(torch.from_numpy(x[b:b+64]).cuda(),input_chans=positions) for b in range(0,400,64)]
            means=torch.cat([r[0] for r in predicted]);logs=torch.cat([r[1] for r in predicted]);del predicted
        assert means.shape==logs.shape==(400,1024) and torch.isfinite(means).all() and torch.isfinite(logs).all()
        np.save(out/'EEG_means400.npy',means.cpu().numpy(),allow_pickle=False);np.save(out/'EEG_logscales400.npy',logs.cpu().numpy(),allow_pickle=False)
        all_records={};all_paths={};phase='ONE_GAUSSIAN_ONE_IMAGE_NO_TARGET_INPUT'
        for arm in arms:
            directory=out/arm;directory.mkdir();records=[];paths=[]
            for index in range(lo,hi):
                guard();query=queries[index];latent_seed=seed(query['stimulus_id'],'latent');token_seed=seed(query['stimulus_id'],'token');source=donor[index] if arm=='donor' else index
                qm,qs=(center.cuda()[None],scale.cuda().log()[None]) if arm=='marginal' else (means[source:source+1],logs[source:source+1])
                rng=torch.Generator(device='cuda').manual_seed(latent_seed)
                with torch.inference_mode():
                    z=reparameterize(qm,qs,generator=rng);image=native.generate(z,seed=token_seed,guidance=0.,top_k=600)
                assert image.shape==(1,3,256,256) and torch.isfinite(image).all()
                pixels=image[0].mul(255).add(.5).clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
                if index==0:
                    # Same-seed replay is engineering verification, never a
                    # second research candidate or a best-of-N selection.
                    with torch.inference_mode():replay=native.generate(z,seed=token_seed,guidance=0.,top_k=600)
                    pix2=replay[0].mul(255).add(.5).clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy();assert np.array_equal(pixels,pix2);del replay,pix2
                image_path=directory/f'{index:04d}.png';Image.fromarray(pixels,mode='RGB').save(image_path)
                np.save(directory/f'{index:04d}_latent.npy',z.cpu().numpy(),allow_pickle=False)
                record=dict(arm=arm,index=index,query_id=query['query_id'],stimulus_id=query['stimulus_id'],array_row=query['array_row'],EEG_row=None if arm=='marginal' else source,donor_intervention=(arm=='donor'),latent_seed=latent_seed,token_seed=token_seed,
                    image=ref(image_path),latent=ref(directory/f'{index:04d}_latent.npy'),pixel_SHA256=hashlib.sha256(pixels.tobytes()).hexdigest(),EEG_parameters_sha256=hashlib.sha256(qm.cpu().numpy().tobytes()+qs.cpu().numpy().tobytes()).hexdigest(),
                    generation=dict(one_Gaussian_sample=True,one_image=True,best_of_N=1,guidance=0.,top_k=600,top_p=0.,resolution=[256,256],scales=[1,2,3,4,5,6,8,10,13,16]),normal_inference_reads_query_image_or_label_or_gallery=False,fit_model_sha256=done['final_model']['sha256'])
                records.append(record);paths.append(image_path);del image,z
            write(directory/'records.json',records);all_records[arm]=records;all_paths[arm]=paths
        assert digest(d.state_dict())==before and digest(vae.state_dict())==vq_before
        del native,p,vae,d,means,logs;gc.collect();torch.cuda.empty_cache();guard();phase='FROZEN_DINO_FEATURES_AND_INDEPENDENT_PIXELS'
        import image_evaluation as E
        assert sha(Path(E.__file__))==s['image_evaluation']['sha256'] and E.DINO_SHA256==s['DINO_weight']['sha256'];check(s['DINO_weight'])
        model=E.dino_model();feature_refs={}
        for arm in arms:
            # Complete image identities/decode/pixel SHA before feature reads.
            for record in all_records[arm]:
                with Image.open(check(record['image'])) as im:pix=np.asarray(im.convert('RGB'));assert im.size==(256,256)
                assert hashlib.sha256(pix.tobytes()).hexdigest()==record['pixel_SHA256']
            features=E.dino_features(all_paths[arm],model,batch_size=20,real_images=False).float().cpu().numpy()
            assert features.shape==(20,768) and np.isfinite(features).all();path=out/arm/'features.npy';np.save(path,features,allow_pickle=False);feature_refs[arm]=ref(path)
        C.verify_protected_files(a);guard()
        write(out/'completion.json',dict(status='COMPLETE_NATIVE_MARGINAL_FIXED_FINAL_TWO_ARM_IMAGE_FEATURE_CHUNK',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(sp),range=[lo,hi],arms=arms,fit_completion=s['fit_completion'],fit_CPU_audit=s['fit_CPU_audit'],EEG_means=ref(out/'EEG_means400.npy'),EEG_logscales=ref(out/'EEG_logscales400.npy'),
            records={arm:ref(out/arm/'records.json') for arm in arms},features=feature_refs,DINO_weight=s['DINO_weight'],image_evaluation=s['image_evaluation'],native_prior_and_VQ_unchanged=True,query_target_images_read=0,effect_scores_read=False,generated_research_images=40,same_seed_engineering_replays=2 if lo==0 else 0,
            peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base,seconds=time.monotonic()-started))
    except BaseException as error:
        write(out/'failure.json',dict(status='NATIVE_MARGINAL_IMAGE_FAILURE_PRESERVED_NO_QUERY_REPLACEMENT',phase=phase,error=repr(error),traceback=traceback.format_exc(),range=[lo,hi],seconds=time.monotonic()-started));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--job-name',required=True);ap.add_argument('--lo',required=True,type=int);ap.add_argument('--hi',required=True,type=int);v=ap.parse_args();run(v.spec,v.lo,v.hi,v.job_name)
