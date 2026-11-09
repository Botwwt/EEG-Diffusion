"""Fixed-final actual-prefix diagnostic; no candidate or sampling-rule change.

All200 own-EEG generations are preserved. Hooks only observe code indices and
the first2 raw-logit trajectories; neither hook reads targets or changes output.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,gc,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path

NAMES=['zero_EEG_same_concept_donor_GT_prefix','own_EEG_own_sampled_prefix','same_concept_donor_EEG_own_sampled_prefix','zero_EEG_own_sampled_prefix',
       'CFG_own_EEG_own_sampled_prefix_untruncated','own_EEG_own_GT_prefix_replay','zero_EEG_own_GT_prefix_replay','CFG_own_EEG_own_GT_prefix_untruncated']
SIDES=(1,2,3,4,5,6,8,10,13,16)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def checked(x):p=Path(x['path']);assert ref(p)==x;return p
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def digest(state):
    import torch
    h=hashlib.sha256()
    for key,value in state.items():
        t=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def seed_for(stimulus):return int.from_bytes(hashlib.sha256(f'1729|AR-Dfit-actual-prefix|sub-01|{stimulus}|R4'.encode()).digest()[:8],'little')%(2**63-1)

def run(sp,job):
    s=json.loads(Path(sp).read_bytes());assert s['status']=='REGISTERED_FIXED_NATIVE_ACTUAL_PREFIX_DFIT200_DIAGNOSTIC_v1' and s['conditions']==NAMES
    for item in s['source_pins']:checked(item)
    sys.path.insert(0,s['training_code']);import AR_anchored_training_contract_CPU_20261029_v1 as C
    a,admission_sha=C.load_admission(s['training_admission'],arm='native_global');C.validate_sources(a);C.verify_protected_files(a)
    prior=json.loads(checked(s['closed_teacher_diagnostic']).read_bytes());audit=json.loads(checked(s['closed_teacher_CPU_audit']).read_bytes())
    assert prior['status']=='COMPLETE_FIXED_NATIVE_FINAL_DFIT200_ALL_FIVE_CONTEXT_CONDITIONS_AND10SCALES' and audit['producer_completion']==s['closed_teacher_diagnostic']
    assert audit['status']=='PASS_INDEPENDENT_ALL200_5CONDITIONS_680TOKENS_10SCALES_AND1000_BOOTSTRAPS_CPU'
    fit=json.loads(checked(prior['fit_completion']).read_bytes());assert fit['final_model']==s['fixed_final'] and fit['updates']==8270 and fit['epochs']==10 and fit['query_exposures']==132300
    query_data=json.loads(checked(s['queries']).read_bytes());rows=query_data['rows'];donors=query_data['same_concept_donor'];assert len(rows)==len(donors)==200
    assert len(set(r['concept_id'] for r in rows))==20 and all(i!=j and rows[i]['concept_id']==rows[j]['concept_id'] and rows[i]['partition']=='Dfit' for i,j in enumerate(donors))
    out=Path(s['output_root']);assert not out.exists();out.mkdir();started=time.monotonic();base=peak();phase='ACTUAL_WRAPPER_AND_CACHED_INPUTS'
    try:
        assert 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode().replace('\0',' ')
        receipt=Path(s['repository_root'])/'audit'/(job+'_job.json')
        for _ in range(30):
            if receipt.exists():break
            time.sleep(.1)
        receipt_value=json.loads(receipt.read_bytes());assert receipt_value['pid']==os.getpid() and receipt_value['status']=='RUNNING' and receipt_value['maximum_seconds']==900
        assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-62a1aee8-4740-dc59-93f4-d50644db650d'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from EEG_generative_backend_loader_20261029_v1 import build_and_load
        from EEG_anchored_attention_20261029_v1 import EEGAnchoredAttentionDecoder
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
        c=np.load(checked(s['c']),allow_pickle=False);targets=np.load(checked(s['VQ_targets']),allow_pickle=False);old_nll=np.load(checked(prior['all_token_NLL']),allow_pickle=False)
        assert c.shape==(200,1024) and c.dtype==np.float32 and np.isfinite(c).all() and targets.shape==(200,680) and targets.dtype==np.int16
        assert old_nll.shape==(200,5,680) and targets.min()>=0 and targets.max()<4096
        def guard():
            assert time.monotonic()-started<800 and peak()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
            assert shutil.disk_usage(out).free>=2<<30
        torch.cuda.reset_peak_memory_stats();phase='UNCHANGED_FIXED_NATIVE_FINAL_AND_FROZEN_VQ'
        vae,decoder,backend=build_and_load(a['author_source'],a['vendor'],checked(a['public_VQ']),checked(a['public_VAR']),seed=1729)
        assert digest(decoder.state_dict())==a['initial_generator_tensor_digest'];final=torch.load(checked(s['fixed_final']),map_location='cpu',weights_only=True,mmap=True)
        assert final['arm']=='native_global' and final['epoch']==10 and final['updates']==8270 and final['admission_sha256']==admission_sha
        decoder.load_state_dict(final['decoder'],strict=True);trained_digest=digest(decoder.state_dict());assert trained_digest==final['trained_decoder_tensor_digest'];del final;gc.collect()
        decoder.cuda().eval().requires_grad_(False);vae.cuda().eval().requires_grad_(False);decoder.cond_drop_rate=0.;decoder.rng=torch.Generator(device='cuda');model=EEGAnchoredAttentionDecoder(decoder).eval().requires_grad_(False)
        VQ_digest=digest(vae.state_dict());(out/'images').mkdir();(out/'records').mkdir();(out/'engineering_replays').mkdir();(out/'first2_logits').mkdir()
        sampled=[];captured_first2=[];records=[];widths=[x*x for x in SIDES]
        def pixels(image):return image[0].mul(255).add(.5).clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
        phase='ALL200_OWN_EEG_ONLY_SAMPLED_PREFIXES_AND_PRESERVED_IMAGES'
        for i,row in enumerate(rows):
            guard();own=torch.from_numpy(c[i:i+1]).cuda();codes=[];logits=[]
            # Hooks return None: observe only, no replacements or target reads.
            def code_hook(module,args):codes.append(args[0].detach().cpu().clone())
            def head_hook(module,args,value):logits.append(value.detach().cpu().clone())
            handle=vae.quantize.embedding.register_forward_pre_hook(code_hook);lh=decoder.head.register_forward_hook(head_hook) if i<2 else None
            try:
                with torch.inference_mode():image=model.generate(own,eeg_memory=None,seed=seed_for(row['stimulus_id']),guidance=1.5,top_k=600)
            finally:
                handle.remove()
                if lh is not None:lh.remove()
            assert len(codes)==10 and all(tuple(x.shape)==(1,w) for x,w in zip(codes,widths));code=torch.cat(codes,1).numpy().astype(np.int16)
            assert code.shape==(1,680) and code.min()>=0 and code.max()<4096;sampled.append(code)
            assert image.shape==(1,3,256,256) and torch.isfinite(image).all();array=pixels(image);p=out/'images'/f'query_{i:03d}.png';Image.fromarray(array,mode='RGB').save(p,format='PNG')
            with Image.open(p) as im:assert np.array_equal(np.array(im.convert('RGB')),array)
            if i<2:
                assert len(logits)==10 and all(tuple(x.shape)==(2,w,4096) for x,w in zip(logits,widths));raw=torch.cat(logits,1).numpy();np.save(out/'first2_logits'/f'query_{i:03d}_conditional_and_zero.npy',raw,allow_pickle=False);captured_first2.append(raw)
                with torch.inference_mode():replay=model.generate(own,eeg_memory=None,seed=seed_for(row['stimulus_id']),guidance=1.5,top_k=600)
                assert np.array_equal(pixels(replay),array),'Observation hooks changed the actual sampling path';Image.fromarray(pixels(replay),mode='RGB').save(out/'engineering_replays'/f'query_{i:03d}_same_seed.png',format='PNG');del replay
            record=dict(query_index=i,stimulus_id=row['stimulus_id'],array_row=row['array_row'],seed=seed_for(row['stimulus_id']),own_EEG_c_sha256=hashlib.sha256(c[i].tobytes()).hexdigest(),sampled_codes_sha256=hashlib.sha256(code.tobytes()).hexdigest(),image=ref(p),pixel_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                inference_inputs='Own frozen EEG c and a single fixed categorical sampling seed only; hooks observe outputs and never read targets, concept labels, donor EEG or target images',CFG=1.5,top_k=600,top_p=0.,native_scale_steps=10,best_of_N=1)
            rp=out/'records'/f'query_{i:03d}.json';write(rp,record);records.append(ref(rp));del own,image;guard()
        sampled=np.concatenate(sampled);np.save(out/'all200_sampled_VQ_indices.npy',sampled,allow_pickle=False)
        phase='ALL200_MISSING_FACTORIAL_CELL_AND_ACTUAL_PREFIX_CONDITIONS'
        nll=np.empty((200,len(NAMES),680),np.float32);support=np.empty((200,2,680),np.bool_);replay_error=0.;cached_logits_error=0.;cached_NLL_error=0.
        # Match the sampler's Python scalar rounding separately for a and1+a.
        amount=torch.cat([torch.full((w,),1.5*k/9,device='cuda') for k,w in enumerate(widths)]).view(1,680,1)
        amount_plus=torch.cat([torch.full((w,),1+1.5*k/9,device='cuda') for k,w in enumerate(widths)]).view(1,680,1)
        with torch.inference_mode():
            for lo in range(0,200,8):
                guard();hi=min(lo+8,200);own=torch.from_numpy(c[lo:hi]).cuda();other=torch.from_numpy(c[np.array(donors[lo:hi])]).cuda();zero=torch.zeros_like(own)
                target=torch.tensor(targets[lo:hi],device='cuda',dtype=torch.long)
                def prefix(values):return vae.quantize.idxBl_to_var_input(list(torch.split(torch.tensor(values,device='cuda',dtype=torch.long),widths,dim=1)))
                actual=prefix(sampled[lo:hi]);correct=prefix(targets[lo:hi]);wrong=prefix(targets[np.array(donors[lo:hi])])
                def loss(value):return F.cross_entropy(value.float().transpose(1,2),target,reduction='none').cpu().numpy()
                nll[lo:hi,0]=loss(model(wrong,zero,eeg_memory=None))
                own_actual=model(actual,own,eeg_memory=None);zero_actual=model(actual,zero,eeg_memory=None)
                nll[lo:hi,1]=loss(own_actual);nll[lo:hi,2]=loss(model(actual,other,eeg_memory=None));nll[lo:hi,3]=loss(zero_actual)
                guided=amount_plus*own_actual-amount*zero_actual;nll[lo:hi,4]=loss(guided);cutoff=guided.topk(600,dim=-1).values[...,-1:];support[lo:hi,0]=(guided.gather(-1,target[:,:,None])>=cutoff).squeeze(-1).cpu().numpy()
                if lo==0:
                    for i in range(2):
                        observed=np.stack([captured_first2[i][0],captured_first2[i][1]])
                        full=torch.stack([own_actual[i],zero_actual[i]]).cpu().numpy();error=float(np.max(abs(observed-full)));cached_logits_error=max(cached_logits_error,error)
                        assert np.allclose(observed,full,atol=1e-3,rtol=1e-4)
                        observed_loss=F.cross_entropy(torch.from_numpy(observed).transpose(1,2),torch.from_numpy(targets[i].astype(np.int64)).expand(2,-1),reduction='none').numpy()
                        error=float(np.max(abs(observed_loss-nll[i,[1,3]])));cached_NLL_error=max(cached_NLL_error,error);assert error<2e-3
                del own_actual,zero_actual,guided,cutoff
                own_GT=model(correct,own,eeg_memory=None);zero_GT=model(correct,zero,eeg_memory=None);nll[lo:hi,5]=loss(own_GT);nll[lo:hi,6]=loss(zero_GT)
                guided=amount_plus*own_GT-amount*zero_GT;nll[lo:hi,7]=loss(guided);cutoff=guided.topk(600,dim=-1).values[...,-1:];support[lo:hi,1]=(guided.gather(-1,target[:,:,None])>=cutoff).squeeze(-1).cpu().numpy()
                replay_error=max(replay_error,float(np.max(abs(nll[lo:hi,5]-old_nll[lo:hi,0]))),float(np.max(abs(nll[lo:hi,6]-old_nll[lo:hi,2]))));assert replay_error<=2e-5
                del own_GT,zero_GT,guided,cutoff;guard()
        assert np.isfinite(nll).all() and np.max(abs(nll[:,1,0]-nll[:,5,0]))<=2e-5 and np.max(abs(nll[:,3,0]-nll[:,6,0]))<=2e-5
        np.save(out/'all200_8conditions_token_NLL.npy',nll,allow_pickle=False);per_scale=np.stack([x.mean(-1,dtype=np.float64) for x in np.split(nll,np.cumsum(widths)[:-1],axis=-1)],axis=-1)
        np.save(out/'all200_8conditions_scale_NLL.npy',per_scale,allow_pickle=False);np.save(out/'actual_sampler_target_top600_support.npy',support,allow_pickle=False)
        assert digest(decoder.state_dict())==trained_digest and digest(vae.state_dict())==VQ_digest and all(not p.requires_grad and p.grad is None for p in model.parameters())
        C.verify_protected_files(a);guard();all_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());assert all_bytes<256<<20
        result=dict(status='COMPLETE_FIXED_NATIVE_DFIT200_ACTUAL_PREFIX_8CONDITIONS_AND_ALL_PRESERVED_GENERATIONS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(sp),fixed_final=s['fixed_final'],prior_diagnostic=s['closed_teacher_diagnostic'],queries=200,concepts=20,conditions=NAMES,scales=10,
            all_token_NLL=ref(out/'all200_8conditions_token_NLL.npy'),all_scale_NLL=ref(out/'all200_8conditions_scale_NLL.npy'),target_top600_support=ref(out/'actual_sampler_target_top600_support.npy'),sampled_codes=ref(out/'all200_sampled_VQ_indices.npy'),
            records=records,images=200,engineering_same_seed_replays=2,total_generate_calls=202,first2_cached_vs_full_logits_max_error=cached_logits_error,first2_cached_vs_full_NLL_max_error=cached_NLL_error,previous_own_and_zero_GT_token_NLL_replay_max_error=replay_error,
            all_VAR_VQ_weights_unchanged=True,new_optimizer_updates=0,main_training_or_Dcheck_image_gate_changed=False,original_Dfit_5condition_result_unchanged=True,
            interpretation='Exploratory extension registered after the teacher-prefix result. Untruncated CFG NLL is not exact top-k sampler NLL; target-support exclusions are reported separately. One native baseline is generated, not a new-method image benefit or independent confirmation.',
            peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base,seconds=time.monotonic()-started,output_bytes_before_completion=all_bytes)
        write(out/'completion.json',result)
    except BaseException as error:
        write(out/'failure.json',dict(status='ACTUAL_PREFIX_DFIT_DIAGNOSTIC_FAILURE_PRESERVED_NO_QUERY_DROPS_OR_AUTOMATIC_RETRY',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,full_query_denominator=200,new_optimizer_updates=0));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--job-name',required=True);q=p.parse_args();run(q.spec,q.job_name)
