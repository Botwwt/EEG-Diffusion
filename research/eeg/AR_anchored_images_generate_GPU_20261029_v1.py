"""Fixed AR EEG-only images, with own-final gates and immutable query receipts."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,gc,hashlib,json,shutil,subprocess,time,traceback
from pathlib import Path
import AR_anchored_images_contract_CPU_20261029_v1 as C

def peak_CPU():return int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmHWM:')))*1024
def tensor_digest(state):
    import torch
    h=hashlib.sha256()
    for key,value in state.items():
        t=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def wrapper_receipt(a,job):
    expected='run_gpu_job_v2.py' if a['host_role']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'
    assert expected in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode().replace('\0',' ')
    assert '/' not in job;path=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1/audit')/(job+'_job.json')
    for _ in range(30):
        if path.is_file():break
        time.sleep(.1)
    value=json.loads(path.read_bytes());assert value['pid']==os.getpid() and value['wrapper_pid']==os.getppid() and value['status']=='RUNNING'
    assert float(value['max_seconds'] if a['host_role']=='old' else value['maximum_seconds'])==900
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==a['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    return path

def run(admission_path,arm,stage,lo,hi,job):
    # Fail drafts and missing fixed finals before torch, outputs or CUDA.
    a,admission_sha=C.load_admission(admission_path);C.validate_sources(a,Path(__file__));C.chunk_bounds(stage,lo,hi)
    policy,panel=C.validate_panel(a);done,fit_ref=C.validate_fit(a,arm,model=True);C.validate_expansion(a,stage)
    started=time.monotonic();base_CPU=peak_CPU();root=Path(a['output_root']);root.mkdir(exist_ok=True)
    arm_root=root/arm;arm_root.mkdir(exist_ok=True)
    for name in ('images','records','generation_chunks'):(arm_root/name).mkdir(exist_ok=True)
    chunk=arm_root/'generation_chunks'/f'{lo:05d}_{hi:05d}';assert not chunk.exists();chunk.mkdir();phase='WRAPPER_AND_REAL_EEG_INPUTS'
    try:
        wrapper=wrapper_receipt(a,job)
        assert shutil.disk_usage(root).free>=a['resources']['unreserved_disk_buffer_bytes']+(hi-lo)*300000
        import numpy as np
        import torch
        from PIL import Image
        from EEG_generative_backend_loader_20261029_v1 import build_and_load
        from EEG_anchored_attention_20261029_v1 import EEGAnchoredAttentionDecoder
        data=a['EEG'];assert data['completion']['sha256']==C.EEG_SHA
        memory=json.loads(C.checked(data['completion']).read_bytes());ids=json.loads(C.checked(data['identities']).read_bytes())
        assert data['identities']['sha256']==memory['files']['identities']['sha256']
        assert data['c']['sha256']==memory['files']['Dcheck400_c']['sha256']
        c=np.load(C.checked(data['c']),mmap_mode='r',allow_pickle=False);assert c.shape==(400,1024) and c.dtype==np.float32 and np.isfinite(c).all()
        h=None
        if arm!='native_global':
            assert data['H']['sha256']==memory['files']['Dcheck400_memory']['sha256']
            h=np.load(C.checked(data['H']),mmap_mode='r',allow_pickle=False);assert h.shape==(400,63,1024) and h.dtype==np.float32 and np.isfinite(h).all()
            assert np.allclose(h.mean(1,dtype=np.float64),c,atol=2e-5,rtol=2e-5)
        assert len(ids['Dcheck'])==400
        for q,eeg in zip(panel['queries'],ids['Dcheck']):assert all(q[key]==eeg[key] for key in ('array_row','stimulus_id','concept_id'))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        phase='EXACT_FIXED_FINAL_VAR_AND_FROZEN_PRETRAINED_VQ'
        vae,decoder,backend=build_and_load(a['author_source'],a['vendor'],C.checked(a['public_VQ']),C.checked(a['public_VAR']),seed=1729)
        assert tensor_digest(decoder.state_dict())==C.INITIAL
        model_ref=dict(done['final_model'],path=a['model_paths'][arm]);state=torch.load(C.checked(model_ref),map_location='cpu',weights_only=True,mmap=True)
        assert state['arm']==arm and state['admission_sha256']==done['admission']['sha256']
        assert state['epoch']==10 and state['updates']==8270 and state['query_exposures']==132300
        assert state['EEG_bank_source_sha256']==C.EEG_SHA and state['token_bank_source_sha256']==C.TOKEN_SHA and state['initial_generator_tensor_digest']==C.INITIAL
        expected=decoder.state_dict();assert set(state['decoder'])==set(expected)
        for key,value in state['decoder'].items():
            assert value.shape==expected[key].shape and value.dtype==expected[key].dtype
            if value.is_floating_point() and 'attn_bias_for_masking' not in key:assert torch.isfinite(value).all(),key
        trained_digest=tensor_digest(state['decoder']);assert trained_digest==state['trained_decoder_tensor_digest']
        decoder.load_state_dict(state['decoder'],strict=True);del state,expected;gc.collect()
        vae.cuda().eval().requires_grad_(False);decoder.cuda().eval().requires_grad_(False);decoder.cond_drop_rate=0.;decoder.rng=torch.Generator(device='cuda')
        model=EEGAnchoredAttentionDecoder(decoder).eval().requires_grad_(False);torch.cuda.reset_peak_memory_stats();vq_digest=tensor_digest(vae.state_dict())
        binding=dict(schema='AR_ANCHORED_GENERATION_CHUNK_BINDING_v1',admission=C.ref(admission_path),arm=arm,stage=stage,range=[lo,hi],fit_completion=fit_ref,model=model_ref,trained_decoder_tensor_digest=trained_digest,
            source_pins=a['source_pins'],policy=a['policy'],panel=a['panel'],c=data['c'],H=None if h is None else data['H'],pretrained_backend=backend,generation_cost=a['generation_cost'],wrapper_job_name=job)
        C.write_once(chunk/'binding.json',binding);binding_ref=C.ref(chunk/'binding.json');records=[]
        def guard():
            assert time.monotonic()-started<a['resources']['body_hard_seconds']
            assert peak_CPU()-base_CPU<a['resources']['maximum_extra_CPU_bytes'] and torch.cuda.max_memory_allocated()<a['resources']['maximum_GPU_allocated_bytes']
            assert shutil.disk_usage(root).free>=a['resources']['unreserved_disk_buffer_bytes']
        phase='EEG_ONLY_ONE_SEED_ONE_PNG_PER_QUERY'
        for index in range(lo,hi):
            guard();row=panel['queries'][index];query=C.stripped_query(row);image_path,record_path=C.image_paths(a,arm,index)
            # An interrupted orphan is preserved and must be reviewed; no replacement.
            assert not image_path.exists() and not record_path.exists()
            global_c=torch.from_numpy(np.array(c[index:index+1],copy=True)).cuda()
            own_H=None if h is None else torch.from_numpy(np.array(h[index:index+1],copy=True)).cuda()
            began_query=time.monotonic()
            with torch.inference_mode():
                image=model.generate(global_c,eeg_memory=own_H,seed=query['image_seed'],guidance=1.5,top_k=600)
            assert image.shape==(1,3,256,256) and torch.isfinite(image).all()
            # Locked author's integer-image rule; no other image or seed is tried.
            pixels=image[0].mul(255).add(.5).clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
            pending=image_path.with_suffix('.png.pending');assert not pending.exists()
            Image.fromarray(pixels,mode='RGB').save(pending,format='PNG');pending.rename(image_path)
            record=dict(schema='AR_ANCHORED_SINGLE_FIXED_IMAGE_v1',arm=arm,query_index=index,query=query,sampling_seed=query['image_seed'],generation_binding=binding_ref,
                policy_sha256=C.POLICY_SHA,panel_sha256=C.PANEL_SHA,fit_completion=fit_ref,model_sha256=model_ref['sha256'],trained_decoder_tensor_digest=trained_digest,
                inference_inputs='own_global_c' if h is None else 'own_global_c_and_own_local_H',normal_inference_reads_query_image_or_label_or_gallery=False,training_donor_memory_used_at_inference=False,
                c_row_SHA256=hashlib.sha256(np.asarray(c[index]).tobytes()).hexdigest(),H_row_SHA256=None if h is None else hashlib.sha256(np.asarray(h[index]).tobytes()).hexdigest(),
                image=C.ref(image_path),raw_pixel_SHA256=hashlib.sha256(pixels.tobytes()).hexdigest(),raw_pixels_dtype='uint8_RGB',resolution=[256,256],best_of_N=1,sampling=policy['generation'],
                seconds=time.monotonic()-began_query,AR_generation_cost=a['generation_cost'])
            C.write_once(record_path,record);records.append(C.ref(record_path));del image,global_c,own_H;guard()
        assert tensor_digest(decoder.state_dict())==trained_digest and tensor_digest(vae.state_dict())==vq_digest
        assert all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(not p.requires_grad and p.grad is None for p in vae.parameters())
        guard();C.write_once(chunk/'completion.json',dict(status='COMPLETE_FIXED_AR_EEG_ONLY_IMAGE_CHUNK',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,stage=stage,query_range=[lo,hi],query_count=hi-lo,
            binding=binding_ref,records=records,fit_completion=fit_ref,all_trained_VAR_and_frozen_VQ_weights_unchanged=True,generated_query_images=hi-lo,best_of_N=1,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak_CPU()-base_CPU,seconds=time.monotonic()-started,
            effect_scores_read=False,query_images_read_by_generation=0,labels_or_gallery_used_as_model_input=False))
    except BaseException as error:
        C.write_once(chunk/'failure.json',dict(status='AR_FIXED_IMAGE_CHUNK_FAILURE_PRESERVED_NO_QUERY_REPLACEMENT',phase=phase,error=repr(error),traceback=traceback.format_exc(),range=[lo,hi],seconds=time.monotonic()-started,full_intended_denominator=stage));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--admission',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--stage',type=int,required=True);p.add_argument('--lo',type=int,required=True);p.add_argument('--hi',type=int,required=True);p.add_argument('--job-name',required=True);q=p.parse_args()
    try:run(q.admission,q.arm,q.stage,q.lo,q.hi,q.job_name)
    except ValueError as error:
        if str(error).startswith('NOT_ADMITTED_'):print(str(error));raise SystemExit(64)
        raise
