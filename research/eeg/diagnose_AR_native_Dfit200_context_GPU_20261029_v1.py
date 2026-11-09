"""One fixed-final native AR diagnostic on all registered Dfit200 conditions."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,gc,hashlib,io,json,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
import AR_Dfit_context_diagnostic_math_CPU_20261029_v1 as Math

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def checked(item):
    p=Path(item['path']);assert p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];return p
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def tensor_digest(state):
    import torch
    h=hashlib.sha256()
    for key,value in state.items():
        t=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def run(sp,job):
    spec=json.loads(Path(sp).read_bytes());assert spec['status']=='REGISTERED_FIXED_FINAL_NATIVE_DFIT200_CONTEXT_DIAGNOSTIC_v1'
    for item in spec['source_pins']:checked(item)
    sys.path[:0]=[spec['training_code'],spec['EEG_code']]
    import AR_anchored_training_contract_CPU_20261029_v1 as C
    a,admission_sha=C.load_admission(spec['training_admission'],arm='native_global');assert a['host_role']=='new'
    C.validate_sources(a);C.verify_protected_files(a)
    done=json.loads(Path(spec['fixed_final_completion_path']).read_bytes())
    assert done['status']=='COMPLETE_FIXED_T_10_EPOCH_ANCHORED_AR_ARM' and done['arm']=='native_global' and done['admission']['sha256']==admission_sha
    assert done['epochs']==10 and done['updates']==8270 and done['query_exposures']==132300 and done['new_research_images']==0
    assert json.loads(Path(spec['native_training_queue_completion']).read_bytes())['status']=='ALL_HOST_ASSIGNED_FIXED_T_AR_ARMS_COMPLETE'
    artifacts=json.loads(Path(spec['native_stage80_artifact']).read_bytes());assert artifacts['status']=='COMPLETE_FIXED_AR_ALL_QUERY_ARTIFACT_AUDIT' and artifacts['valid_queries']==80
    for lo in (0,20,40,60):assert json.loads((Path(spec['native_feature_root'])/f'{lo:05d}_{lo+20:05d}/completion.json').read_bytes())['status']=='COMPLETE_FIXED_AR_FROZEN_DINO_FEATURE_CHUNK'
    binding=json.loads(checked(spec['Dfit_input_completion']).read_bytes());assert binding['status']=='COMPLETE_DFIT200_COUNTERFACTUAL_CONTEXT_INPUTS_PHYSICAL_R4_AND_ORIGINAL_BYTES_CPU_VERIFIED' and binding['Dfit_likelihoods_read'] is False
    files=binding['files'];protocol=json.loads(checked(files['protocol.json']).read_bytes());query_data=json.loads(checked(files['queries.json']).read_bytes());rows=query_data['rows'];donors=query_data['same_concept_donor']
    assert len(rows)==200 and len(set(r['concept_id'] for r in rows))==20 and protocol['conditions']==list(Math.CONDITIONS)
    assert all(j!=i and rows[i]['concept_id']==rows[j]['concept_id'] and rows[i]['partition']=='Dfit' for i,j in enumerate(donors))
    out=Path(spec['output_root']);assert not out.exists();out.mkdir();started=time.monotonic();base=peak();phase='ACTUAL_GPU_WRAPPER'
    try:
        parent=(Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode().replace('\0',' ');assert 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py' in parent
        receipt=Path(spec['repository_root'])/'audit'/(job+'_job.json')
        for _ in range(30):
            if receipt.exists():break
            time.sleep(.1)
        r=json.loads(receipt.read_bytes());assert r['pid']==os.getpid() and r['wrapper_pid']==os.getppid() and r['status']=='RUNNING' and r['maximum_seconds']==900
        assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-62a1aee8-4740-dc59-93f4-d50644db650d'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        from eeg_labram_pretrained_20261028_v1 import build_encoder
        from eeg_labram_interface_20261028_v1 import CacheNativeEEGTransform
        from EEG_generative_backend_loader_20261029_v1 import build_and_load
        from EEG_anchored_attention_20261029_v1 import EEGAnchoredAttentionDecoder
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
        def guard():
            assert time.monotonic()-started<800 and peak()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
            assert shutil.disk_usage(out).free>=2<<30
        torch.cuda.reset_peak_memory_stats();phase='FIXED_FULL_T_EEG_ENCODER_AND_ROLE_ONLY_NORMALIZATION'
        role=spec['full_EEG'];edone=json.loads(checked(role['completion']).read_bytes());assert edone['status']=='COMPLETE_SUB01_FIXED_50_EPOCH_T_ENCODER' and edone['role']=='full' and edone['epochs']==50
        estate=torch.load(checked(role['model']),map_location='cpu',weights_only=True);assert estate['role']=='full' and estate['epoch']==50 and estate['spec_sha256']=='aa0964d7e94b936459c01005e33ae19b7179c071c287f846d894406e5bd90c56'
        encoder=build_encoder(spec['author_source'],spec['vendor'],seed=1729);assert len(estate['model'])==len(encoder.state_dict())==224;encoder.load_state_dict(estate['model'],strict=True)
        before_EEG=tensor_digest(encoder.state_dict());encoder.cuda().eval().requires_grad_(False)
        stats=json.loads(checked(role['normalization']).read_bytes());assert stats['role']=='full' and stats['training_positions']==list(range(13230))
        metadata=json.loads(checked(files['EEG_metadata.json']).read_bytes());assert stats['channel_names']==metadata['channel_names']
        transform=CacheNativeEEGTransform(tuple(stats['channel_names']),np.array(stats['center']),np.array(stats['scale']));means=np.load(checked(files['Dfit200_mean63.npy']),allow_pickle=False);assert means.shape==(200,63,250) and means.dtype==np.float32
        from models.labram import get_input_chans
        channels=get_input_chans(stats['channel_names']);conditions=[]
        with torch.inference_mode():
            for lo in range(0,200,64):
                guard();x=torch.from_numpy(transform(means[lo:lo+64],channel_names=tuple(stats['channel_names']))).cuda();c,image_loss,text_loss=encoder(x,input_chans=channels)
                assert image_loss is None and text_loss is None and c.shape==(len(x),1024) and torch.isfinite(c).all();conditions.append(c.cpu());del x,c
        c_bank=torch.cat(conditions).numpy();np.save(out/'Dfit200_c.npy',c_bank,allow_pickle=False);assert tensor_digest(encoder.state_dict())==before_EEG
        del encoder,estate,conditions,means;gc.collect();torch.cuda.empty_cache();guard()
        phase='FROZEN_STRICT_VQ_AND_NATIVE_FIXED_FINAL_VAR'
        vae,decoder,backend=build_and_load(spec['author_source'],spec['vendor'],checked(a['public_VQ']),checked(a['public_VAR']),seed=1729)
        assert tensor_digest(decoder.state_dict())==a['initial_generator_tensor_digest'];final_path=checked(done['final_model']);final=torch.load(final_path,map_location='cpu',weights_only=True,mmap=True)
        assert final['arm']=='native_global' and final['epoch']==10 and final['updates']==8270 and final['admission_sha256']==admission_sha
        assert final['EEG_bank_source_sha256']=='c11bd68227feffe5efaebeac309e99090c5ab08cc8c44b87b7abe3882d4c2607' and final['token_bank_source_sha256']=='e88a6ee59d4dba2d675d4760ea4e24a25214ecd3daa42624e6043437fd3b3205'
        decoder.load_state_dict(final['decoder'],strict=True);trained_digest=tensor_digest(decoder.state_dict());assert trained_digest==final['trained_decoder_tensor_digest'];del final;gc.collect()
        vae.cuda().eval().requires_grad_(False);decoder.cuda().eval().requires_grad_(False);decoder.cond_drop_rate=0.;decoder.rng=torch.Generator(device='cuda');model=EEGAnchoredAttentionDecoder(decoder).eval().requires_grad_(False)
        VQ_digest=tensor_digest(vae.state_dict());image_transform=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BILINEAR),transforms.ToTensor()]);codes=[]
        phase='ALL200_ORIGINAL_DFIT_IMAGES_FROZEN_VQ_TARGETS'
        with zipfile.ZipFile(checked(files['original_Dfit200_images.zip'])) as archive,torch.inference_mode():
            for lo in range(0,200,8):
                guard();images=[]
                for row in rows[lo:lo+8]:
                    b=archive.read(row['image_member']);assert len(b)==row['image_bytes'] and hashlib.sha256(b).hexdigest()==row['image_sha256']
                    with Image.open(io.BytesIO(b)) as im:assert im.width==im.height;images.append(image_transform(im.convert('RGB')).mul(2).sub(1))
                indices=vae.img_to_idxBl(torch.stack(images).cuda());code=torch.cat(indices,1);assert code.shape==(len(images),680) and code.min()>=0 and code.max()<4096;codes.append(code.cpu())
        code_bank=torch.cat(codes).numpy().astype(np.int16);np.save(out/'Dfit200_VQ_indices.npy',code_bank,allow_pickle=False);del codes;guard()
        phase='ALL_FIVE_FROZEN_TEACHER_PREFIX_CONDITIONS'
        widths=[s*s for s in (1,2,3,4,5,6,8,10,13,16)];nll=np.empty((200,5,680),np.float32)
        with torch.inference_mode():
            for lo in range(0,200,8):
                guard();hi=min(lo+8,200);own=torch.from_numpy(c_bank[lo:hi]).cuda();other=torch.from_numpy(c_bank[np.array(donors[lo:hi])]).cuda()
                target=torch.tensor(code_bank[lo:hi],device='cuda',dtype=torch.long);donor_target=torch.tensor(code_bank[np.array(donors[lo:hi])],device='cuda',dtype=torch.long)
                own_prefix=vae.quantize.idxBl_to_var_input(list(torch.split(target,widths,dim=1)));donor_prefix=vae.quantize.idxBl_to_var_input(list(torch.split(donor_target,widths,dim=1)))
                for condition,(eeg,prefix) in enumerate(((own,own_prefix),(other,own_prefix),(torch.zeros_like(own),own_prefix),(own,donor_prefix),(other,donor_prefix))):
                    guard();logits=model(prefix,eeg,eeg_memory=None);loss=F.cross_entropy(logits.float().transpose(1,2),target,reduction='none');assert loss.shape==(hi-lo,680) and torch.isfinite(loss).all();nll[lo:hi,condition]=loss.cpu().numpy();del logits,loss
        first_error=float(max(np.max(np.abs(nll[:,0,0]-nll[:,3,0])),np.max(np.abs(nll[:,1,0]-nll[:,4,0]))));assert first_error<=2e-5
        np.save(out/'all_condition_token_NLL.npy',nll,allow_pickle=False);per_scale=np.stack([x.mean(-1,dtype=np.float64) for x in np.split(nll,np.cumsum(widths)[:-1],axis=-1)],axis=-1);assert per_scale.shape==(200,5,10)
        np.save(out/'all_condition_scale_NLL.npy',per_scale,allow_pickle=False);summary=Math.summarize(per_scale,[r['concept_id'] for r in rows]);write(out/'teacher_prefix_diagnostic_summary.json',summary)
        assert tensor_digest(decoder.state_dict())==trained_digest and tensor_digest(vae.state_dict())==VQ_digest
        assert all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(not p.requires_grad and p.grad is None for p in vae.parameters());C.verify_protected_files(a);guard()
        result=dict(status='COMPLETE_FIXED_NATIVE_FINAL_DFIT200_ALL_FIVE_CONTEXT_CONDITIONS_AND10SCALES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(sp),input_completion=spec['Dfit_input_completion'],protocol=files['protocol.json'],fixed_final=done['final_model'],fit_completion=ref(Path(spec['fixed_final_completion_path'])),
            EEG_full=role,queries=200,concepts=20,conditions=list(Math.CONDITIONS),scales=10,all_token_NLL=ref(out/'all_condition_token_NLL.npy'),all_scale_NLL=ref(out/'all_condition_scale_NLL.npy'),summary=ref(out/'teacher_prefix_diagnostic_summary.json'),
            first_scale_prefix_causality_max_NLL_error=first_error,frozen_EEG_VAR_VQ_unchanged=True,teacher_forcing_uses_heldout_images_but_normal_generation_not_modified=True,
            fit_or_sampling_configuration_changed=False,primary_Dcheck_image_gate_changed=False,new_generated_images=0,new_optimizer_updates=0,original_three_arm_training_unchanged=True,
            peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base,seconds=time.monotonic()-started)
        write(out/'completion.json',result)
    except BaseException as error:
        write(out/'failure.json',dict(status='NATIVE_DFIT200_CONTEXT_DIAGNOSTIC_FAILURE_PRESERVED_NO_QUERY_EXCLUSION',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,full_query_denominator=200,new_optimizer_updates=0));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--job-name',required=True);q=p.parse_args();run(q.spec,q.job_name)
