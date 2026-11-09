"""Profile/admission-gated producer. Importing or a DRAFT spec never trains.

Actual dispatch is owned by the separately invoked finite controller/wrapper.
Native-global reads no local-memory file. All recovery is from whole logical
optimizer steps, with bounded newly-owned resume slots and immutable attempts.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,gc,hashlib,json,shutil,signal,subprocess,time,traceback
from pathlib import Path
from AR_anchored_training_contract_CPU_20261029_v1 import checked,ref,sha,load_admission,validate_sources,validate_profiles,validate_image_binding,load_training_banks,verify_protected_files

def write_once(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def replace_json(path,value):
    pending=path.with_name(path.name+f'.pending.{os.getpid()}.{time.time_ns()}');write_once(pending,value);os.replace(pending,path)
def peak_CPU():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def tensor_digest(state):
    import torch
    digest=hashlib.sha256()
    for key,value in state.items():
        t=value.detach().cpu().contiguous();digest.update(key.encode());digest.update(str(t.dtype).encode());digest.update(str(tuple(t.shape)).encode());digest.update(t.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
def payload_bytes(root):
    return sum(p.stat().st_size for p in root.glob('*/resume/slot_*/state.pt'))+sum(p.stat().st_size for p in root.glob('*/final_model.pt'))
def disk_guard(admission):
    root=Path(admission['output_root']);r=admission['resources'];planned=len(admission['arms'])*(2*r['maximum_resume_payload_bytes']+r['maximum_final_model_bytes'])
    remaining=max(0,planned-payload_bytes(root))
    assert shutil.disk_usage(root).free>=remaining+r['unreserved_disk_buffer_bytes'], 'Registered remaining outputs plus buffer do not fit'

def log_prefix(attempt_root,start,end,*,arm,admission_sha):
    digest=hashlib.sha256()
    for update in range(start,end+1):
        p=attempt_root/'updates'/f'update_{update:08d}.json';raw=p.read_bytes();event=json.loads(raw)
        assert event['arm']==arm and event['spec_sha256']==admission_sha and event['update']==update
        assert sum(event['micro_batch_rows'])==len(event['array_positions']) and len(event['array_positions']) in (16,14)
        digest.update(p.name.encode());digest.update(len(raw).to_bytes(8,'little'));digest.update(raw)
    return digest.hexdigest()
def validate_log_chain(arm_root,chain,*,updates,arm,admission_sha):
    end=0
    for item in chain:
        assert item['start_update']==end+1 and item['end_update']>=item['start_update'] and 1<=item['attempt_index']<=100000
        attempt=arm_root/'attempts'/f"attempt_{item['attempt_index']:05d}"
        assert log_prefix(attempt,item['start_update'],item['end_update'],arm=arm,admission_sha=admission_sha)==item['prefix_sha256']
        end=item['end_update']
    assert end==updates

class CompletedStepBoundary(Exception):pass

def run(admission_path,arm,attempt_index,maximum_updates,job_name):
    # All of these checks precede torch import, output creation and CUDA.
    admission,admission_sha=load_admission(admission_path,arm=arm)
    assert 1<=attempt_index<=100000 and 1<=maximum_updates<=admission['resources']['maximum_updates_per_job']
    validate_sources(admission,producer_path=Path(__file__))
    validate_profiles(admission);validate_image_binding(admission);verify_protected_files(admission)
    began=time.monotonic();base_CPU=peak_CPU();root=Path(admission['output_root']);root.mkdir(exist_ok=True)
    assert root.resolve()==root and not root.is_symlink()
    registration=root/'registration.json'
    if not registration.exists():write_once(registration,dict(admission_sha256=admission_sha,admission=ref(admission_path)))
    else:assert json.loads(registration.read_bytes())['admission_sha256']==admission_sha
    arm_root=root/arm;arm_root.mkdir(exist_ok=True);attempts=arm_root/'attempts';attempts.mkdir(exist_ok=True)
    prior_attempts=sorted(p for p in attempts.iterdir() if p.is_dir());assert [p.name for p in prior_attempts]==[f'attempt_{i:05d}' for i in range(1,len(prior_attempts)+1)]
    assert attempt_index==len(prior_attempts)+1 and not (arm_root/'completion.json').exists()
    attempt=attempts/f'attempt_{attempt_index:05d}';attempt.mkdir();(attempt/'updates').mkdir();phase='LOCK_AND_DEVICE'
    requested_stop=[]
    def stop_signal(signum,frame):requested_stop.append(signum)
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    try:
        R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
        parent=Path('/proc')/str(os.getppid());parent_args=(parent/'cmdline').read_bytes().decode().replace('\0',' ')
        expected_wrapper='run_gpu_job_v2.py' if admission['host_role']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'
        assert expected_wrapper in parent_args and '/' not in job_name
        receipt_path=R/'audit'/(job_name+'_job.json')
        for _ in range(30):
            if receipt_path.exists():break
            time.sleep(.1)
        receipt=json.loads(receipt_path.read_bytes());assert receipt['pid']==os.getpid() and receipt['wrapper_pid']==os.getppid() and receipt['status']=='RUNNING'
        assert float(receipt['max_seconds'] if admission['host_role']=='old' else receipt['maximum_seconds'])==900
        assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==admission['GPU_UUID']
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        disk_guard(admission);phase='FROZEN_DATA_IDENTITY_AND_REAL_BACKEND'
        global_bank,memory_bank,targets,concepts=load_training_banks(admission,arm)
        import torch
        import AR_anchored_training_core_DRAFT_20261029_v3 as Core
        from EEG_generative_backend_loader_20261029_v1 import build_and_load
        from EEG_anchored_attention_20261029_v1 import EEGAnchoredAttentionDecoder
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        vae,decoder,backend=build_and_load(admission['author_source'],admission['vendor'],checked(admission['public_VQ']),checked(admission['public_VAR']),seed=1729)
        initial=tensor_digest(decoder.state_dict());assert initial==admission['initial_generator_tensor_digest']
        assert backend['status']=='EXACT_AUTHOR_VQ_AND_EXPLICIT_CLASS_TO_EEG_VAR_TRANSFER'
        vae.cuda().eval().requires_grad_(False);decoder.cuda();decoder.rng=torch.Generator(device='cuda');decoder.cond_drop_rate=.1
        model=EEGAnchoredAttentionDecoder(decoder);optimizer=Core.optimizer_for(model);shuffle=torch.Generator().manual_seed(1729)
        torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
        cursor=Core.new_cursor();chain=[];store=arm_root/'resume'
        if not store.exists():
            store.mkdir();Core.create_rotating_store(store,arm=arm,spec_sha256=admission_sha,authorize_owned_previous_retirement=True)
        else:
            state,manifest=Core.read_rotating_current(store)
            assert manifest['status']=='READY', 'Interrupted store preparation needs an explicit audited recovery'
            chain=state['producer_log_chain'];validate_log_chain(arm_root,chain,updates=state['cursor']['updates'],arm=arm,admission_sha=admission_sha)
            cursor=Core.restore(state,model,optimizer,shuffle,arm=arm,spec_sha256=admission_sha,count=13230,batch=16,total_epochs=10,device='cuda',micro_batch=2)
            del state;gc.collect()
        start_update=cursor['updates'];assert start_update<8270;torch.cuda.reset_peak_memory_stats()
        resources=admission['resources'];phase='COMPLETE_T_LOGICAL_MICRO2_UPDATES'
        def guard():
            assert torch.cuda.max_memory_allocated()<resources['maximum_GPU_allocated_bytes']
            assert peak_CPU()-base_CPU<resources['maximum_extra_CPU_bytes']
            assert time.monotonic()-began<resources['producer_hard_seconds']
        def observe(event,current):
            Core.write_update_event(attempt/'updates',event,arm=arm,spec_sha256=admission_sha)
            replace_json(arm_root/'progress.json',dict(status='FORMAL_FIXED_T_AR_TRAINING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,
                attempt_index=attempt_index,epoch=current['epoch'],completed_epochs=current['completed_epochs'],position=current['position'],updates=current['updates'],
                expected_updates=8270,query_exposures=current['query_exposures'],expected_query_exposures=132300,micro_batch=2,last_event=event,
                admission_sha256=admission_sha,checkpoint_committed_through_update=start_update))
            if requested_stop or (root/'STOP_REQUESTED.json').exists() or time.monotonic()-began>=resources['training_soft_seconds']:
                raise CompletedStepBoundary()
        try:
            Core.run_updates(model,optimizer,shuffle,cursor,arm=arm,global_bank=global_bank,memory_bank=memory_bank,target_bank=targets,concept_ids=concepts,
                total_epochs=10,batch=16,micro_batch=2,maximum_updates=maximum_updates,device='cuda',guard=guard,observer=observe)
        except CompletedStepBoundary:
            assert cursor['logical_update_in_progress'] is False
        assert cursor['updates']>start_update and cursor['logical_update_in_progress'] is False
        phase='BOUNDED_CHECKPOINT_AND_IMMUTABLE_LOG_COMMIT';disk_guard(admission)
        chain.append(dict(attempt_index=attempt_index,start_update=start_update+1,end_update=cursor['updates'],
            prefix_sha256=log_prefix(attempt,start_update+1,cursor['updates'],arm=arm,admission_sha=admission_sha)))
        state=Core.snapshot(model,optimizer,shuffle,cursor,arm=arm,spec_sha256=admission_sha,device='cuda',micro_batch=2);state['producer_log_chain']=chain
        manifest=Core.publish_rotating_checkpoint(store,state);assert manifest['current']['state_bytes']<=resources['maximum_resume_payload_bytes']
        del state;gc.collect();guard()
        for record in cursor['epoch_records']:
            epochs=arm_root/'epoch_records';epochs.mkdir(exist_ok=True);p=epochs/f"epoch_{record['epoch']:02d}.json"
            if p.exists():assert json.loads(p.read_bytes())==record
            else:write_once(p,record)
        if cursor['completed_epochs']==10:
            assert cursor['updates']==8270 and cursor['query_exposures']==132300
            phase='FIXED_FINAL_MODEL_PUBLICATION';final=arm_root/'final_model.pt';assert not final.exists()
            pending=arm_root/f'final_model.pending.{os.getpid()}.pt'
            with pending.open('xb') as f:
                torch.save(dict(decoder=Core.cpu_tree(decoder.state_dict()),arm=arm,admission_sha256=admission_sha,
                    epoch=10,updates=8270,query_exposures=132300,EEG_bank_source_sha256=admission['data']['EEG_completion']['sha256'],
                    token_bank_source_sha256=admission['data']['token_completion']['sha256'],initial_generator_tensor_digest=initial,
                    trained_decoder_tensor_digest=tensor_digest(decoder.state_dict())),f);f.flush();os.fsync(f.fileno())
            assert pending.stat().st_size<=resources['maximum_final_model_bytes'];pending.rename(final);guard();verify_protected_files(admission)
            write_once(arm_root/'completion.json',dict(status='COMPLETE_FIXED_T_10_EPOCH_ANCHORED_AR_ARM',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,
                admission=ref(admission_path),host_role=admission['host_role'],GPU_UUID=admission['GPU_UUID'],epochs=10,updates=8270,query_exposures=132300,
                final_model=ref(final),current_resume=manifest['current'],previous_resume=manifest['previous'],committed_log_chain=chain,
                checkpoint_selection='Fixed final epoch10; no D/V/official score selection',native_reads_local_memory_file=memory_bank is not None,
                pretrained_backend=backend,new_research_images=0,all10_training_claimed=False,old_ordinary_unchanged=True))
        verify_protected_files(admission);disk_guard(admission)
        write_once(attempt/'completion.json',dict(status='COMPLETE_FINITE_FORMAL_AR_LOGICAL_STEP_CHUNK',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,
            attempt_index=attempt_index,start_update=start_update,end_update=cursor['updates'],completed_epochs=cursor['completed_epochs'],query_exposures=cursor['query_exposures'],
            local_memory_file_read=memory_bank is not None,micro_batch=2,current_resume=manifest['current'],log_prefix=chain[-1],
            stop_requested=bool(requested_stop) or (root/'STOP_REQUESTED.json').exists(),peak_GPU_allocated=torch.cuda.max_memory_allocated(),
            peak_GPU_reserved=torch.cuda.max_memory_reserved(),extra_CPU_peak=peak_CPU()-base_CPU,seconds=time.monotonic()-began,admission_sha256=admission_sha))
    except BaseException as error:
        write_once(attempt/'failure.json',dict(status='FORMAL_AR_CHUNK_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,
            error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-began,attempt_index=attempt_index,arm=arm,
            partial_micro_update=locals().get('cursor',{}).get('logical_update_in_progress'),new_research_images=0))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--admission',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--attempt-index',type=int,required=True)
    p.add_argument('--maximum-updates',type=int,required=True);p.add_argument('--job-name',required=True);a=p.parse_args()
    try:run(a.admission,a.arm,a.attempt_index,a.maximum_updates,a.job_name)
    except ValueError as error:
        if str(error).startswith('NOT_ADMITTED_'):print(str(error));raise SystemExit(64)
        raise
