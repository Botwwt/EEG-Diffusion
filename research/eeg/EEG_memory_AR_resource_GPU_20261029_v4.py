"""Actual pretrained resource test: real46 T rows, public VQ and public VAR.

Three matched logical16/micro2 branches take three discarded Adam steps each.
This measures pretrained activation/gradient/optimizer feasibility; method
effectiveness remains untested. No learned state or research image is published.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,ctypes,datetime,gc,hashlib,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from EEG_generative_backend_loader_20261029_v1 import build_and_load,sha,SCALES
from EEG_anchored_attention_20261029_v1 import EEGAnchoredAttentionDecoder
import AR_anchored_training_core_DRAFT_20261029_v3 as Core

def check(item):
    p=Path(item['path']);assert sha(p)==item['sha256'],str(p)
    if 'bytes' in item:assert p.stat().st_size==item['bytes']
    return p
def ref(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def tensor_digest(state):
    h=hashlib.sha256()
    for key,value in state.items():
        v=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def tree_digest(value):
    h=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            x=item.detach().cpu().contiguous();h.update(b'T');h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item,dict):
            h.update(b'D')
            for key,child in item.items():h.update(repr(key).encode());visit(child)
        elif isinstance(item,(tuple,list)):
            h.update(b'L')
            for child in item:visit(child)
        else:h.update(repr(item).encode())
    visit(value);return h.hexdigest()

def run(spec_path):
    began=time.monotonic();spec=json.loads(spec_path.read_bytes());out=Path(spec['output']);assert not out.exists();out.mkdir();phase='REAL_T_MATCHED_MICROBATCH_RESOURCE_INPUTS';base=peak()
    try:
        for item in spec['source_pins']:check(item)
        uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==spec['GPU_UUID'] and uuid in {'GPU-0a99d971-a46c-6e81-f157-c33243c6d57a','GPU-62a1aee8-4740-dc59-93f4-d50644db650d'}
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        memory=json.loads(check(spec['memory_completion']).read_bytes());assert memory['status']=='COMPLETE_REAL_EEG_T13230_GENUINE_OOF_AND_DCHECK400_LOCAL_MEMORY' and memory['all_T_rows_exactly_once_from_concept_excluded_encoder']
        identities=json.loads(check(memory['files']['identities']).read_bytes());assert len(identities['T'])==13230
        global_full=np.load(check(memory['files']['T_OOF_c']),mmap_mode='r',allow_pickle=False);local_full=np.load(check(memory['files']['T_OOF_memory']),mmap_mode='r',allow_pickle=False);target_full=np.load(check(spec['token_codes']),mmap_mode='r',allow_pickle=False)
        assert global_full.shape==(13230,1024) and local_full.shape==(13230,63,1024) and target_full.shape==(13230,680) and target_full.dtype==np.int16
        g=np.array(global_full[:46],copy=True);h=np.array(local_full[:46],copy=True);targets=np.array(target_full[:46],copy=True);concepts=[q['concept_id'] for q in identities['T'][:46]]
        assert np.isfinite(g).all() and np.isfinite(h).all() and np.allclose(h.mean(1,dtype=np.float64),g,atol=2e-5,rtol=2e-5)
        assert len({q['stimulus_id'] for q in identities['T'][:46]})==46 and targets.min()>=0 and targets.max()<4096
        torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        phase='STRICT_PRETRAINED_VQ_AND_PUBLIC_VAR_WITH_SHARED_SEEDED_EEG_PROJECTION'
        vae,decoder,backend_proof=build_and_load(spec['author_source'],spec['vendor'],check(spec['VQ_checkpoint']),check(spec['VAR_checkpoint']))
        assert sum(p.numel() for p in decoder.parameters())==310308096
        initial={key:value.detach().clone() for key,value in decoder.state_dict().items()};initial_digest=tensor_digest(initial)
        vq_before=tensor_digest(vae.state_dict());vae.cuda();decoder.cuda();decoder.rng=torch.Generator(device='cuda')
        wrapper=EEGAnchoredAttentionDecoder(decoder);assert {id(p) for p in wrapper.parameters()}=={id(p) for p in decoder.parameters()}
        maximum_gpu=0;maximum_reserved=0;proofs=[];rng_initial=[];rng_final=[];resume_probe=None
        def cuda_rng_digests():return [hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest() for state in torch.cuda.get_rng_state_all()]
        def budget():
            nonlocal maximum_gpu,maximum_reserved
            maximum_gpu=max(maximum_gpu,torch.cuda.max_memory_allocated());maximum_reserved=max(maximum_reserved,torch.cuda.max_memory_reserved())
            assert time.monotonic()-began<840 and maximum_gpu<20<<30 and peak()-base<10<<30
        for arm in Core.ARMS:
            phase='ACTUAL_LOGICAL_16_MICRO_2_'+arm;decoder.load_state_dict(initial,strict=True);decoder.cond_drop_rate=.1
            torch.manual_seed(1729);shuffle=torch.Generator().manual_seed(1729);cursor=Core.new_cursor();cursor['order']=torch.arange(46,dtype=torch.int64)
            optimizer=Core.optimizer_for(wrapper);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();arm_start=time.monotonic()
            rng_initial.append(cuda_rng_digests())
            events=Core.run_updates(wrapper,optimizer,shuffle,cursor,arm=arm,global_bank=g,memory_bank=None if arm=='native_global' else h,target_bank=targets,concept_ids=concepts,total_epochs=1,batch=16,micro_batch=2,maximum_updates=3,device='cuda',guard=budget)
            rng_final.append(cuda_rng_digests())
            torch.cuda.synchronize();budget()
            update_compute_seconds=time.monotonic()-arm_start
            assert len(events)==3 and [event['micro_batch_rows'] for event in events]==[[2]*8,[2]*8,[2]*7]
            assert [event['array_positions'] for event in events]==[list(range(16)),list(range(16,32)),list(range(32,46))]
            assert cursor['updates']==3 and cursor['query_exposures']==46 and cursor['completed_epochs']==1 and cursor['logical_update_in_progress'] is False
            qkv=decoder.blocks[0].attn.mat_qkv.weight;assert qkv.grad is not None and torch.isfinite(qkv.grad).all() and qkv.grad.norm()>0
            assert all(torch.isfinite(parameter).all() for parameter in decoder.parameters())
            assert float(optimizer.state[qkv]['step'])==3 and all(parameter.grad is None for parameter in vae.parameters())
            if arm=='native_global':
                phase='ONE_OWNED_NATIVE_FULL_ADAM_RNG_SERIALIZATION_AND_RESTORE_PROBE'
                live_model_digest=tensor_digest(wrapper.state_dict());live_optimizer_digest=tree_digest(optimizer.state_dict())
                saved=Core.snapshot(wrapper,optimizer,shuffle,cursor,arm=arm,spec_sha256=sha(spec_path),device='cuda',micro_batch=2)
                payload=out/'OWNED_NATIVE_RESUME_PROBE_transient.pt';assert not payload.exists();save_start=time.monotonic()
                with payload.open('xb') as stream:torch.save(saved,stream);stream.flush();os.fsync(stream.fileno())
                save_seconds=time.monotonic()-save_start;payload_bytes=payload.stat().st_size
                # Release the complete 3.7GB CPU snapshot before mmap reload.
                del saved;gc.collect()
                try:ctypes.CDLL(None).malloc_trim(0)
                except (AttributeError,OSError):pass
                payload_sha=sha(payload);budget();optimizer=None;decoder.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                load_start=time.monotonic();loaded=torch.load(payload,map_location='cpu',weights_only=True,mmap=True);load_seconds=time.monotonic()-load_start
                assert tensor_digest(loaded['model'])==live_model_digest and tree_digest(loaded['optimizer'])==live_optimizer_digest
                assert all(float(state['step'])==3 for state in loaded['optimizer']['state'].values())
                # Prove actual restoration into an empty optimizer and changed
                # model/RNG/shuffle state, rather than rereading an unused file.
                with torch.no_grad():decoder.head.weight[0,0].add_(1.)
                torch.manual_seed(999);shuffle.manual_seed(999);optimizer=Core.optimizer_for(wrapper);restore_start=time.monotonic()
                restored=Core.restore(loaded,wrapper,optimizer,shuffle,arm=arm,spec_sha256=sha(spec_path),count=46,batch=16,total_epochs=1,device='cuda',micro_batch=2)
                torch.cuda.synchronize();restore_seconds=time.monotonic()-restore_start
                assert tensor_digest(wrapper.state_dict())==live_model_digest and tree_digest(optimizer.state_dict())==live_optimizer_digest
                assert restored['updates']==3 and restored['query_exposures']==46 and restored['completed_epochs']==1
                assert tree_digest(restored)==tree_digest(loaded['cursor'])
                assert torch.equal(shuffle.get_state(),loaded['shuffle_rng']) and torch.equal(torch.get_rng_state(),loaded['torch_rng'])
                assert all(torch.equal(actual,expected) for actual,expected in zip(torch.cuda.get_rng_state_all(),loaded['cuda_rng']))
                resume_probe=dict(status='ACTUAL_NATIVE_FULL_MODEL_ADAM_CURSOR_AND_RNG_SERIALIZE_RELOAD_RESTORE_PASSED',payload_bytes=payload_bytes,payload_sha256=payload_sha,serialize_seconds=save_seconds,mmap_load_seconds=load_seconds,restore_seconds=restore_seconds,model_state_exact=True,all_Adam_moments_and_steps_exact=True,Adam_state_count=len(loaded['optimizer']['state']),parameter_count=len(list(wrapper.parameters())),cursor_exact=True,CPU_CUDA_shuffle_RNG_exact=True,CPU_snapshot_freed_before_reload=True,owned_engineering_transient_retirement_authorized=True,retired_after_verification=False,peak_extra_CPU_after_probe=peak()-base)
                del loaded;gc.collect();budget()
            record=dict(arm=arm,logical_updates=3,real_query_exposures=46,microbatch_sequence=[8,8,7],queries_per_logical_update=[16,16,14],events=events,cursor_completed_epochs=cursor['completed_epochs'],peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),three_update_compute_seconds=update_compute_seconds,seconds_including_native_resume_probe=time.monotonic()-arm_start,existing_QKV_gradient_nonzero=True)
            proofs.append(record);write(out/(arm+'_resource.json'),record)
            optimizer=None;decoder.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();budget()
            if arm=='native_global':
                assert payload.parent.resolve()==out.resolve() and payload.name=='OWNED_NATIVE_RESUME_PROBE_transient.pt' and payload.is_file() and not payload.is_symlink()
                assert payload.stat().st_size==resume_probe['payload_bytes'] and sha(payload)==resume_probe['payload_sha256']
                payload.unlink();assert not payload.exists();resume_probe['retired_after_verification']=True;resume_probe['retirement_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
                write(out/'native_resume_serialization_probe.json',resume_probe)
        decoder.load_state_dict(initial,strict=True);assert tensor_digest(decoder.state_dict())==initial_digest and tensor_digest(vae.state_dict())==vq_before
        assert not any(p.requires_grad or p.grad is not None for p in vae.parameters());budget()
        assert resume_probe and resume_probe['retired_after_verification']
        write(out/'completion.json',dict(status='PASS_REAL_46_T_LOGICAL16_MICRO2_ALL3_ARMS_PRETRAINED_VAR_RESOURCE',resource_probe_version=4,native_resume_serialization_probe=resume_probe,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(spec_path),GPU_UUID=uuid,random_VAR=False,pretrained_VAR=True,pretrained_VAR_loaded=True,pretrained_VQ=True,strict_backend_transfer=backend_proof,public_VAR_sha256=backend_proof['VAR_SHA256'],public_VQ_sha256=backend_proof['VQ_SHA256'],actual_core_source_sha256=sha(Path(Core.__file__)),source_EEG=spec['memory_completion'],source_T_codes=spec['token_codes'],source_VAR=spec['VAR_checkpoint'],fixed_query_positions=list(range(46)),logical_batch=16,micro_batch=2,logical_batch_rows=[16,16,14],arm_names=list(Core.ARMS),physical_micro_batches_per_arm=[8,8,7],optimizer_updates_per_arm=3,total_discarded_optimizer_updates=9,total_resource_query_exposures=138,all_temporary_updates_finite=True,all_temporary_updates_reset=True,all_initial_VAR_values_restored=True,VQ_frozen_unchanged=True,INITIAL_FULL_GENERATOR_TENSOR_DIGEST=initial_digest,initial_generator_tensor_digest=initial_digest,initial_generator_digest_scope='VAR state_dict including new seeded EEG projection; frozen VQ digest recorded separately',VAR_initial_tensor_digest=initial_digest,VQ_tensor_digest=vq_before,CUDA_RNG_matched=rng_initial[0]==rng_initial[1]==rng_initial[2] and rng_final[0]==rng_final[1]==rng_final[2],CUDA_RNG_initial_digests=rng_initial,CUDA_RNG_final_digests=rng_final,CUDA_RNG_comparison_scope='Initial and final full CUDA RNG states across arms; no bitwise parameter trajectory claim',VAR_parameters=310308096,added_wrapper_parameters=0,native_training_exercised_without_memory_argument=True,training_conditional_dropout=.1,training_drop_path=.1*16/24,attention_l2_normalization=True,fused_or_flash_attention=False,resource_fixtures_partial_last_concept_not_formal_training=True,proofs=proofs,peak_GPU_allocated=maximum_gpu,peak_GPU_reserved=maximum_reserved,extra_CPU_peak=peak()-base,seconds=time.monotonic()-began,new_formal_trained_models=0,new_research_images=0,efficacy_claimed=False))
        print(json.dumps(dict(status='PASS_REAL_46_T_LOGICAL16_MICRO2_ALL3_ARMS_PRETRAINED_VAR_RESOURCE',seconds=time.monotonic()-began,peak_GPU_allocated=maximum_gpu,INITIAL_FULL_GENERATOR_TENSOR_DIGEST=initial_digest)))
    except BaseException as error:
        write(out/'failure.json',dict(status='AR_LOGICAL16_MICRO2_FULLSHAPE_RESOURCE_FAILURE_PRESERVED',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-began));raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);run(parser.parse_args().spec)
