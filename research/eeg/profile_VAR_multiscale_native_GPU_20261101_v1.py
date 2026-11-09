"""Full pretrained native VAR: T8 x all1000 x all680, no EEG fitting."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
import VAR_multiscale_profile_contract_CPU_20261101_v1 as C


def digest(module):
    import torch
    h=hashlib.sha256()
    for key,value in module.state_dict().items():
        a=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run(protocol,job):
    s=C.setup(protocol);C.resources(s,starting=True)
    existing=C.mean_contract();parent=C.read(C.checked(existing.PARENT));R=Path(parent['repository_root'])
    receipt=C.read(R/'audit'/(job+'_job.json'))
    assert receipt['status']=='RUNNING' and receipt['pid']==os.getpid() and receipt['max_seconds']==900
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_bytes().decode().replace('\0',' ')
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==C.GPU
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    out=C.OUT/'profile';out.mkdir();started=time.monotonic();phase='STRICT_ASSETS';base=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    try:
        a=s['assets']
        for root in reversed(s['native_module_roots']):sys.path.insert(0,root)
        sys.path.insert(0,a['vendor'])
        for key in ('author_inventory','public_VQ','public_VAR'):C.checked(a[key])
        for pin in a['data'].values():C.checked(pin)
        token_completion=C.read(a['data']['token_completion']['path'])
        assert token_completion['status']=='COMPLETE_T13230_FROZEN_STRICT_VQ_TOKEN_BANK_INDEPENDENT_CPU_JOIN_AUDITED'
        assert token_completion['queries']==13230 and token_completion['token_width']==680 and token_completion['source_images_T_only'] is True
        assert (token_completion['indices']['sha256'],token_completion['identities']['sha256'])==(a['data']['tokens']['sha256'],a['data']['token_identities']['sha256'])
        ids=C.read(a['data']['token_identities']['path']);assert len(ids)==13230 and len({v['concept_id'] for v in ids})==1323
        assert [ids[p] for p in C.POSITIONS]==s['fixed_T_rows']
        import numpy as np
        import torch
        from PIL import Image
        from EEG_generative_backend_loader_20261029_v1 import build_and_load
        from EEG_native_marginal_core_20261029_v2 import NativeMarginalDecoder
        import EEG_multiscale_native_kernel_DRAFT_20261101_v1 as M
        for module_name,pin in s['native_module_bindings'].items():
            assert Path(sys.modules[module_name].__file__).resolve()==Path(C.checked(pin)).resolve()
        torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert not torch.cuda.is_initialized()
        source=np.load(a['data']['tokens']['path'],mmap_mode='r',allow_pickle=False)
        assert source.shape==(13230,680) and source.dtype==np.int16
        target=np.array(source[C.POSITIONS],dtype=np.int64);assert target.min()>=0 and target.max()<4096
        del source
        vae,d,backend=build_and_load(a['author_source'],a['vendor'],a['public_VQ']['path'],a['public_VAR']['path'],seed=1729)
        state=torch.load(a['public_VAR']['path'],weights_only=True,map_location='cpu',mmap=True);classes=state['class_emb.weight'].clone();del state
        native=NativeMarginalDecoder(d,classes).eval().requires_grad_(False)
        before=dict(VAR=digest(native),VQ=digest(vae));torch.cuda.reset_peak_memory_stats()
        native.cuda();vae.cuda().eval().requires_grad_(False)
        assert d.L==680 and d.C==1024 and d.V==4096 and len(native.native_class_embedding)==1001
        guard_reserve=C.resources(s)['required_remaining_bytes'];last_guard=0.
        def guard():
            nonlocal last_guard
            assert time.monotonic()-started<s['soft_GPU_seconds'],'Registered engineering soft time exhausted; no retry'
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base<s['maximum_extra_CPU_bytes']
            assert torch.cuda.max_memory_allocated()<s['maximum_GPU_bytes']
            if time.monotonic()-last_guard>10:
                assert C.mean_contract().headroom()>=s['minimum_runtime_CPU_headroom_bytes']
                assert shutil.disk_usage(out).free>=guard_reserve
                last_guard=time.monotonic()
        phase='T8_ALL1000_ALL680_NATIVE_FP32_LIKELIHOODS';guard()
        lp_bank=np.lib.format.open_memmap(out/'true_token_logprobs.npy',mode='w+',dtype=np.float32,shape=(8,1000,680));lp_bank[:]=np.nan
        scale_ll=np.full((8,1000,10),np.nan,np.float64);coverage=np.zeros((8,1000),np.uint8);batch_times=[];loop=time.monotonic()
        prefixes=[]
        with torch.inference_mode():
            for index in range(8):
                true=torch.from_numpy(target[index:index+1]).cuda()
                features=vae.quantize.idxBl_to_var_input(list(true.split([p*p for p in s['scales']],dim=1)))
                if index==0:prefixes.append(features.clone())
                for lo in range(0,1000,s['class_batch']):
                    guard();hi=min(lo+s['class_batch'],1000);torch.cuda.synchronize();tick=time.monotonic()
                    full=native(features.expand(hi-lo,-1,-1),native.native_class_embedding[lo:hi])
                    lp=full.float().log_softmax(-1).gather(-1,true.expand(hi-lo,-1)[:,:,None]).squeeze(-1)
                    values=lp.cpu().numpy();assert np.isfinite(values).all() and (values<=0).all()
                    lp_bank[index,lo:hi]=values
                    scale_ll[index,lo:hi]=np.stack([values[:,begin:end].sum(-1,dtype=np.float64) for begin,end in d.begin_ends],axis=1)
                    coverage[index,lo:hi]+=1;torch.cuda.synchronize();batch_times.append(time.monotonic()-tick)
                    del full,lp,values
                lp_bank.flush()
                pending=out/'progress.pending';pending.write_text(json.dumps(dict(T_rows_complete=index+1,classes_per_row=1000,tokens_per_row_class=680,loop_seconds=time.monotonic()-loop))+'\n')
                os.replace(pending,out/'progress.json')
            loop_seconds=time.monotonic()-loop;assert (coverage==1).all() and np.isfinite(scale_ll).all()
            np.save(out/'scale_log_likelihoods.npy',scale_ll,allow_pickle=False)
            np.save(out/'true_codes.npy',target,allow_pickle=False)
            np.save(out/'coverage.npy',coverage,allow_pickle=False)
            phase='COMPLETE_PREFIX_NATIVE_KERNEL_NUMERICS';condition=native.native_class_embedding[C.CLASSES]
            probe=prefixes[0].expand(4,-1,-1);codes=torch.from_numpy(target[:1]).cuda().expand(4,-1)
            full=native(probe,condition);errors=[]
            for k,(begin,end) in enumerate(d.begin_ends):
                guard();fresh=M.stage_logits(native,probe[:,:end-1],condition,k)
                assert torch.allclose(fresh,full[:,begin:end],**s['numerical_guard'])
                errors.append(float((fresh-full[:,begin:end]).abs().max()))
            explicit=M.complete_scale_log_likelihoods(native,probe,codes,condition).cpu().numpy()
            assert np.allclose(explicit,scale_ll[0,C.CLASSES],atol=2e-3,rtol=2e-5)
            altered=list(codes[:1].split([p*p for p in s['scales']],dim=1))
            altered=[v if k<3 else (v+7)%d.V for k,v in enumerate(altered)]
            altered_features=vae.quantize.idxBl_to_var_input(altered);end=d.begin_ends[3][1]
            assert torch.equal(prefixes[0][:,:end-1],altered_features[:,:end-1])
            assert torch.equal(M.stage_logits(native,prefixes[0][:,:end-1],condition[:1],3),
                M.stage_logits(native,altered_features[:,:end-1],condition[:1],3))
            # Same real T history, but old-class cached hidden states at the first three scales.
            old_cond=torch.cat((condition[:1],native.native_class_embedding[-1:]))
            new_cond=torch.cat((condition[1:2],native.native_class_embedding[-1:]))
            position=d.lvl_embed(d.lvl_1L)+d.pos_1LC
            token=old_cond[:,None]+d.pos_start+position[:,:1];accum=old_cond.new_zeros(1,d.Cvae,16,16)
            true_parts=list(codes[:1].split([p*p for p in s['scales']],dim=1))
            with native._cached_attention():
                for k in range(4):
                    active=old_cond if k<3 else new_cond;hidden=token
                    for block in d.blocks:hidden=block(hidden,d.shared_ada_lin(active),None)
                    if k==3:stale=d.get_logits(hidden,active);break
                    embedding=vae.quantize.embedding(true_parts[k]).transpose(1,2).reshape(1,d.Cvae,s['scales'][k],s['scales'][k])
                    accum,following=vae.quantize.get_next_autoregressive_input(k,10,accum,embedding)
                    lo,hi=d.begin_ends[k+1]
                    token=(d.word_embed(following.flatten(2).transpose(1,2))+position[:,lo:hi]).repeat(2,1,1)
            fresh=M.stage_logits(native,prefixes[0][:,:end-1].repeat(2,1,1),new_cond,3)
            stale_difference=float((stale-fresh).abs().max())
            np.savez(out/'old_class_stale_vs_current_class_fresh.npz',stale=stale.cpu().numpy(),fresh=fresh.cpu().numpy(),old_class=0,current_class=11,stage=3)
            del full,probe,codes,explicit
            # Capture the original cached implementation without editing its source or any tensor.
            phase='CACHED_VERSUS_FRESH_CONSTANT_CLASS_GENERATION';captured=[];indices=[]
            original_logits=d.get_logits;original_embedding=vae.quantize.embedding.forward
            def capture_logits(hidden,cond):
                value=original_logits(hidden,cond);captured.append(value.detach().cpu().numpy().copy());return value
            def capture_codes(value):indices.append(value.detach().clone());return original_embedding(value)
            d.get_logits=capture_logits;vae.quantize.embedding.forward=capture_codes
            try:reference=native.generate(condition[:1],seed=1729,guidance=0.,top_k=600)
            finally:d.get_logits=original_logits;vae.quantize.embedding.forward=original_embedding
            guard();sequence=torch.zeros((1,10),device='cuda',dtype=torch.long)
            replay,proof=M.generate_class_chain(native,sequence,image_seed=1729,top_k=600,keep_tensor_trace=True)
            token_equal=all(torch.equal(item['codes'],old) for item,old in zip(proof['records'],indices))
            pixel_equal=torch.equal(reference,replay)
            fixed_errors=[]
            fixed_payload={}
            for k,item in enumerate(proof['records']):
                fixed_payload[f'cached_logits_{k}']=captured[k]
                fixed_payload[f'fresh_logits_{k}']=item['logits'].cpu().numpy()
                fixed_payload[f'cached_codes_{k}']=indices[k].cpu().numpy()
                fixed_payload[f'fresh_codes_{k}']=item['codes'].cpu().numpy()
                fixed_errors.append(float(np.max(np.abs(fixed_payload[f'cached_logits_{k}']-fixed_payload[f'fresh_logits_{k}']))))
            np.savez(out/'fixed_class_native_and_fresh_trace.npz',**fixed_payload)
            del fixed_payload,captured
            def save_image(tensor,name):
                array=tensor[0].detach().cpu().numpy();np.save(out/(name+'_pixels.npy'),array,allow_pickle=False)
                Image.fromarray((array.transpose(1,2,0).clip(0,1)*255).round().astype(np.uint8)).save(out/(name+'.png'))
            save_image(reference,'fixed_native');save_image(replay,'fixed_fresh')
            C.write(out/'constant_class_actual_parity.json',dict(all680_sampled_tokens_exact=token_equal,decoded_FP32_pixels_exact=pixel_equal,
                cached_vs_fresh_logit_maxerrors=fixed_errors,numerical_guard=s['numerical_guard']))
            assert token_equal and pixel_equal,'Full native constant-class replay failed; evidence retained without retry'
            with np.load(out/'fixed_class_native_and_fresh_trace.npz',allow_pickle=False) as archive:
                assert all(np.allclose(archive[f'cached_logits_{k}'],item['logits'].cpu().numpy(),**s['numerical_guard']) for k,item in enumerate(proof['records']))
            phase='SWITCHED_CLASSES_DEFINED_FRESH_PREFIX_KERNEL';guard()
            chain=torch.tensor([[0,11,287,999,0,11,287,999,0,11]],device='cuda',dtype=torch.long)
            switched,trace=M.generate_class_chain(native,chain,image_seed=1729,top_k=600,keep_tensor_trace=True)
            history=vae.quantize.idxBl_to_var_input([item['codes'] for item in trace['records']]);switch_errors=[];payload={}
            for k,(begin,end) in enumerate(d.begin_ends):
                guard();cond=torch.cat((native.native_class_embedding[chain[:,k]],native.native_class_embedding[-1:]))
                expected=M.stage_logits(native,history[:,:end-1].repeat(2,1,1),cond,k)
                assert torch.allclose(trace['records'][k]['logits'],expected,**s['numerical_guard'])
                switch_errors.append(float((trace['records'][k]['logits']-expected).abs().max()))
                payload[f'logits_{k}']=trace['records'][k]['logits'].cpu().numpy();payload[f'codes_{k}']=trace['records'][k]['codes'].cpu().numpy()
            np.savez(out/'switched_class_fresh_trace.npz',classes=chain.cpu().numpy(),**payload);del payload
            save_image(switched,'switched_fresh')
            future=chain.clone();future[:,1:]=999
            future_image,future_trace=M.generate_class_chain(native,future,image_seed=1729,top_k=600,keep_tensor_trace=True)
            assert torch.equal(trace['records'][0]['codes'],future_trace['records'][0]['codes'])
            save_image(future_image,'changed_future_class')
            phase='FROZEN_STATE_AND_RESOURCE_CLOSURE';guard()
            assert before==dict(VAR=digest(native),VQ=digest(vae))
            assert all(not b.attn.caching and b.attn.cached_k is None and b.attn.cached_v is None for b in d.blocks)
        for pin in s['inherited_source_pins']:C.checked(pin)
        lp_bank.flush();del lp_bank
        files=[C.ref(p) for p in sorted(out.iterdir()) if p.is_file()]
        C.write(out/'completion.json',dict(status='COMPLETE_PRETRAINED_T8_ALL1000_FULL680_NATIVE_VAR_ENGINEERING_ONLY',
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=C.ref(protocol),backend=backend,
            T_rows=8,classes=1000,scales=10,full_tokens=680,class_batch=8,fixed_T_rows=s['fixed_T_rows'],
            native_FP32_no_autocast=True,fresh_prefix_vs_full680_maxerrors=errors,
            actual_current_and_future_target_perturbation_preserves_current_scale=True,
            old_class_stale_cache_vs_current_class_fresh_maxerror=stale_difference,
            fixed_class_all680_tokens_exact=token_equal,fixed_class_decoded_pixels_exact=pixel_equal,
            fixed_class_cached_vs_fresh_logit_maxerrors=fixed_errors,switched_class_prefix_logit_maxerrors=switch_errors,
            future_class_does_not_change_first_scale_codes=True,all_VAR_VQ_weights_unchanged=True,
            state_digests=before,loop_seconds=loop_seconds,batch_seconds=batch_times,seconds=time.monotonic()-started,
            peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base,
            files=files,engineering_VQ_decodes=4,new_research_images=0,new_optimizer_updates=0,
            real_EEG_fit=False,full13230_image_emission_bank=False,novelty_or_EEG_image_gain_established=False,NEW_contacted=False))
        C.resources(s)
    except BaseException as error:
        C.write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,no_automatic_retry=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--job-name',required=True)
    a=p.parse_args();run(a.protocol,a.job_name)
