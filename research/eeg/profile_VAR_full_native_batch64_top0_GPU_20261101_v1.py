"""Same FP32 native full likelihood, class batch64; exact top0 kernel replay."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import VAR_full_native_throughput_contract_CPU_20261101_v1 as C


def run(protocol,job):
    import resource
    q=C.setup(protocol);n,gate=C.dependency(q);s=C.read(C.NATIVE['path']);C.resources(q,starting=True)
    parent=C.read(n.mean_contract().PARENT['path']);repo=Path(parent['repository_root'])
    receipt=C.read(repo/'audit'/(job+'_job.json'))
    assert receipt['status']=='RUNNING' and receipt['pid']==os.getpid() and receipt['max_seconds']==900
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_bytes().decode().replace('\0',' ')
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==C.GPU
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    out=C.OUT/'profile';out.mkdir();started=time.monotonic();base=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;phase='STRICT_EXISTING_INPUTS'
    try:
        for root in reversed(s['native_module_roots']):sys.path.insert(0,root)
        sys.path.insert(0,s['assets']['vendor']);sys.path.insert(0,str(n.CODE))
        import numpy as np
        import torch
        from PIL import Image
        from EEG_generative_backend_loader_20261029_v1 import build_and_load
        from EEG_native_marginal_core_20261029_v2 import NativeMarginalDecoder
        import EEG_multiscale_native_kernel_DRAFT_20261101_v1 as K
        from profile_VAR_multiscale_native_GPU_20261101_v1 import digest
        for name,pin in s['native_module_bindings'].items():
            assert Path(sys.modules[name].__file__).resolve()==Path(n.checked(pin)).resolve()
        for key in ('author_inventory','public_VQ','public_VAR'):n.checked(s['assets'][key])
        for pin in s['assets']['data'].values():n.checked(pin)
        for pin in q['reference'].values():C.checked(pin)
        reference=np.load(q['reference']['token_logprobs']['path'],mmap_mode='r',allow_pickle=False)
        old_ll=np.load(q['reference']['scale_likelihoods']['path'],allow_pickle=False)
        assert reference.shape==(8,1000,680) and reference.dtype==np.float32 and old_ll.shape==(8,1000,10)
        tokens=np.load(s['assets']['data']['tokens']['path'],mmap_mode='r',allow_pickle=False)
        target=np.array(tokens[q['T_positions']],dtype=np.int64);del tokens
        torch.set_num_threads(2);torch.manual_seed(1729)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert not torch.cuda.is_initialized()
        assets=s['assets'];vae,d,backend=build_and_load(assets['author_source'],assets['vendor'],assets['public_VQ']['path'],assets['public_VAR']['path'],seed=1729)
        state=torch.load(assets['public_VAR']['path'],map_location='cpu',weights_only=True,mmap=True)
        native=NativeMarginalDecoder(d,state['class_emb.weight'].clone()).eval().requires_grad_(False);del state
        before=dict(VAR=digest(native),VQ=digest(vae))
        native.cuda();vae.cuda().eval().requires_grad_(False);torch.cuda.reset_peak_memory_stats()
        assert d.L==680 and d.V==4096 and d.C==1024
        last=0.
        def guard():
            nonlocal last
            assert time.monotonic()-started<q['GPU_soft_seconds']
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base<q['maximum_extra_CPU_bytes']
            assert torch.cuda.max_memory_allocated()<q['maximum_GPU_bytes']
            if time.monotonic()-last>10:C.resources(q);last=time.monotonic()
        phase='SAME_T8_ALL1000_FULL680_CLASS_BATCH64';guard()
        lp_bank=np.lib.format.open_memmap(out/'true_token_logprobs.npy',mode='w+',dtype=np.float32,shape=(8,1000,680));lp_bank[:]=np.nan
        ll=np.full((8,1000,10),np.nan,np.float64);coverage=np.zeros((8,1000),np.uint8)
        batch_times=[];maxerror=0.;loop=time.monotonic()
        with torch.inference_mode():
            for i in range(8):
                true=torch.from_numpy(target[i:i+1]).cuda()
                features=vae.quantize.idxBl_to_var_input(list(true.split([p*p for p in q['scales']],dim=1)))
                for lo in range(0,1000,64):
                    guard();hi=min(lo+64,1000);torch.cuda.synchronize();tick=time.monotonic()
                    output=native(features.expand(hi-lo,-1,-1),native.native_class_embedding[lo:hi])
                    lp=output.float().log_softmax(-1).gather(-1,true.expand(hi-lo,-1)[:,:,None]).squeeze(-1)
                    value=lp.cpu().numpy();assert value.dtype==np.float32 and np.isfinite(value).all()
                    error=float(np.abs(value-reference[i,lo:hi]).max());maxerror=max(maxerror,error)
                    assert np.allclose(value,reference[i,lo:hi],**q['numerical_guard'])
                    lp_bank[i,lo:hi]=value;coverage[i,lo:hi]+=1
                    ll[i,lo:hi]=np.stack([value[:,a:b].sum(-1,dtype=np.float64) for a,b in d.begin_ends],axis=1)
                    torch.cuda.synchronize();batch_times.append(time.monotonic()-tick);del output,lp,value
                lp_bank.flush()
            loop_seconds=time.monotonic()-loop
            assert np.array_equal(coverage,np.ones((8,1000),np.uint8)) and np.isfinite(ll).all()
            np.save(out/'scale_log_likelihoods.npy',ll,allow_pickle=False);np.save(out/'coverage.npy',coverage,allow_pickle=False)
            np.save(out/'true_codes.npy',target,allow_pickle=False)
            phase='NATIVE_UNTRUNCATED_TOP0_CACHED_AND_FRESH_PARITY';guard()
            captured=[];old_embedding=vae.quantize.embedding.forward
            def capture_codes(value):captured.append(value.detach().cpu().numpy().copy());return old_embedding(value)
            vae.quantize.embedding.forward=capture_codes
            condition=native.native_class_embedding[:1]
            try:cached=native.generate(condition,seed=1729,guidance=0.,top_k=0)
            finally:vae.quantize.embedding.forward=old_embedding
            fresh,trace=K.generate_class_chain(native,torch.zeros((1,10),device='cuda',dtype=torch.long),image_seed=1729,top_k=0,keep_tensor_trace=True)
            assert len(captured)==10 and all(np.array_equal(captured[k],record['codes'].cpu().numpy()) for k,record in enumerate(trace['records']))
            assert torch.equal(cached,fresh)
            np.savez(out/'top0_codes.npz',**{kind+str(k):(captured[k] if kind=='cached_' else trace['records'][k]['codes'].cpu().numpy()) for kind in ('cached_','fresh_') for k in range(10)})
            for name,tensor in (('cached_top0',cached),('fresh_top0',fresh)):
                pixels=tensor[0].detach().cpu().numpy();np.save(out/(name+'_pixels.npy'),pixels,allow_pickle=False)
                Image.fromarray((pixels.transpose(1,2,0).clip(0,1)*255).round().astype(np.uint8)).save(out/(name+'.png'))
            guard();assert before==dict(VAR=digest(native),VQ=digest(vae))
            assert not any(b.attn.caching or b.attn.cached_k is not None or b.attn.cached_v is not None for b in d.blocks)
        lp_bank.flush();del lp_bank
        C.write(out/'completion.json',dict(status='COMPLETE_SAME_NATIVE_FP32_BATCH64_T8_K1000_FULL680_AND_TOP0_PARITY',
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=C.ref(protocol),backend=backend,
            all5440000_target_logprobs_compared=True,maximum_target_logprob_difference=maxerror,
            class_batches=len(batch_times),batch_seconds=batch_times,loop_seconds=loop_seconds,seconds=time.monotonic()-started,
            peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base,
            top0_all680_codes_exact=True,top0_decoded_FP32_pixels_exact=True,top0_VQ_decodes=2,
            all_weights_byte_unchanged=True,state_digests=before,reference_loop_seconds=q['reference_loop_seconds'],
            files=[C.ref(p) for p in sorted(out.iterdir()) if p.is_file()],
            new_research_images=0,new_optimizer_updates=0,no_full_training_bank_or_EEG_admission=True))
        C.resources(q)
    except BaseException as error:
        C.write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,no_automatic_retry=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--job-name',required=True)
    a=p.parse_args();run(a.protocol,a.job_name)
