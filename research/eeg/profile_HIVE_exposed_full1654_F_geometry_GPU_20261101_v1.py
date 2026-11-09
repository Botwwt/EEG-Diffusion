"""Four discarded updates per matched head on the actual full1654 input bank.

Engineering only: no learned checkpoint, image or scientific outcome is saved.
This program cannot perform or authorize the final31014-update fits.
"""
import argparse,datetime,gc,hashlib,io,json,os,sys,time
from pathlib import Path


def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)


def run(path):
    start=time.monotonic();s=json.loads(Path(path).read_bytes())
    assert s['status']=='SOURCE_REVIEWED_FULL1654_F_GEOMETRY_ENGINEERING_ONLY'
    assert s['host']=='old' and s['training_seed']==1729 and s['temporary_updates_per_arm']==4
    assert not s['final_fits_admitted'] and not s['official_test_payloads_allowed']
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_bytes().decode()
    for pin in s['source_pins']:assert ref(pin['path'])==pin
    sys.path.insert(0,str(Path(path).parent));sys.path.insert(1,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    sys.path.insert(0,str(Path(path).parent))
    import numpy as np,torch
    import torch.nn.functional as F
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_ordinary_geometry_control_20261030_v1 import EuclideanConditionControl
    import brainhive_exposed_fullfit_component_inputs_CPU_20261101_v1 as I
    from brainhive_functional_training_contract_v2 import resource_headroom
    before=resource_headroom();assert before['effective_headroom_bytes']>=24<<30
    import subprocess
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert ref(s['consumer_review']['path'])==s['consumer_review']
    review=json.loads(Path(s['consumer_review']['path']).read_bytes())
    assert review['status']=='PASS_ACTUAL_ALL165400_NATIVE_RAW_JOIN_AND_ALL20_OWN_AFFINE_CONSUMER'
    assert review['input_bank']==s['input_bank'] and review['asset_protocol']==s['asset_protocol']
    inputs=I.load_verified(s['input_bank'],s['asset_protocol'])
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    order_seed=int(hashlib.sha256(b'1729|BrainHIVE_bounded_v1|pass0').hexdigest()[:16],16)%(2**63-1)
    indices=torch.randperm(165400,generator=torch.Generator().manual_seed(order_seed))[:64].numpy()
    cpu=I.cpu_batch(inputs,indices);batch={k:cpu[k].cuda() for k in ('condition','weight','bias','target')}
    results=[];initial_hash=None
    def tensor_hash(model):
        h=hashlib.sha256()
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    for name,cls in [('bounded_F',BoundedConditionAdapter),('Euclidean_F',EuclideanConditionControl)]:
        torch.manual_seed(1729);model=cls().cuda();assert sum(v.numel() for v in model.parameters())==525568
        current=tensor_hash(model)
        if initial_hash is None:initial_hash=current
        assert current==initial_hash
        c,w,b,y=[batch[k] for k in ('condition','weight','bias','target')]
        changed,_=model(c,w,b);assert torch.equal(changed,c)
        disabled,_=model(c,w,b,enabled=False);assert torch.equal(disabled,c)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        losses=[];gradients=[];tick=time.monotonic()
        for step in range(4):
            sl=slice(step*16,(step+1)*16);optimizer.zero_grad(set_to_none=True)
            changed,_=model(c[sl],w[sl],b[sl]);loss=.1*(1-F.cosine_similarity(changed.float(),y[sl].float(),dim=-1)).mean()
            assert torch.isfinite(loss);loss.backward();g=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            assert model.direction.weight.grad is not None and model.direction.weight.grad.norm()>0
            optimizer.step();losses.append(float(loss));gradients.append(float(g))
        torch.cuda.synchronize();seconds=time.monotonic()-tick
        assert {int(v['step']) for v in optimizer.state.values()}=={4}
        with torch.no_grad():observed=model(c,w,b)[0].cpu()
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        target=cls().double().eval();target.load_state_dict({k:v.double() for k,v in state.items()},strict=True)
        with torch.no_grad():expected=target(cpu['condition'].double(),cpu['weight'].double(),cpu['bias'].double())[0]
        maximum=float((observed.double()-expected).abs().max())
        assert torch.allclose(observed.double(),expected,atol=1e-5,rtol=5e-6)
        buffer=io.BytesIO();torch.save(dict(model=state,optimizer=optimizer.state_dict(),CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all()),buffer)
        buffer.seek(0);restored=torch.load(buffer,weights_only=True,map_location='cpu')
        assert all(torch.equal(restored['model'][k],v) for k,v in state.items())
        assert {int(v['step']) for v in restored['optimizer']['state'].values()}=={4}
        assert torch.equal(restored['CPU_RNG'],torch.get_rng_state())
        actual_rng=torch.cuda.get_rng_state_all();assert all(torch.equal(a,b) for a,b in zip(restored['CUDA_RNG'],actual_rng))
        results.append(dict(arm=name,temporary_updates=4,EEG_exposures=64,losses=losses,gradient_norms=gradients,
            initial_zero_head_and_disabled_native_bit_exact=True,same_525568_initial_tensor_SHA256=current,
            same_formula_CPU_FP64_max_absolute_error=maximum,serialization_model_Adam_RNG_bytes_exact=True,
            future_GPU_trajectory_bit_exactness_tested=False,learned_weights_saved=False,training_seconds=seconds))
        del model,optimizer,target,state,restored,buffer;gc.collect();torch.cuda.empty_cache()
    assert time.monotonic()-start<820 and torch.cuda.max_memory_allocated()<20<<30
    result=dict(status='PASS_ACTUAL_FULL1654_F_GEOMETRY_ENGINEERING_ONLY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(path),
        input_bank=s['input_bank'],consumer_review=s['consumer_review'],input_subset=indices.tolist(),arms=results,
        conditions_from_all165400_verified_bank=True,UNet_calls=0,learned_weights_saved=False,research_images=0,
        final_fits_admitted=False,official_test_payloads_opened=False,NEW_contacted=False,peak_GPU_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start)
    output=Path(s['profile_output']);assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],review=ref(output),seconds=result['seconds'],temporary_updates=8,final_fits_admitted=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();run(a.protocol)
