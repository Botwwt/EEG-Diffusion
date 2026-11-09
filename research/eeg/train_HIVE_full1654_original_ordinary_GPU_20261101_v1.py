"""Future admitted original ordinary3pass fit; immutable planned segment states.

This entry cannot run with the existing engineering-only protocol. Complete
generation/evaluation sources and the actual successful host profile are gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1 as F
E=F.E;K=F.K


def run(protocol,segment):
    s=F.setup_fit(protocol)
    import numpy as np
    import torch
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from profile_HIVE_full1654_original_ordinary_GPU_20261101_v1 import exact
    assert not torch.cuda.is_initialized() and 0<=segment<128
    inputs,latents,binding=E.verified_CPU_inputs(s)
    K.require_wrapper();initial_resources=K.resources(s,starting=True)
    root=Path(s['output_root'])/'ordinary_full1654'
    if segment==0:assert not root.exists();root.mkdir(parents=True)
    else:assert root.exists() and not (root/'fit_completion.json').exists()
    part=root/'segments'/f'{segment:03d}';assert not part.exists();part.mkdir(parents=True)
    started,base=time.monotonic(),K.rss()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    def guard():
        assert time.monotonic()-started<840 and K.rss()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
        K.resources(s)
    try:
        pipe,A,P,prompts,ddpm,host,loader=E.load_host(s,part,torch)
        data=E.make_GPU_inputs(inputs,latents,torch)
        modules=dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P)
        frozen={key:E.module_digest(module) for key,module in modules.items()}
        order=E.orders(torch);order_sha=[hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in order]
        provenance=dict(protocol=K.ref(protocol),input_binding=binding,orders_sha256=order_sha,method=E.METHOD,
                        frozen_host_state_hashes=frozen,host=host,old_Dfit_now_train_not_calibration=True)
        if segment==0:K.write(root/'input_provenance.json',provenance)
        else:assert K.read(root/'input_provenance.json')==provenance
        torch.manual_seed(1729);model=BoundedConditionAdapter(predict_risk=False).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        rng=torch.Generator(device='cuda').manual_seed(E.stable_seed('shared_training_noise'))
        updates=exposures=0;loss_sum=0.;coverage=torch.zeros(3,165400,dtype=torch.bool);old_checkpoints=[]
        if segment:
            saved,pointer=F.load_checkpoint(root,torch)
            assert saved['segment']==segment-1 and saved['provenance']==K.ref(root/'input_provenance.json')
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            rng.set_state(saved['noise_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            assert exact(E.clone_CPU(model.state_dict(),torch),saved['model'],torch)
            assert exact(E.clone_CPU(optimizer.state_dict(),torch),saved['optimizer'],torch)
            assert torch.equal(rng.get_state(),saved['noise_rng']) and torch.equal(torch.get_rng_state(),saved['torch_rng'])
            assert exact(torch.cuda.get_rng_state_all(),saved['cuda_rng'],torch)
            updates=saved['updates'];exposures=saved['exposures'];loss_sum=saved['loss_sum'];coverage=saved['coverage']
            old_checkpoints=saved['checkpoint_history']+[pointer]
            assert torch.equal(coverage,F.expected_coverage(updates,order,torch))
            K.write(part/'actual_restore_before_next_GPU_step.json',dict(status='EXACT_SERIALIZED_STATE_RESTORED_BEFORE_NEXT_STEP',
                    checkpoint=pointer,model_optimizer_RNG_and_coverage_exact=True,future_GPU_bit_exactness_proved=False))
        start_update=updates;last_snapshot=None
        log=part/'training.jsonl'
        def snapshot():
            nonlocal last_snapshot
            if last_snapshot is not None and last_snapshot['updates']==updates:return last_snapshot
            K.resources(s,extra_bytes=8<<20)
            state=dict(model=E.clone_CPU(model.state_dict(),torch),optimizer=E.clone_CPU(optimizer.state_dict(),torch),
                noise_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                coverage=coverage.clone(),updates=updates,exposures=exposures,loss_sum=loss_sum,segment=segment,
                provenance=K.ref(root/'input_provenance.json'),orders_sha256=order_sha,
                checkpoint_history=old_checkpoints.copy(),log_prefixes=[K.ref(log)])
            last_snapshot=F.save_checkpoint(root,state,torch);old_checkpoints.append(last_snapshot)
            return last_snapshot
        fit_started=time.monotonic()
        with log.open('x',buffering=1) as stream:
            while updates<31014:
                if time.monotonic()-started>=s['segment_soft_boundary_seconds']:break
                epoch,batch,_=F.prefix_counts(updates);offset=batch*16
                indices=order[epoch][offset:offset+16]
                assert not coverage[epoch,indices].any()
                result=E.ordinary_step(model,optimizer,pipe,A,P,prompts,ddpm,data,indices,rng,torch)
                coverage[epoch,indices]=True;updates+=1;exposures+=len(indices);loss_sum+=result['loss']*len(indices)
                if updates%64==0 or updates%10338==0:
                    event=dict(updates=updates,exposures=exposures,pass_number=epoch+1,
                        cumulative_mean_loss=loss_sum/exposures,seconds_this_segment=time.monotonic()-started,**result)
                    stream.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True);guard()
                if updates==128 or updates%512==0 or updates%10338==0:snapshot()
            assert updates>start_update
            pointer=snapshot()
        torch.cuda.synchronize();fit_seconds=time.monotonic()-fit_started
        assert exposures==F.prefix_counts(updates)[2] and torch.equal(coverage,F.expected_coverage(updates,order,torch))
        assert {int(value['step']) for value in optimizer.state.values()}=={updates}
        assert {key:E.module_digest(module) for key,module in modules.items()}==frozen
        K.check(s['old_ordinary']);guard()
        result=dict(status='COMPLETE_PLANNED_ORDINARY_TRAIN_SEGMENT',protocol=K.ref(protocol),segment=segment,
                    start_updates=start_update,updates=updates,exposures=exposures,checkpoint=pointer,
                    fit_seconds=fit_seconds,program_seconds=time.monotonic()-started,loader=loader,
                    peak_GPU_bytes=torch.cuda.max_memory_allocated(),maximum_extra_CPU_bytes=K.rss()-base,
                    initial_resources=initial_resources,future_GPU_bit_exactness_claimed=False,R80_payloads_opened=False)
        if updates==31014:
            assert coverage.all() and exposures==496200
            final=root/'component_final.pt'
            K.resources(s,extra_bytes=3<<20)
            E.save_tensor(final,dict(model=E.clone_CPU(model.state_dict(),torch),arm='ordinary_full1654',seed=1729,
                strength=1.,radius=.25,updates=updates,exposures=exposures,passes=3,method=E.METHOD,
                protocol=K.ref(protocol),provenance=K.ref(root/'input_provenance.json')),torch)
            K.resources(s,extra_bytes=165400*1024*4+4096)
            output=np.lib.format.open_memmap(root/'actual_train_outputs.npy',mode='w+',dtype=np.float32,shape=(165400,1024))
            with torch.inference_mode():
                for lo in range(0,165400,256):
                    hi=min(lo+256,165400);aid=data['affine_ids'][lo:hi]
                    output[lo:hi]=model(data['condition'][lo:hi],data['weight'][aid],data['bias'][aid])[0].cpu().numpy()
            output.flush();del output;guard()
            K.write(root/'fit_completion.json',dict(status='COMPLETE_FULL1654_ORIGINAL_ORDINARY_THREE_PASSES_FIXED_FINAL',
                protocol=K.ref(protocol),component=K.ref(final),latest_checkpoint=pointer,
                actual_train_outputs=K.ref(root/'actual_train_outputs.npy'),updates=updates,exposures=exposures,
                parameters=525568,seed=1729,strength=1.,radius=.25,final_segment=segment,old_Dfit_calibration_performed=False,
                R80_payloads_opened=False,future_GPU_bit_exactness_claimed=False,independent_H=False))
        K.write(part/'completion.json',result)
        print(json.dumps(result))
    except BaseException as error:
        K.write(part/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--segment',type=int,required=True)
    args=parser.parse_args();run(args.protocol,args.segment)
