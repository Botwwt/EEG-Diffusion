"""Four disposable real original-objective updates; never a formal learned head."""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback
import brainhive_full1654_ordinary_engine_20261101_v1 as E
K=E.K


def exact(left,right,torch):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(exact(left[key],right[key],torch) for key in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(exact(a,b,torch) for a,b in zip(left,right))
    return left==right


def run(protocol):
    s=E.setup_profile(protocol)
    import numpy as np
    import torch
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    assert not torch.cuda.is_initialized()
    inputs,latents,binding=E.verified_CPU_inputs(s)
    K.require_wrapper();initial=K.resources(s,starting=True)
    parent=Path(s['output_root']);parent.mkdir(exist_ok=True)
    root=parent/'profile';assert not root.exists();root.mkdir()
    started,base=time.monotonic(),K.rss()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    def guard():
        assert time.monotonic()-started<840 and K.rss()-base<10<<30
        assert torch.cuda.max_memory_allocated()<20<<30
        K.resources(s)
    try:
        pipe,A,P,prompts,ddpm,host,loader=E.load_host(s,root,torch)
        data=E.make_GPU_inputs(inputs,latents,torch);del inputs,latents;gc.collect()
        modules=dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P)
        before={key:E.module_digest(module) for key,module in modules.items()}
        torch.manual_seed(1729)
        model=BoundedConditionAdapter(predict_risk=False).cuda()
        assert sum(p.numel() for p in model.parameters())==525568
        initial_model=E.clone_CPU(model.state_dict(),torch)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        rng=torch.Generator(device='cuda').manual_seed(E.stable_seed('shared_training_noise'))
        order=E.orders(torch)
        records=[]
        # All20 source charts are exercised with the exact native coordinates.
        with torch.no_grad():
            for aid in range(20):
                index=int(torch.nonzero(data['affine_ids']==aid)[0])
                c=data['condition'][index:index+1];weight=data['weight'][aid:aid+1];bias=data['bias'][aid:aid+1]
                assert torch.equal(model(c,weight,bias)[0],c)
                assert torch.equal(model(c,weight,bias,enabled=False)[0],c)
        steps_started=time.monotonic()
        for step in range(4):
            guard();indices=order[0][16*step:16*(step+1)]
            # Reuse the previous immutable CPU snapshot in the trace so torch
            # stores the same before/after transition only once, within64MiB.
            prior=initial_model if step==0 else records[-1]['model_after']
            prior_optimizer=E.clone_CPU(optimizer.state_dict(),torch) if step==0 else records[-1]['optimizer_after']
            result=E.ordinary_step(model,optimizer,pipe,A,P,prompts,ddpm,data,indices,rng,torch,
                                   trace=True,baseline=step==0)
            result.update(model_before=prior,optimizer_before=prior_optimizer,
                          model_after=E.clone_CPU(model.state_dict(),torch),
                          optimizer_after=E.clone_CPU(optimizer.state_dict(),torch))
            records.append(result)
        torch.cuda.synchronize();step_seconds=time.monotonic()-steps_started
        assert {int(value['step']) for value in optimizer.state.values()}=={4}
        assert {key:E.module_digest(module) for key,module in modules.items()}==before
        assert any(not torch.equal(initial_model[key],value.cpu()) for key,value in model.state_dict().items())
        state=dict(model=E.clone_CPU(model.state_dict(),torch),optimizer=E.clone_CPU(optimizer.state_dict(),torch),
                   noise_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                   temporary_updates=4,formal_training_use_forbidden=True,protocol=K.ref(protocol))
        E.save_tensor(root/'engineering_state_NOT_FOR_FORMAL.pt',state,torch)
        saved=torch.load(root/'engineering_state_NOT_FOR_FORMAL.pt',weights_only=True,map_location='cpu')
        assert exact(state,saved,torch)
        restored=BoundedConditionAdapter(predict_risk=False).cuda();restored.load_state_dict(saved['model'],strict=True)
        restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=.01)
        restored_optimizer.load_state_dict(saved['optimizer'])
        restored_rng=torch.Generator(device='cuda');restored_rng.set_state(saved['noise_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        assert exact(E.clone_CPU(restored.state_dict(),torch),saved['model'],torch)
        assert exact(E.clone_CPU(restored_optimizer.state_dict(),torch),saved['optimizer'],torch)
        assert torch.equal(restored_rng.get_state(),saved['noise_rng'])
        assert torch.equal(torch.get_rng_state(),saved['torch_rng'])
        assert exact(torch.cuda.get_rng_state_all(),saved['cuda_rng'],torch)
        # Deliberately no subsequent GPU step/equality or future-exactness claim.
        E.save_tensor(root/'actual_four_step_trace.pt',records,torch)
        K.write(root/'actual_input_binding.json',binding)
        K.check(s['old_ordinary'])
        guard()
        result=dict(status='PASS_ACTUAL_FULL1654_ORIGINAL_ORDINARY_ENGINEERING_ONLY',protocol=K.ref(protocol),
                    temporary_updates=4,temporary_EEG_exposures=64,parameters=525568,
                    actual_all20_zero_and_disabled_source_chart_checks=True,
                    original_epsilon_plus_token_objective=True,source_target_latent_binding=K.ref(root/'actual_input_binding.json'),
                    model_optimizer_and_RNG_serialization_restore_exact=True,
                    future_GPU_bit_exactness_NOT_proved=True,uninterrupted_GPU_determinism_NOT_proved=True,
                    existing_old_future_bit_exact_failures_preserved=True,formal_learned_checkpoint_saved=False,
                    engineering_state=K.ref(root/'engineering_state_NOT_FOR_FORMAL.pt'),trace=K.ref(root/'actual_four_step_trace.pt'),
                    loader=loader,host=host,frozen_host_state_hashes_before_and_after=before,
                    four_update_training_seconds=step_seconds,seconds=time.monotonic()-started,
                    peak_GPU_bytes=torch.cuda.max_memory_allocated(),maximum_extra_CPU_bytes=K.rss()-base,
                    initial_resources=initial,new_research_images=0,R80_payloads_opened=False,
                    formal_training_admitted=False,benchmark_admitted=False,independent_H=False,NEW_contacted=False)
        K.write(root/'completion.json',result)
        print(json.dumps({k:result[k] for k in ('status','temporary_updates','four_update_training_seconds','peak_GPU_bytes','maximum_extra_CPU_bytes')}))
    except BaseException as error:
        K.write(root/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    run(parser.parse_args().protocol)
