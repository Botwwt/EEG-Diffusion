"""D20 engineering: exact native paths, actual pair gradients and throughput.

Uses four fixed T queries, no effect gate or fitted candidate checkpoint.
Tiny temporary changes are discarded; full training is registered separately.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_error_pair_lora_20261027_v2 as V
import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N
from brainhive_bounded_component_v1 import BoundedConditionAdapter

def fingerprint(parameters):
    h=hashlib.sha256()
    for name,p in parameters:
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def ordinary_conditions(records,conditions,contract,checkpoint):
    model=BoundedConditionAdapter();saved=torch.load(U.checked(checkpoint),weights_only=False,map_location='cpu')
    assert saved['arm']=='ordinary' and saved['calibration']['selected_strength']==1.
    model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
    subject=next(s for s in contract['subjects'] if s['subject']=='sub-01');affines={}
    for source in subject['sources']:
        if source['role'] not in ('oof0','oof1'):continue
        m=torch.load(U.checked({'path':source['checkpoint_path'],'sha256':source['checkpoint_sha256']}),weights_only=True,map_location='cpu')['model']
        affines[int(source['role'][-1])]=(m['block.layers.0.norm.weight'],m['block.layers.0.norm.bias'])
    assert set(affines)=={0,1};out=[]
    for i,q in enumerate(records):
        w,b=affines[q['fold']]
        with torch.no_grad():out.append(model(torch.from_numpy(np.array(conditions[i:i+1],copy=True)),w[None],b[None])[0])
    return torch.cat(out)

def main():
    p=argparse.ArgumentParser()
    for key in ('spec','native-registration','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    started=time.monotonic();assert a.max_seconds==300 and not a.output.exists();a.output.mkdir(parents=True)
    stage='INPUTS';stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    def log(x):
        x=x|{'seconds':time.monotonic()-started}
        with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(x),flush=True)
    try:
        spec=U.read(a.spec)
        for r in spec['source_pins']:U.checked(r)
        old=U.read(U.checked(spec['D19_fit']));prep=U.read(U.checked(old['prepared_inputs']));mapping=U.read(U.checked(prep['mapping']))
        lat=np.load(U.checked(old['encoded_latents']['T400']),allow_pickle=False,mmap_mode='r');assert lat.shape==(400,2,4,64,64)
        cond=np.load(U.checked(prep['conditions']['T400']),allow_pickle=False)
        contract=U.read(U.checked(spec['native_input_contract']));ordinary=ordinary_conditions(mapping['T400'],cond,contract,spec['ordinary_checkpoint'])
        U.save_array(a.output/'all400_original_ordinary_conditions.npy',ordinary.numpy())
        import brainhive_paired_study_contract_20261025_v1 as C
        config=C.read(a.native_registration);C.validate(config);C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch
        doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=300,maximum_output_bytes=160<<20,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():assert not stopped[0] and time.monotonic()-started<270,'finite D20 profile';budget.check()
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);C.resources(config,training_start=True)
        from brainhive_component_generator_v1 import load_native,denoise
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        stage='STRICT_NATIVE_LOAD';folder=a.output/'loader';folder.mkdir();observer=ConvertedWeightAudit(torch,folder);observer.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:observer.restore()
        loader=observer.complete();check()
        original=list(pipe.unet.named_parameters());before=fingerprint(original)
        with torch.no_grad():native_tokens=projector(adapter(torch.from_numpy(cond[:1]).cuda().half())).detach()
        stage='ORIGINAL_NATIVE_T_PIXEL_PARITY';q=mapping['T400'][0]
        img,obs=N.sample_observed(pipe,prompts,native_tokens,q['image_seed'],torch);saved=N.save_image(a.output/'native_T_pixel_parity.webp',img)
        assert saved['generated_sha256']==q['generated']['sha256'] and saved['generated_pixel_sha256']==q['generated_pixel_sha256']
        stage='ZERO_LORA_NATIVE_PIXEL_PARITY';torch.manual_seed(1729);layers=V.install(pipe.unet);params=V.new_parameters(layers)
        initial=V.new_state(layers);assert all(not p.requires_grad for _,p in original)
        img,zobs=N.sample_observed(pipe,prompts,native_tokens,q['image_seed'],torch);zero=N.save_image(a.output/'zero_LoRA_T_pixel_parity.webp',img)
        for k in ['generated_sha256','generated_pixel_sha256']:assert zero[k]==saved[k]
        assert zobs==obs
        U.atomic(a.output/'pixel_parities.json',dict(reference=q['generated'],native=saved,zero_LoRA=zero,noise=obs,original_pipeline_and_zero_LoRA_exact=True))
        check();pipe.unet.train();pipe.unet.enable_gradient_checkpointing();assert float(pipe.unet.config.dropout)==0
        from diffusers import DDPMScheduler
        scheduler=DDPMScheduler.from_config(pipe.scheduler.config);assert scheduler.config.prediction_type=='epsilon'
        chosen=[0,10,20,30];x=torch.from_numpy(np.array(lat[chosen],copy=True)).cuda();c=ordinary[chosen].cuda()
        with torch.no_grad():tokens=projector(adapter(c.half())).detach()
        generator=torch.Generator(device='cuda').manual_seed(201729);events=[];precision_probe=None
        stage='ACTUAL_PAIRED_POSITIVE_AND_WRONG_NEGATIVE_GRADIENTS'
        for j,arm in enumerate(V.ARMS):
            V.load_state(layers,initial);optim=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.99),weight_decay=.01)
            for step,timestep in enumerate([999,749,499,249]):
                check();ix=slice(0,2);n=2;positive=x[ix,0];negative=x[ix,1]
                if arm=='sameconcept_wrong_error':
                    # Wrong generated image from the next stimulus within the
                    # same concept, not a class-label or c permutation.
                    negative=torch.from_numpy(np.array(lat[[1,11],1],copy=True)).cuda()
                noise=torch.randn(positive.shape,generator=generator,device='cuda');eps=torch.cat([noise,noise]);t=torch.full((2*n,),timestep,device='cuda',dtype=torch.long)
                noisy=scheduler.add_noise(torch.cat([positive,negative]),eps,t);tok=torch.cat([tokens[ix],tokens[ix]])
                V.enabled(layers,False)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):reference=denoise(pipe,prompts,noisy,t,tok).detach()
                V.enabled(layers,True);optim.zero_grad(set_to_none=True)
                if arm=='positive_only' and step==0:
                    for _,m in layers:m.legacy_fp16_cast=True
                    with torch.autocast('cuda',dtype=torch.bfloat16):legacy=denoise(pipe,prompts,noisy,t,tok);legacy_loss,_=V.objective(legacy,reference,eps,arm)
                    legacy_loss.backward()
                    precision_probe={'same_actual_batch_noise_parameters_and_loss':True,'legacy_FP16_zero_gradient_modules':[name for name,m in layers if m.up.grad is not None and float(m.up.grad.norm())==0.],
                        'legacy_FP16_total_gradient_norm':float(torch.stack([m.up.grad.norm() for _,m in layers]).norm())}
                    optim.zero_grad(set_to_none=True)
                    for _,m in layers:m.legacy_fp16_cast=False
                torch.cuda.synchronize();begin=time.monotonic()
                with torch.autocast('cuda',dtype=torch.bfloat16):student=denoise(pipe,prompts,noisy,t,tok);loss,terms=V.objective(student,reference,eps,arm)
                if step==0:assert torch.equal(student,reference)
                assert torch.isfinite(loss);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                assert grad>0 and all(p.grad is None for _,p in original)
                connected=sum(int(m.up.grad is not None and torch.isfinite(m.up.grad).all()) for _,m in layers);assert connected==140
                nonzero=sum(int(m.up.grad.norm()>0) for _,m in layers)
                if arm=='positive_only' and step==0:
                    precision_probe.update(direct_BF16_zero_gradient_modules=[name for name,m in layers if float(m.up.grad.norm())==0.],direct_BF16_total_gradient_norm=float(torch.stack([m.up.grad.norm() for _,m in layers]).norm()))
                    U.atomic(a.output/'actual_FP16_vs_direct_BF16_gradient_probe.json',precision_probe)
                optim.step();max_ratio=max(m.project() for _,m in layers);torch.cuda.synchronize()
                event=dict(arm=arm,step=step+1,timestep=timestep,pairs=n,UNet_batch=2*n,student_forward_backward_update_seconds=time.monotonic()-begin,gradient_norm=float(grad),nonzero_up_gradients=nonzero,max_relative_projection_change=max_ratio,**{k:float(v.detach()) for k,v in terms.items()});events.append(event);log(event)
        V.load_state(layers,initial);V.enabled(layers,False);after=fingerprint(original);assert before==after
        assert all(p.grad is None for _,p in original)
        U.atomic(a.output/'completion.json',dict(status='D20_NATIVE_IP_LOW_RANK_ERROR_PAIR_GRADIENTS_PIXEL_PARITY_AND_THROUGHPUT_PROFILE_COMPLETE',source=U.ref(__file__),spec=U.ref(a.spec),wrapper=wrapper,loader=loader,host=host,
            projection_modules=len(layers),rank=V.RANK,new_parameters=sum(p.numel() for p in params),original_UNet_parameter_sha256=before,frozen_weights_unchanged=before==after,
            ordinary_condition_bank=U.ref(a.output/'all400_original_ordinary_conditions.npy'),ordinary_checkpoint=spec['ordinary_checkpoint'],old_EEG_and_ordinary_updated=False,
            profile_queries=chosen,temporary_updates_per_arm=4,temporary_optimization_is_not_formal_training=True,temporary_states_discarded=True,
            native_and_zero_pixel_parity=True,all140_IP_modules_have_finite_connected_gradients=True,nonzero_module_counts=[e['nonzero_up_gradients'] for e in events],gradient_events=events,precision_probe=precision_probe,
            true_source='D19 original T400 image files commonVAE-mode',negative_source='same-query original native parent generation; not ordinary generation',
            wrong_negative='same concept next-stimulus native generation, positive/c fixed',normal_inference_target_access=False,Dcheck_targets_read=False,
            exact_Turbo_likelihood_or_posterior=False,new_effect_images=0,formal_D_resumed=False,new_sampling_steps_at_inference=0,
            seconds=time.monotonic()-started,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=torch.cuda.max_memory_allocated(),SOTA_established=False))
        log({'stage':'COMPLETE_ENGINEERING_ONLY'})
    except BaseException as e:
        U.atomic(a.output/'failure.json',dict(status='FAILED_D20_ENGINEERING_PROFILE_PRESERVED',stage=stage,error=repr(e),seconds=time.monotonic()-started,scientific_negative=False,formal_D_resumed=False));traceback.print_exc();raise

if __name__=='__main__':main()
