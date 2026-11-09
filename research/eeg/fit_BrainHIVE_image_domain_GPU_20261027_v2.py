"""D19 three matched image-domain heads and all400 bounded endpoint edits."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,gc,hashlib,io,json,signal,time,traceback,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
import torch.nn.functional as F
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_image_domain_20261027_v1 as V
import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N

def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return x

def state_sha(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load_vae(output,expected_host):
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    folder=output/'loader';folder.mkdir();audit=ConvertedWeightAudit(torch,folder);audit.install()
    try:pipe,adapter,projector,prompts,host=load_native(training=False)
    finally:audit.restore()
    proof=audit.complete();assert host==expected_host
    initial=state_sha(pipe.vae.state_dict());pipe.vae.to(dtype=torch.float32)
    assert state_sha({k:v.to(torch.float16) for k,v in pipe.vae.state_dict().items()})==initial
    calls=[0];pipe.unet.register_forward_pre_hook(lambda *_:calls.__setitem__(0,calls[0]+1))
    # Only VAE encoding/decoding is used in this task. Release GPU-only objects,
    # preserving their source weight files and the strict original load proof.
    pipe.unet=None;pipe.text_encoder=None;pipe.text_encoder_2=None
    del adapter,projector,prompts;gc.collect();torch.cuda.empty_cache()
    pipe.vae.enable_slicing()
    assert pipe.unet is None and pipe.vae.use_slicing is True
    return pipe,proof,calls,dict(original_FP16_VAE_state_sha256=initial,widening_FP32_roundtrips_exactly=True,scale=float(pipe.vae.config.scaling_factor),
        unused_UNet_and_text_modules_released_after_exact_load=True,VAE_batch_slicing=True)

def image_tensor(payload):
    with Image.open(io.BytesIO(payload)) as opened:
        image=ImageOps.fit(opened.convert('RGB'),(512,512),method=Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/127.5-1.

def encode_records(pipe,records,real,check):
    results=[];containers={}
    try:
        for lo in range(0,len(records),8):
            check();batch=[]
            for row in records[lo:lo+8]:
                if real:
                    item=row['target'];name=item['image_zip']
                    if name not in containers:containers[name]=zipfile.ZipFile(name)
                    payload=containers[name].read(item['image_member']);assert hashlib.sha256(payload).hexdigest()==item['image_sha256']
                else:
                    path=U.checked(row['generated']);payload=path.read_bytes()
                batch.append(image_tensor(payload))
            with torch.no_grad():latent=pipe.vae.encode(torch.stack(batch).cuda()).latent_dist.mode()*pipe.vae.config.scaling_factor
            assert latent.dtype==torch.float32 and torch.isfinite(latent).all();results.append(latent)
    finally:
        for archive in containers.values():archive.close()
    return torch.cat(results)

def decode(pipe,z):
    with torch.no_grad():
        image=pipe.vae.decode(z/pipe.vae.config.scaling_factor,return_dict=False)[0]
        assert torch.isfinite(image).all();return pipe.image_processor.postprocess(image,output_type='pil')

def main():
    p=argparse.ArgumentParser()
    for key in ('spec','native-registration','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    invoked=time.monotonic();assert a.max_seconds==420 and not a.output.exists();a.output.mkdir(parents=True)
    stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));active='INPUTS';models={};optimizers={};epoch=-1;updates=0;exposures=0
    def log(value):
        value=value|dict(seconds=time.monotonic()-invoked)
        with (a.output/'training_log.jsonl').open('a') as f:f.write(json.dumps(value)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(value),flush=True)
    try:
        spec=U.read(a.spec);assert spec['schema']==V.SCHEMA and spec['epochs']==20 and spec['radius']==V.RADIUS
        for ref in spec['source_pins']:U.checked(ref)
        prepared=U.read(U.checked(spec['prepared_inputs']));mapping=U.read(U.checked(prepared['mapping']))
        d09=U.read(U.checked(spec['D09_completion']));inputs=U.read(U.checked(d09['inputs']))
        import brainhive_paired_study_contract_20261025_v1 as C
        config=C.read(a.native_registration);C.validate(config);C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=420,maximum_output_bytes=768<<20,maximum_extra_CPU_bytes=10<<30,
            maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():assert not stopped[0] and time.monotonic()-invoked<390,'finite D19 fit/export budget';budget.check()
        torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(1729);torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);C.resources(config,training_start=True)
        active='UNCHANGED_NATIVE_VAE_LOAD_AND_COMMON_REAL_GENERATED_ENCODING';pipe,loader,calls,vae_proof=load_vae(a.output,inputs['expected_host'])
        encoded={};encode_started=time.monotonic()
        for name,records in mapping.items():
            encoded[name]=torch.stack([encode_records(pipe,records,True,check),encode_records(pipe,records,False,check)],1) if name!='Dcheck400' else encode_records(pipe,records,False,check)
            U.save_array(a.output/(name+'_latents.npy'),encoded[name].cpu().numpy());log(dict(stage='COMPLETE_COMMON_IMAGE_ENCODING',partition=name,shape=list(encoded[name].shape)))
        encoding_seconds=time.monotonic()-encode_started
        cond={name:torch.from_numpy(np.load(U.checked(item))).cuda() for name,item in prepared['conditions'].items()}
        mean=cond['T400'].mean(0);scale=(cond['T400']-mean).square().mean().sqrt();assert scale>.01
        normalized={name:(value-mean)/scale for name,value in cond.items()};permutation=torch.from_numpy(V.wrong_positions(mapping['T400'])).cuda()
        fixed_sigma=torch.tensor(V.SIGMAS,device='cuda')[None,:,None].expand(200,4,2).reshape(-1)
        held_generator=torch.Generator(device='cuda').manual_seed(171729);held_noise=torch.randn((200,4,64,64),device='cuda',generator=held_generator)
        held_samples=V.corrupt(encoded['Dfit200'][:,0],encoded['Dfit200'][:,1],held_noise).reshape(-1,4,64,64)
        held_c=normalized['Dfit200'][:,None,None].expand(200,4,2,1024).reshape(-1,1024)
        torch.manual_seed(1729);base=V.DomainHead();initial_sha=state_sha(base.state_dict())
        for arm in V.FIELDS:
            models[arm]=copy.deepcopy(base).cuda();assert state_sha(models[arm].state_dict())==initial_sha
            optimizers[arm]=torch.optim.AdamW(models[arm].parameters(),lr=.0003,betas=(.9,.99),weight_decay=.01)
        del base
        U.atomic(a.output/'provenance.json',dict(schema=V.SCHEMA,spec=U.ref(a.spec),prepared_inputs=spec['prepared_inputs'],source=U.ref(__file__),wrapper=wrapper,
            all3_initial_states_identical=True,initial_state_sha256=initial_sha,parameters_per_field=sum(p.numel() for p in models[V.FIELDS[0]].parameters()),
            loader_proof=loader,VAE=vae_proof,all_inputs='same512center-cropRGB commonVAE-mode, latent scaling identical for real and generated image files',
            condition_mean=mean.cpu().tolist(),condition_scalar_RMS=float(scale),sigma_grid=V.SIGMAS,class_prior=.5,
            training_T_queries=400,training_T_concepts=40,subjects=1,not_complete_all_T=True,Dfit_used_only_for_logs=True,
            Dcheck_targets_loaded=False,normal_inference_target_access=False,diffusion_score_or_posterior_claim=False,original_D_resumed=False))
        def held(at_epoch,save=False):
            values={};all_logits=[]
            with torch.no_grad():
                for arm,model in models.items():
                    model.eval();cc=torch.zeros_like(held_c) if arm=='unconditional' else held_c
                    logits=torch.cat([model(held_samples[lo:lo+128],fixed_sigma[lo:lo+128],cc[lo:lo+128]).float() for lo in range(0,1600,128)]).reshape(200,4,2)
                    metrics=V.proper(logits);values[arm]={key:value.mean(0).cpu().tolist() for key,value in metrics.items()};all_logits.append(logits.cpu())
            if save:U.save_array(a.output/'Dfit_logits.npy',torch.stack(all_logits,1).numpy())
            log(dict(stage='FIXED_DFIT200_FOUR_SIGMA_PROPER_LOG_ONLY',epoch=at_epoch,metrics_by_sigma=values,correct_c_for_both_conditioned_models=True))
        held(0);active='THREE_MATCHED_FULL_T400_DOMAIN_HEADS';fit_started=time.monotonic();generator=torch.Generator(device='cuda').manual_seed(1729)
        for epoch in range(20):
            order=torch.randperm(400,device='cuda',generator=generator);coverage=np.zeros(400,np.int8);sums={arm:0. for arm in V.FIELDS}
            for lo in range(0,400,32):
                check();ix=order[lo:lo+32];n=len(ix);coverage[ix.cpu().numpy()]+=1
                noise=torch.randn((n,4,64,64),device='cuda',generator=generator);samples=V.corrupt(encoded['T400'][ix,0],encoded['T400'][ix,1],noise).reshape(-1,4,64,64)
                sigma=torch.tensor(V.SIGMAS,device='cuda')[None,:,None].expand(n,4,2).reshape(-1);labels=torch.tensor([1.,0.],device='cuda').repeat(n*4)
                for arm,model in models.items():
                    model.train();optimizers[arm].zero_grad(set_to_none=True)
                    cc=normalized['T400'][permutation[ix] if arm=='wrong_condition' else ix]
                    if arm=='unconditional':cc=torch.zeros_like(cc)
                    cc=cc[:,None,None].expand(n,4,2,1024).reshape(-1,1024)
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(samples,sigma,cc).float();loss=F.binary_cross_entropy_with_logits(logits,labels)
                    assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    if updates==16:
                        grad=model.image[0].weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
                        U.atomic(a.output/(arm+'_actual_image_branch_gradient.json'),dict(stage='ACTUAL_UPDATE17',image_gradient_finite_nonzero=True,norm=float(grad.norm()),global_norm=float(norm)))
                    optimizers[arm].step();sums[arm]+=float(loss.detach())*n
                updates+=1;exposures+=n
            assert np.all(coverage==1) and updates==13*(epoch+1) and exposures==400*(epoch+1)
            log(dict(stage='COMPLETE_T400_PASS',epoch=epoch+1,updates_per_head=updates,query_exposures_per_head=exposures,full_coverage_min=1,full_coverage_max=1,
                order_sha256=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest(),balanced_BCE={arm:value/400 for arm,value in sums.items()}));held(epoch+1,save=epoch==19)
        training_seconds=time.monotonic()-fit_started;assert updates==260 and exposures==8000
        for model in models.values():model.eval().requires_grad_(False)
        checkpoint=a.output/'fixed_final_domain_heads.pt';torch.save(dict(models={arm:cpu(x.state_dict()) for arm,x in models.items()},optimizers={arm:cpu(x.state_dict()) for arm,x in optimizers.items()},
            condition_mean=mean.cpu(),condition_scale=scale.cpu(),epoch=20,updates_per_head=260,query_exposures_per_head=8000,CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),spec=U.ref(a.spec)),checkpoint)
        active='ALL400_FINAL_ENDPOINT_EDITS_NO_TARGET_ACCESS';source=encoded['Dcheck400'];bank=[source.cpu()];diagnostics={}
        for arm,model in models.items():
            parts=[];changes=[];before=[];after=[];grad_norm=[]
            for lo in range(0,400,32):
                check();z=source[lo:lo+32];cc=normalized['Dcheck400'][lo:lo+32]
                if arm=='unconditional':cc=torch.zeros_like(cc)
                zero=torch.zeros(len(z),device='cuda')
                with torch.no_grad():before.extend(model(z,zero,cc).cpu().tolist())
                leaf=z.detach().requires_grad_(True);g=torch.autograd.grad(model(leaf,zero,cc).sum(),leaf)[0];grad_norm.extend(g.flatten(1).norm(dim=-1).cpu().tolist());assert torch.isfinite(g).all()
                out,detail=V.edit(model,z,cc)
                with torch.no_grad():after.extend(model(out,zero,cc).cpu().tolist())
                changes.append(detail);parts.append(out.cpu())
            assert max(grad_norm)>0;bank.append(torch.cat(parts));diagnostics[arm]=dict(changes=changes,source_logits=before,edited_logits=after,input_gradient_L2=grad_norm,
                note='logit ascent is a proxy, not an image gain conclusion')
        bank=torch.stack(bank,1);assert bank.shape==(400,4,4,64,64) and torch.equal(bank[:,0],source.cpu())
        U.save_array(a.output/'Dcheck_edited_latents.npy',bank.numpy());U.atomic(a.output/'Dcheck_edit_diagnostics.json',diagnostics)
        U.save_array(a.output/'Dfit_noise.npy',held_noise.cpu().numpy())
        parity=encoded['T400'][0,1:2];U.save_array(a.output/'parity_latent.npy',parity.cpu().numpy());saved=N.save_image(a.output/'parity_vae_roundtrip.webp',decode(pipe,parity)[0])
        U.atomic(a.output/'VAE_parity_reference.json',dict(image=saved,latent=U.ref(a.output/'parity_latent.npy'),same_common_reencoded_source_not_original_sampler_terminal=True))
        lock=dict(status='D19_ALL400_FOUR_LATENT_ARMS_LOCKED_BEFORE_IMAGES',latents=U.ref(a.output/'Dcheck_edited_latents.npy'),model=U.ref(checkpoint),spec=U.ref(a.spec),
            source_latents=U.ref(a.output/'Dcheck400_latents.npy'),source_images=prepared['mapping'],arms=V.ARMS,query_ids=[q['query_id'] for q in mapping['Dcheck400']],
            fixed_radius=.1,fixed_edit_steps=8,zero_arm_bitwise_source_identity=True,Dcheck_target_pixels_F_or_scores_used=False,normal_inference_only_current_query_EEG_and_generated_image=True,
            diffusion_score_or_posterior_claim=False,extra_UNet_calls=0)
        U.atomic(a.output/'edit_lock.json',lock);assert calls[0]==0
        for ref in spec['source_pins']:U.checked(ref)
        check();assert sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())<768<<20
        done=dict(status='D19_THREE_FIXED_IMAGE_DOMAIN_HEADS_AND_ALL400_LOCKED_ENDPOINT_EDITS_COMPLETE',schema=V.SCHEMA,source=U.ref(__file__),spec=U.ref(a.spec),wrapper=wrapper,
            prepared_inputs=spec['prepared_inputs'],model=U.ref(checkpoint),edit_lock=U.ref(a.output/'edit_lock.json'),Dfit_logits=U.ref(a.output/'Dfit_logits.npy'),Dfit_noise=U.ref(a.output/'Dfit_noise.npy'),
            encoded_latents={name:U.ref(a.output/(name+'_latents.npy')) for name in mapping},edit_diagnostics=U.ref(a.output/'Dcheck_edit_diagnostics.json'),VAE_parity=U.ref(a.output/'VAE_parity_reference.json'),
            epochs=20,updates_per_head=260,query_exposures_per_head=8000,classification_examples_per_head=64000,parameters_per_head=sum(p.numel() for p in models[V.FIELDS[0]].parameters()),
            T_training_queries=400,T_training_concepts=40,training_subjects=1,all20_passes_complete=True,Dfit_optimizer_or_model_selection=False,Dcheck_target_access=False,
            independent_Gaussian_sampling_noise=False,endpoint_radius=.1,endpoint_edit_steps=8,image_admission='FIXED_ACTUAL_IMAGE_TEST_WITHOUT_PROXY_EFFECT_GATE',
            ordinary_EEG_prior_UNet_or_VAE_weights_updated=False,original_D_resumed=False,UNet_forward_calls=0,VAE_encode_images=1600,VAE_decode_parity_images=1,
            normal_inference_UNet_NFE=4,additional_VAE_encode_images_per_query=1,additional_VAE_decode_images_per_query=1,additional_small_CNN_gradient_calls_per_query=8,
            source_distribution='VAE-mode reencoded original generated image files; not sampler endpoint latent',encoding_seconds=encoding_seconds,training_seconds=training_seconds,
            seconds=time.monotonic()-invoked,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,
            exact_diffusion_score_or_posterior=False,independent_confirmation=False,SOTA_established=False)
        U.atomic(a.output/'completion.json',done);log(dict(stage='COMPLETE',fixed_heads=3,locked_latent_arms=4));return 0
    except BaseException as error:
        if models:torch.save(dict(models={arm:cpu(x.state_dict()) for arm,x in models.items()},optimizers={arm:cpu(x.state_dict()) for arm,x in optimizers.items()},epoch=epoch,updates=updates,exposures=exposures),a.output/'partial_domain_heads.pt')
        U.atomic(a.output/'failure.json',dict(status='D19_INCOMPLETE_RETAINED_NO_AUTOMATIC_RETRY',active=active,updates_per_head=updates,query_exposures_per_head=exposures,
            seconds=time.monotonic()-invoked,error=repr(error),traceback=traceback.format_exc()));raise

if __name__=='__main__':raise SystemExit(main())
