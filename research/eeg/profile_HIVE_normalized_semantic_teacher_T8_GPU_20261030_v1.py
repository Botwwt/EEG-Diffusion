"""Fixed T8 actual nonlinear E-step and original EEG M-step engineering.

Three matched training-only teacher arms, four temporary E updates each. All
are reported; no effect score, research image, new EEG update or final model.
"""
import argparse,datetime,gc,hashlib,importlib.util,io,json,math,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')


def read(path):return json.loads(Path(path).read_bytes())
def ref(path):
    path=Path(path)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest)
def checked(item):
    actual=ref(item['path']);assert actual['sha256']==item['sha256']
    if 'bytes' in item:assert actual['bytes']==item['bytes']
    return Path(item['path'])
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main(protocol_path):
    s=read(protocol_path)
    assert s['status']=='REGISTERED_FIXED_T8_NORMALIZED_SEMANTIC_TEACHER_ENGINEERING' and s['host']=='old'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_text()
    for item in s['source_pins']+s['payloads']:checked(item)
    from brainhive_teacher_profile_contract_CPU_20261030_v1 import dependency_state
    def is_live(pid,ticks):
        path=Path('/proc',str(pid),'stat')
        try:fields=path.read_text().rsplit(')',1)[1].split()
        except FileNotFoundError:return False
        return fields[0] not in ('Z','X') and int(fields[19])==ticks
    assert dependency_state(s['dependencies'],is_live,lambda path:read(path) if Path(path).is_file() else None)['ready']
    out=Path(s['output_root']);out.mkdir();phase='IMPORTS';started=time.monotonic()
    def memory():return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:')))*1024
    baseline=memory()
    sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    sys.path.insert(0,str(Path(protocol_path).parent))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from diffusers import EulerAncestralDiscreteScheduler
    from PIL import Image
    import brainhive_native_models_v1 as N
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_generator_v1 import load_native,denoise
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from brainhive_semantic_teacher_Torch_20261029_v1 import power_rsample,power_entropy,power_mean,native_F_condition,F_native_coordinates,mixture_prior_log_density,semantic_teacher_loss,exact_moment_M_step
    from brainhive_spherical_predictive_density_CPU_20261029_v1 import vmf_log_normalizer,vmf_log_normalizer_array
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,s['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory))
    def budget():
        assert time.monotonic()-started<800
        assert memory()-baseline<s['maximum_extra_CPU_bytes']
        assert torch.cuda.max_memory_allocated()<s['maximum_GPU_bytes']
        assert shutil.disk_usage(out).free>3<<30
        assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<s['output_cap_bytes']
    def digest(module):
        value=hashlib.sha256()
        for name,tensor in module.state_dict().items():value.update(name.encode());value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()
    try:
        phase='FROZEN_CLIP_AND_NATIVE_HOST'
        spec=importlib.util.spec_from_file_location('pinned_CLIP_visual',s['CLIP_source']['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        jit=torch.jit.load(s['CLIP_weights']['path'],map_location='cpu').eval()
        state={name[len('visual.'):]:tensor for name,tensor in jit.state_dict().items() if name.startswith('visual.')}
        critic=module.VisionTransformer(224,14,1024,24,16,768).eval().requires_grad_(False);critic.load_state_dict(state,strict=True);critic.cuda().float();del state,jit;gc.collect()
        (out/'loader').mkdir();audit=ConvertedWeightAudit(torch,out/'loader');audit.install()
        try:pipe,A,P,prompts,host=load_native(training=False)
        finally:audit.restore()
        loading=audit.complete()
        ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
        ordinary.load_state_dict(torch.load(s['ordinary']['path'],map_location='cpu',weights_only=True)['model'],strict=True)
        frozen=dict(A=A,P=P,UNet=pipe.unet,VAE=pipe.vae,critic=critic,ordinary=ordinary)
        before={name:digest(model) for name,model in frozen.items()}
        pipe.vae.float();pipe.vae.enable_slicing();vae_float_digest=digest(pipe.vae)
        prior=np.load(s['prior']['path'],allow_pickle=False);inputs=np.load(s['T8_inputs']['path'],allow_pickle=False)
        metadata=read(s['T8_metadata']['path']);raw=torch.load(s['EEG']['path'],map_location='cpu',weights_only=True,mmap=True)
        bindings=read(s['actual_input_bindings']['path']);assert bindings['selected_indices']==list(range(46,63))
        cfg=read(s['scheduler_config']['path']);assert cfg['prediction_type']=='epsilon' and cfg['timestep_spacing']=='trailing'
        device=lambda value:torch.as_tensor(value,dtype=torch.float64,device='cuda')
        weights=device(prior['weights']);means=device(prior['means']);concentrations=device(prior['concentrations'])
        logC=device(vmf_log_normalizer_array(1023,prior['concentrations']))
        F_weight=device(prior['F_weight']);F_bias=device(prior['F_bias']);radius=float(prior['reference_pre_affine_radius'])
        assert prior['means'].shape==(32,1023) and len(metadata)==8 and np.array_equal(inputs['positions'],s['T_positions'])
        k_sem=s['semantic_concentration'];semantic_logC=vmf_log_normalizer(768,k_sem)
        models={};records=[];teacher_audit={}
        def preprocess(image):
            image=F.interpolate(image,size=(224,224),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            mean=image.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None]
            std=image.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
            return (image-mean)/std
        def render(condition,seed,gradient=False,latent_only=False):
            scheduler=EulerAncestralDiscreteScheduler.from_config(cfg);scheduler.set_timesteps(4,device='cuda');pipe.scheduler=scheduler
            generator=torch.Generator(device='cuda').manual_seed(seed)
            z=pipe.prepare_latents(1,4,512,512,prompts[0].dtype,torch.device('cuda'),generator,None)
            tokens=P(A(condition.half()))
            for t in scheduler.timesteps:
                value=scheduler.scale_model_input(z,t)
                eps=checkpoint(lambda zz,tk,tt:denoise(pipe,prompts,zz,tt,tk),value,tokens,t,use_reentrant=False) if gradient else denoise(pipe,prompts,value,t,tokens)
                z=scheduler.step(eps,t,z,generator=generator).prev_sample
            if latent_only:return z
            def decode(value):return pipe.vae.decode(value.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
            image=checkpoint(decode,z,use_reentrant=False) if gradient else decode(z)
            image=preprocess((image*.5+.5).clamp(0,1))
            feature=checkpoint(critic,image,use_reentrant=False) if gradient else critic(image)
            return F.normalize(feature.float(),dim=-1),z
        def predictive_condition(condition,enabled=True):
            if not enabled:return condition
            return native_F_condition(F_native_coordinates(condition,F_weight,F_bias),F_weight,F_bias,radius).float()
        for j,item in enumerate(metadata):
            phase='FIXED_T8_QUERY_'+str(j);budget();tick=time.monotonic();source=item['subject01_source']
            if source['role'] not in models:
                checked(dict(path=source['checkpoint_path'],sha256=source['checkpoint_SHA256']))
                model=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval().requires_grad_(False)
                model.load_state_dict(torch.load(source['checkpoint_path'],map_location='cpu',weights_only=True)['model'],strict=True)
                models[source['role']]=model
            model=models[source['role']];original_EEG_digest=digest(model)
            eeg=raw[item['array_row']:item['array_row']+1,bindings['selected_indices'],:].cuda()
            seed=int(hashlib.sha256(('HIVE_semantic_teacher_T_'+str(item['array_row'])).encode()).hexdigest()[:8],16)
            with zipfile.ZipFile(item['image_zip']) as archive:image_bytes=archive.read(item['image_member'])
            assert hashlib.sha256(image_bytes).hexdigest()==item['image_sha256']
            real=Image.open(io.BytesIO(image_bytes)).convert('RGB');real_tensor=torch.from_numpy(np.asarray(real).copy()).permute(2,0,1)[None].cuda().float()/255.
            with torch.no_grad():target=F.normalize(critic(preprocess(real_tensor)).float(),dim=-1)
            del real_tensor
            with torch.no_grad():
                condition=model(eeg)
                assert float((condition-device(inputs['individual_raw_condition'][0,j:j+1]).float()).abs().max())<3e-5
                gamma=model.block.layers[0].norm.weight;beta=model.block.layers[0].norm.bias
                adjusted=ordinary(condition,gamma[None].expand_as(condition),beta[None].expand_as(condition))[0]
                disabled=ordinary(condition,gamma[None].expand_as(condition),beta[None].expand_as(condition),enabled=False)[0]
                assert torch.equal(disabled,condition) and predictive_condition(condition,enabled=False) is condition
                projected=predictive_condition(condition)
                assert torch.isfinite(projected).all()
                recovered=F_native_coordinates(projected,F_weight,F_bias)
                assert float((recovered.norm(dim=-1)-radius).abs().max())<3e-5
                parity=[]
                for name,c in (('native',condition),('ordinary',adjusted)):
                    tokens=P(A(c.half()));native=pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(seed),prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,guidance_scale=0.,output_type='latent').images
                    manual=render(c,seed,latent_only=True);assert torch.equal(native,manual)
                    parity.append(name)
            arm_records=[]
            for arm in s['teacher_arms']:
                if arm=='correct_EEG':evidence=device(inputs['correct_evidence'][j]);logZ=device(inputs['barycenter_log_evidence'][j])
                elif arm=='visual_only':evidence=torch.zeros(1023,dtype=torch.float64,device='cuda');logZ=device(0.)
                else:
                    assert arm=='sameconcept_wrong_EEG';evidence=device(inputs['wrong_sameconcept_evidence'][j]);logZ=device(inputs['wrong_barycenter_log_evidence'][j])
                direction=torch.nn.Parameter(device(inputs['visual_latent_mean'][j]))
                log_kappa=torch.nn.Parameter(device(math.log(s['initial_teacher_concentration'])))
                optimizer=torch.optim.Adam([dict(params=[direction],lr=s['direction_learning_rate']),dict(params=[log_kappa],lr=s['log_concentration_learning_rate'])],betas=(.9,.999),weight_decay=0)
                steps=[]
                for step in range(s['temporary_E_updates']):
                    optimizer.zero_grad(set_to_none=True)
                    # Same seed schedule/initialization, not an exact common-random-number assertion for Beta acceptance sampling.
                    torch.manual_seed(1729+j*100+step)
                    kappa=log_kappa.exp();raw_kappa=kappa+torch.log(-torch.expm1(-kappa))
                    v=power_rsample(direction,raw_kappa)
                    c=native_F_condition(v[None],F_weight,F_bias,radius).float()
                    feature,endpoint=render(c,seed,gradient=True)
                    if step==0:
                        with torch.no_grad():plain=render(c.detach(),seed,latent_only=True)
                        assert torch.equal(endpoint.detach(),plain),'Gradient path must preserve every native endpoint'
                    log_prior=mixture_prior_log_density(v,weights,means,concentrations,logC,evidence,logZ)
                    entropy=power_entropy(direction,raw_kappa)
                    loss,terms=semantic_teacher_loss(feature,target,k_sem,semantic_logC,entropy,log_prior)
                    # Common scalar rescaling preserves the exact q derivative
                    # while keeping the frozen FP16 generator backward in the
                    # same range as the earlier squared-semantic profile.
                    loss=loss.mean();(loss*(1024./k_sem)).backward()
                    for parameter in (direction,log_kappa):
                        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                        parameter.grad.mul_(k_sem/1024.)
                    direction_norm=float(direction.grad.norm());concentration_gradient=float(log_kappa.grad)
                    assert direction_norm>0
                    steps.append(dict(step=step,objective=float(loss.detach()),semantic_NLL=float(terms['normalized_semantic_NLL'].mean().detach()),sampled_KL=float(terms['sampled_KL_to_working_prior'].mean().detach()),teacher_concentration=float(kappa.detach()),direction_gradient_norm=direction_norm,log_concentration_gradient=concentration_gradient))
                    optimizer.step()
                    with torch.no_grad():direction.div_(direction.norm())
                    assert all(parameter.grad is None for module in frozen.values() for parameter in module.parameters())
                    del loss,terms,feature,endpoint,c,v,raw_kappa,kappa,entropy,log_prior;budget()
                with torch.no_grad():
                    kappa=log_kappa.exp();raw_kappa=kappa+torch.log(-torch.expm1(-kappa));moment=power_mean(direction,raw_kappa).detach()[None]
                # Actual original EEG network: parameter gradients only, fixed q.
                model.requires_grad_(True);model.block.layers[0].norm.requires_grad_(False);model.zero_grad(set_to_none=True)
                current=model(eeg)
                mloss=exact_moment_M_step(F_native_coordinates(current,F_weight,F_bias),moment,prior['weights'],prior['means'],prior['concentrations'],float(inputs['subject_strengths'][0])).mean()
                mloss.backward();grads=[parameter.grad for parameter in model.parameters() if parameter.requires_grad]
                assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
                mnorm=float(torch.sqrt(sum(grad.double().square().sum() for grad in grads)));assert mnorm>0
                model.zero_grad(set_to_none=True);model.requires_grad_(False);assert digest(model)==original_EEG_digest
                key=str(j)+'_'+arm;teacher_audit[key+'_mean']=direction.detach().cpu().numpy();teacher_audit[key+'_concentration']=log_kappa.detach().exp().cpu().numpy();teacher_audit[key+'_moment']=moment.cpu().numpy()
                arm_records.append(dict(arm=arm,temporary_E_updates=steps,actual_EEG_M_step_parameter_gradient_norm=mnorm,M_step_theta_dependent_cross_entropy=float(mloss.detach()),new_EEG_optimizer_updates=0))
                del direction,log_kappa,optimizer,moment,current,mloss,grads;gc.collect();torch.cuda.empty_cache();budget()
            record=dict(position=item['position'],array_row=item['array_row'],seed=seed,source=source,parity=parity,all_three_arms=arm_records,seconds=time.monotonic()-tick)
            records.append(record);write(out/('T'+str(j)+'.json'),record);print(json.dumps(record),flush=True)
        assert before['VAE']==digest(pipe.vae.half());pipe.vae.float();assert digest(pipe.vae)==vae_float_digest
        assert all(before[name]==digest(module) for name,module in frozen.items() if name!='VAE')
        for item in s['payloads']:checked(item)
        np.savez_compressed(out/'all_T8_temporary_teacher_audit.npz',**teacher_audit);budget()
        result=dict(status='COMPLETE_FIXED_T8_NORMALIZED_SEMANTIC_TEACHER_E_M_ENGINEERING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(protocol_path),queries=8,teacher_arms=s['teacher_arms'],temporary_E_updates=96,actual_EEG_parameter_gradient_checks=24,new_EEG_optimizer_updates=0,new_models_saved=0,new_research_images=0,all_native_and_ordinary_manual_endpoints_exact=True,all_gradient_path_endpoints_exact=True,all_frozen_weight_values_unchanged=True,actual_T8_auxiliary_teachers_temporarily_optimized=True,full_T_teacher_or_EEG_fit_admitted=False,new_method_image_gain_established=False,working_prior_calibration_established=False,full_chain_OOF=False,CLIP_critic_shared_with_official_CLIP_metric=True,DINO_not_used_for_training=True,normalized_semantic_likelihood_not_Turbo_pixel_likelihood=True,MC_objective_not_exact_monotone_EM=True,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=memory()-baseline,seconds=time.monotonic()-started,loader=loading,host=host,NEW_contacted=False)
        write(out/'completion.json',result)
    except BaseException as error:
        write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);main(parser.parse_args().protocol)
