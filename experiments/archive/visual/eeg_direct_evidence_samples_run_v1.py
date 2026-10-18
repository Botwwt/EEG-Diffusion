"""SERVER ONLY run function. Fixed-budget direct visual evidence experiment."""
def run():
    import gc
    import io
    import zipfile
    from typing import Dict, List, Tuple, Union
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TVF
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
    from diffusers import StableDiffusionXLPipeline
    from diffusers.models.embeddings import ImageProjection
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint, convert_open_clip_checkpoint
    import diffusers.schedulers.scheduling_euler_ancestral_discrete as scheduling
    from safetensors.torch import load_file
    verify(TRAIN); verify(QUERY)
    gate = json.loads((TRAIN / 'results.json').read_bytes())
    assert gate['full_VAE512_resize224_gradient_chain_passed']
    assert gate['three_distinct_fit_concept_gradient_probes_passed']
    ids = json.loads((QUERY / 'image_plan.json').read_bytes())['source_image_indices']
    assert len(ids) == len(set(ids)) == len({i // 10 for i in ids}) == 32
    prior_dir = Path('/root/mced_eeg_norm_stats_point_priors_v1')
    verify(prior_dir)
    prior = json.loads((prior_dir / 'results.json').read_bytes())
    os.kill(prior['pid'], 0)
    pe = next(r for r in prior['exports'] if r['name'] == 'private_norm_stats_prior_predictions.pt')
    pred = torch.load(adopt(pe), weights_only=True)
    assert pred['target_space'] == 'Original_CLIP_H_final_raw1024'
    positions = {v: i for i, v in enumerate(pred['source_indices'].tolist())}
    select = [positions[i] for i in ids]
    point = pred['predictions']['2026109501']['point_no_stats'][select].clone().float()
    constant = pred['fit_mean'][None].expand(32, -1).clone().float()
    conditions = {arm: point if arm == 'point_eeg' else constant for arm in ARMS}
    del pred
    cg = Path('/sys/fs/cgroup')
    stat = dict(x.split() for x in (cg / 'memory.stat').read_text().splitlines())
    host = int((cg / 'memory.max').read_text()) - int((cg / 'memory.current').read_text()) + int(stat['file']) - int(stat['shmem'])
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 20 << 30 and host > 26 << 30
    write('resource_preflight.json', dict(host_reclaimable_available_bytes=host, required_host_bytes=26 << 30,
        required_GPU_free_bytes=20 << 30, loading_budget_is_estimate_not_peak_guarantee=True))
    saved = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), torch.get_num_threads(),
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    hooks = []; modules = []; fixed = []; original_randn = scheduling.randn_tensor
    try:
        source = B / 'eeg_generator_conditioning_training_v1.py'
        assert identity(source) == dict(bytes=21503, sha256='8d05598382a46b0f41c0eb97c54ab0570c20fa64b62fc9b6a1694c0385e4a659')
        body = next(n for n in ast.parse(source.read_bytes()).body if isinstance(n, ast.Try)).body
        lo = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pipe' for t in n.targets))
        hi = next(i for i, n in enumerate(body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'load_ip_adapter')
        namespace = {**globals(), **locals()}
        assert 'namespace' not in namespace
        exec(compile(ast.Module(body=body[lo:hi], type_ignores=[]), str(source), 'exec'), namespace)
        pipe = namespace['pipe']; write('strict_base_weight_coverage.json', namespace['coverage']); del namespace
        pipe.load_ip_adapter(str(ASSET), subfolder='', weight_name='adapter.safetensors', image_encoder_folder=None, torch_dtype=torch.float32)
        pipe.set_ip_adapter_scale(1.)
        projection = pipe.unet.encoder_hid_proj.image_projection_layers[0]
        assert type(projection) is ImageProjection and projection.image_embeds.in_features == 1024
        weights = load_file(str(ASSET / 'adapter.safetensors'), device='cpu'); consumed = set()
        for dst, src in [('image_embeds.weight', 'image_proj.proj.weight'), ('image_embeds.bias', 'image_proj.proj.bias'),
                         ('norm.weight', 'image_proj.norm.weight'), ('norm.bias', 'image_proj.norm.bias')]:
            assert torch.equal(projection.state_dict()[dst].cpu().float(), weights[src].float()); consumed.add(src)
        for index, (name, processor) in enumerate(pipe.unet.attn_processors.items()):
            if name.endswith('attn1.processor'): continue
            for suffix in ['to_k_ip', 'to_v_ip']:
                src = f'ip_adapter.{index}.{suffix}.weight'
                assert torch.equal(processor.state_dict()[suffix + '.0.weight'].cpu().float(), weights[src].float()); consumed.add(src)
        assert consumed == set(weights)
        write('strict_IP_adapter_coverage.json', dict(complete=True, tensor_keys=len(consumed), projection_replaced=False,
            all_loaded_tensors_exact_after_declared_FP32_cast=True))
        del weights
        pipe.to('cuda'); pipe.set_progress_bar_config(disable=True)
        modules = [pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2]
        for model in modules: model.eval().requires_grad_(False)
        assert pipe.image_encoder is None and pipe.unet.config.time_cond_proj_dim is None
        assert pipe.scheduler.config.prediction_type == 'epsilon'
        assert float(pipe.vae.config.scaling_factor) == .13025
        assert identity(Path(scheduling.__file__))['sha256'] == 'fd96727bebe53793747ac7c151afbb4769f9d413a92cff8186fd51bcab3eb1ba'
        conditions = {k: v.cuda() for k, v in conditions.items()}
        trace, noise_trace, guidance_trace = [], [], []
        run_name = [None]
        def tensor_hash(x):
            return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
        def unet_before(model, args, kwargs):
            x = args[0] if args else kwargs['sample']; t = args[1] if len(args) > 1 else kwargs['timestep']
            c = kwargs['added_cond_kwargs']['image_embeds'][0]
            assert x.shape == (1,4,64,64) and c.shape == (1,1,1024)
            COUNTS['UNet_calls'] += 1
            trace.append(dict(run=run_name[0], timestep=float(t), latent_sha256=tensor_hash(x), condition_sha256=tensor_hash(c)))
        def vae_before(model, args): COUNTS['VAE_decoder_calls'] += 1
        def text_before(model, args): COUNTS['text_encoder_calls'] += 1
        def project_before(model, args): COUNTS['image_projection_calls'] += 1
        def recorded_randn(*args, **kwargs):
            value = original_randn(*args, **kwargs)
            noise_trace.append(dict(run=run_name[0], noise_sha256=tensor_hash(value)))
            return value
        hooks = [pipe.unet.register_forward_pre_hook(unet_before, with_kwargs=True), pipe.vae.decoder.register_forward_pre_hook(vae_before),
            pipe.text_encoder.register_forward_pre_hook(text_before), pipe.text_encoder_2.register_forward_pre_hook(text_before),
            projection.register_forward_pre_hook(project_before)]
        scheduling.randn_tensor = recorded_randn
        with torch.no_grad():
            prompt, _, pooled, _ = pipe.encode_prompt(prompt='', device=torch.device('cuda'), num_images_per_prompt=1, do_classifier_free_guidance=False)
            time_ids = pipe._get_add_time_ids((512,512), (0,0), (512,512), dtype=prompt.dtype,
                text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to('cuda')
            run_name[0] = 'reference_pipeline/0'
            reference = pipe(prompt_embeds=prompt, pooled_prompt_embeds=pooled,
                ip_adapter_image_embeds=[conditions['constant_prior'][:1,None,:]], num_inference_steps=4,
                guidance_scale=0., height=512, width=512, generator=torch.Generator(device='cuda').manual_seed(2026111600), output_type='latent').images
        for model in [pipe.text_encoder, pipe.text_encoder_2]: model.to('meta')
        gc.collect(); torch.cuda.empty_cache(); ctypes.CDLL(None).malloc_trim(0)

        def sample_latent(condition, noise_seed, cue=None):
            gen = torch.Generator(device='cuda').manual_seed(noise_seed)
            pipe.scheduler.set_timesteps(4, device=torch.device('cuda'))
            with torch.no_grad():
                z = pipe.prepare_latents(1,4,512,512,prompt.dtype,torch.device('cuda'),gen,None)
            for step, timestep in enumerate(pipe.scheduler.timesteps):
                with torch.no_grad():
                    inp = pipe.scheduler.scale_model_input(z, timestep)
                    epsilon = pipe.unet(inp,timestep,encoder_hidden_states=prompt,
                        added_cond_kwargs=dict(text_embeds=pooled,time_ids=time_ids,image_embeds=[condition[None,None,:]]),return_dict=False)[0]
                if cue is not None and step >= 2:
                    sigma = pipe.scheduler.sigmas[pipe.scheduler.step_index]
                    with torch.enable_grad():
                        x0 = (z - sigma * epsilon).detach().requires_grad_(True)
                        rgb = (pipe.vae.decode(x0 / pipe.vae.config.scaling_factor, return_dict=False)[0] / 2 + .5).clamp(0,1)
                        v = F.normalize(head(prefix12(clip_preprocess(rgb))), dim=-1)
                        score = temperature * (v * cue).sum()
                        grad = torch.autograd.grad(score, x0)[0]; COUNTS['backward_calls'] += 1
                    assert torch.isfinite(grad).all() and grad.norm() > 0
                    guidance_trace.append(dict(run=run_name[0],step=step,sigma=float(sigma),score_before=float(score.detach()),
                        gradient_norm=float(grad.norm()),clean_prediction_delta_norm=float((sigma.square()*grad).norm()),
                        cue_sha256=tensor_hash(cue),temperature=temperature))
                    epsilon = epsilon - sigma * grad.detach()
                    del x0, rgb, v, score, grad
                with torch.no_grad(): z = pipe.scheduler.step(epsilon,timestep,z,generator=gen,return_dict=False)[0]
                assert torch.isfinite(z).all()
            return z
        run_name[0] = 'reference_manual/0'
        replay = sample_latent(conditions['constant_prior'][0],2026111600)
        assert torch.equal(reference,replay)
        assert [r['latent_sha256'] for r in trace[:4]] == [r['latent_sha256'] for r in trace[4:8]]
        assert [r['noise_sha256'] for r in noise_trace[:4]] == [r['noise_sha256'] for r in noise_trace[4:8]]
        write('manual_sampler_replay.json',dict(complete=True,original_pipeline_final_latent_bitwise_equal=True,
            all_four_scaled_UNet_inputs_bitwise_equal=True,all_four_actual_ancestral_noise_draws_bitwise_equal=True,UNet_calls=8))
        del reference,replay

        # Reuse the exact passed strict CLIP/Visual loader, ending before any
        # fit-image preparation. No pixel, gradient, or model call is executed by this slice.
        source = B / 'eeg_direct_evidence_vae_preflight_v1.py'
        assert identity(source) == dict(bytes=18223,sha256='713e25ad95ff5b8c753264d6a401dff363460cffc8df8342163f844720a818f5')
        body = next(n for n in ast.parse(source.read_bytes()).body if isinstance(n,ast.Try)).body
        lo = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='parent' for t in n.targets))
        hi = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='fit_ids' for t in n.targets))
        ns = {**globals(), **locals(), 'ASSET':PARENT,
            'MODEL':Path('/root/mced_eeg_single_visual_persistent_checkpoints_v1'),
            'AUDIT':Path('/root/mced_eeg_brainhive_reference_audit_v1/source')}
        exec(compile(ast.Module(body=body[lo:hi],type_ignores=[]),str(source),'exec'),ns)
        vision,visual,packet,state = [ns[k] for k in ['vision','visual','packet','state']]
        write('vision_asset_adoption.json',dict(original=ns['asset'],own_readonly_path=str(ns['own']),identity=identity(ns['own'])))
        qe,cp = ns['qe'],ns['cp']; del ns
        qpositions = {v:i for i,v in enumerate(packet['validation_indices'].tolist())}
        correct = F.normalize(packet['queries']['2026109501']['validation']['real_stats'][[qpositions[i] for i in ids],:1024].clone().cuda(),dim=-1)
        temperature = float(state['logit_scale'].exp().clamp(max=100))
        assert 0 < temperature <= 100
        del packet,state
        # Discard unused inference copies only, preserving the whole original file FD.
        for layer in vision.vision_model.encoder.layers[12:]: layer.to('meta')
        vision.vision_model.post_layernorm.to('meta'); vision.visual_projection.to('meta')
        modules.extend([vision,visual])
        mean = torch.tensor([.48145466,.4578275,.40821073],device='cuda')[None,:,None,None]
        std = torch.tensor([.26862954,.26130258,.27577711],device='cuda')[None,:,None,None]
        fs = dict(vision=vision,visual=visual,F=F,COUNTS=COUNTS,TVF=TVF,InterpolationMode=InterpolationMode,mean=mean,std=std)
        nodes = [n for n in body if isinstance(n,ast.FunctionDef) and n.name in ['prefix12','head','resize_rgb','clip_preprocess']]
        assert [n.name for n in nodes] == ['prefix12','head','resize_rgb','clip_preprocess']
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),fs)
        prefix12,head,clip_preprocess = [fs[k] for k in ['prefix12','head','clip_preprocess']]
        verify(FEATURE,['train_block12_token_mean.pt'])
        with torch.no_grad():
            cached = torch.load(FEATURE/'train_block12_token_mean.pt',weights_only=True)[ids].clone().cuda()
            oracle = F.normalize(head(cached),dim=-1)
        cues = dict(direct_eeg=correct,direct_shuffled_eeg=correct.roll(1,0),direct_true_visual_oracle=oracle)
        assert all(torch.isfinite(v).all() for v in list(conditions.values())+list(cues.values()))
        write('cue_bindings.json',dict(source_query_export=qe,trained_visual_checkpoint=identity(cp),
            original_training_temperature=temperature,correct_per_image=[tensor_hash(v[None]) for v in correct],
            sham_per_image=[tensor_hash(v[None]) for v in cues['direct_shuffled_eeg']],
            oracle_per_image=[tensor_hash(v[None]) for v in oracle],sham_source_indices=ids[-1:]+ids[:-1],
            only_oracle_uses_target_image_features=True))
        fixed = [p for model in [pipe.unet,pipe.vae,vision,visual] for p in list(model.parameters())+list(model.buffers()) if p.device.type != 'meta']
        versions = [p._version for p in fixed]
        assert all(p.dtype == torch.float32 for p in fixed if p.is_floating_point())
        cohorts,exports,first_latents,noise_by_image,compatibility = [],[],{},{},[]
        for arm in ARMS:
            images = []
            for position,index in enumerate(ids):
                run_name[0] = arm+'/'+str(index); start=len(trace); nstart=len(noise_trace)
                cue = cues[arm][position:position+1] if arm in cues else None
                z = sample_latent(conditions[arm][position],2026111600+index,cue)
                with torch.no_grad():
                    rgb = (pipe.vae.decode(z/pipe.vae.config.scaling_factor,return_dict=False)[0]/2+.5).clamp(0,1)
                    image = rgb[0].permute(1,2,0).cpu().numpy()
                    score = (F.normalize(head(prefix12(clip_preprocess(rgb))),dim=-1)*correct[position:position+1]).sum()
                assert image.shape==(512,512,3) and np.isfinite(image).all() and len(trace)-start==4 and len(noise_trace)-nstart==4
                first = trace[start]['latent_sha256']; noises=[r['noise_sha256'] for r in noise_trace[nstart:]]
                if index in first_latents: assert first_latents[index]==first and noise_by_image[index]==noises
                else: first_latents[index]=first; noise_by_image[index]=noises
                compatibility.append(dict(arm=arm,index=index,correct_EEG_cosine=float(score),diagnostic_not_independent_quality=True))
                images.append(np.rint(image*255).astype(np.uint8))
                if (position+1)%4==0: print(json.dumps(dict(stage='DIRECT_EVIDENCE_IMAGES',arm=arm,images=position+1,counts=COUNTS)),flush=True)
            name=arm+'_samples.npz'; lib=ctypes.CDLL(None,use_errno=True)
            fd=lib.syscall(ctypes.c_long(319),ctypes.c_char_p(name.encode()),ctypes.c_uint(3)); assert fd>=0,ctypes.get_errno(); FDS.append(fd)
            with os.fdopen(os.dup(fd),'wb') as f: np.savez_compressed(f,arr_0=np.stack(images),image_indices=np.asarray(ids,dtype=np.int64))
            fcntl.fcntl(fd,1033,15); assert fcntl.fcntl(fd,1034)==15
            p=Path(f'/proc/{os.getpid()}/fd/{fd}'); (OUT/name).symlink_to(p)
            row=dict(arm=arm,file=name,images=32,memfd_path=str(p),identity=identity(p),oracle_reference=arm=='direct_true_visual_oracle',storage='sealed_RAM_volatile')
            cohorts.append(row); exports.append(dict(name=name,memfd_path=str(p),**row['identity'])); write(arm+'_ready.json',row)
            del images
        expected=dict(UNet_calls=648,VAE_decoder_calls=352,text_encoder_calls=2,image_projection_calls=648,
            optimizer_updates=0,EEG_calls=0,prior_calls=0,prefix12_calls=352,prefix12_states=352,
            visual_head_calls=353,visual_head_states=384,backward_calls=192)
        assert COUNTS==expected,COUNTS
        assert versions==[p._version for p in fixed] and all(p.grad is None for p in fixed)
        write('trace.json',trace); write('ancestral_noise_trace.json',noise_trace); write('guidance_trace.json',guidance_trace)
        write('compatibility_diagnostic.json',compatibility)
        write('input_bindings.json',dict(gradient_gate_commit=identity(TRAIN/'final_commit.json'),query_plan=identity(QUERY/'image_plan.json'),
            prediction_export=pe,point_prior_commit=identity(prior_dir/'final_commit.json'),evidence_loader=identity(source)))
        return dict(complete=True,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-START,
            counts=COUNTS,cohorts=cohorts,exports=exports,image_indices=ids,new_images=160,intermediate_guidance_VAE_decodes=192,
            paired_initial_latent_hashes={str(k):v for k,v in first_latents.items()},paired_actual_ancestral_noise_hashes={str(k):v for k,v in noise_by_image.items()},
            original_generator_and_IPAdapter_weights_unchanged=True,manual_sampler_bitwise_replayed=True,
            extra_guidance_compute_not_matched=True,one_fixed_EEG_seed=True,source_heldout_only=True,source32_reused_exploratory=True,
            official_test_read=False,calibrated_likelihood=False,metrics_not_yet_computed=True,novelty_established=False,SOTA=False)
    finally:
        scheduling.randn_tensor=original_randn
        for handle in hooks: handle.remove()
        fixed.clear()
        for model in modules: model.to('meta')
        conditions.clear()
        cpu,cuda,threads,tf32,cudnn=saved
        torch.set_rng_state(cpu); torch.cuda.set_rng_state_all(cuda); torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32=tf32; torch.backends.cudnn.allow_tf32=cudnn
