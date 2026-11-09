"""Actual frozen Brain-HIVE H14 Fusion Prior -> Turbo images, one/query.

Normal inference receives only audited EEG conditions, metadata and fixed noise.
Oracle conditions are a separate diagnostic arm, never a normal method input.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import time

from brainhive_native_models_v1 import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['published', 'full'], required=True)
    parser.add_argument('--intervention', choices=['correct', 'permuted', 'missing', 'oracle'], required=True)
    parser.add_argument('--split', choices=['benchmark', 'V', 'Tdiag'], required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    assert '/' not in args.run_name
    assert (args.role == 'published') == (args.split != 'V')
    if args.intervention == 'oracle':
        assert args.split == 'Tdiag', 'True-image condition is a diagnostic, never a normal ranked arm'
    parent_root=WORK/'runs'/(args.role+'_source_dropout_v2')
    assert (parent_root/'completion.json').exists()
    assert (ASSETS/'transfer_complete.json').exists()
    output = WORK/'images'/args.run_name
    output.mkdir(parents=True, exist_ok=True)
    assert not (output/'completion.json').exists(), 'Never regenerate a completed arm'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from diffusers import StableDiffusionXLPipeline
    import diffusers
    from PIL import Image, features
    import numpy as np
    assert features.check('webp'), 'Lossless encoder unavailable; stop before any image is generated'
    started = time.monotonic()
    attention_path = ASSETS/'prior_H14/attn_adapter/diffusion_pytorch_model.safetensors'
    plan = json.loads((ASSETS/'transfer_plan.json').read_text())
    asset = next(a for a in plan['assets'] if a['path'].startswith('prior_H14/attn_adapter/'))
    assert attention_path.stat().st_size == asset['bytes'] and sha(attention_path) == asset['sha256']
    verification = json.loads((WORK/'audit/native_asset_verification.json').read_text())
    verification['files'][str(attention_path)] = dict(bytes=asset['bytes'], sha256=asset['sha256'])
    write(WORK/'audit/native_asset_verification.json', verification)
    models = prior_models('cpu', include_attention=True)
    generator_cfg = json.loads((ROOT/'configs/ATM_public_decoder_registration_20261020_v1.json').read_text())
    generator_assets = Path(generator_cfg['generator_asset_root'])
    # Same publicly pinned SDXL-Turbo weights/config, with the author's learned
    # Brain-HIVE IP K/V replacing the generic image adapter completely.
    payload = next(a for a in generator_cfg['generator_assets'] if 'sdxl_turbo_fp16' in a['path'])
    assert sha(payload['path']) == payload['sha256']
    pipe = StableDiffusionXLPipeline.from_single_file(str(payload['path']),
        config=str(generator_assets/'sdxl_config'), torch_dtype=torch.float16,
        local_files_only=True, add_watermarker=False)
    assert type(pipe.scheduler).__name__ == 'EulerAncestralDiscreteScheduler'
    assert pipe.scheduler.config.timestep_spacing == 'trailing'
    adapter = models['attn_adapter'].to(dtype=torch.float16)
    expected_names = adapter.attn_processor_keys
    assert tuple(pipe.unet.attn_processors) == tuple(expected_names), 'Do not partially load an incompatible attention layout'
    adapter.bind_unet(pipe.unet)
    cross_processors = {n:p for n,p in pipe.unet.attn_processors.items() if n.endswith('attn2.processor')}
    assert sum(p.numel() for p in adapter.parameters()) == 340787200
    assert all(type(p).__name__ == 'IPAttnProcessor2_0' for p in cross_processors.values())
    assert all(p.to_k_ip.weight.dtype == torch.float16 and p.to_v_ip.weight.dtype == torch.float16
               for p in cross_processors.values()), 'Learned IP K/V must match UNet/token dtype'
    for module in pipe.unet.modules():
        # 0.30 Attention lacks the newer optional Q/K norm attributes on some builds.
        if type(module).__name__ == 'Attention':
            for name in ('norm_q', 'norm_k'):
                if not hasattr(module, name):
                    setattr(module, name, None)
    pipe.to('cuda')
    for module in pipe.components.values():
        if isinstance(module, nn.Module):
            module.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    fusion_adapter = models['fusion_encoder'].adapter.to('cuda', torch.float16).eval()
    projector = models['proj'].to('cuda', torch.float16).eval()
    del models['fusion_encoder']
    with torch.inference_mode():
        prompt, _, pooled, _ = pipe.encode_prompt('', device='cuda', do_classifier_free_guidance=False)
    protocol = json.loads((ROOT/'configs/split_manifest.json').read_text())
    bindings = json.loads((WORK/'data/actual_input_bindings.json').read_text())
    allowed_inputs=json.loads((WORK/'data/generation_inputs'/(args.split+'.json')).read_text())
    assert not allowed_inputs['target_image_or_class_fields_present']
    native_inputs=allowed_inputs['inputs']
    assert all(set(r)=={'stimulus_id','array_row'} for r in native_inputs)
    expected_queries=allowed_inputs['expected_queries']
    repeats=allowed_inputs['repeat_protocol']
    write(output/'inference_inputs.json', native_inputs)
    write(output/'model_receipt.json', dict(
        source_commit='f06db69fcc4a029439e369c5c3dba6099e57560e', generator='stabilityai/sdxl-turbo',
        generator_revision=payload['revision'], generator_sha256=payload['sha256'],
        prior_revision='135a284109e71f5cde3d3e6d5dee6268a3c7987a',
        resolution=512, steps=4, CFG=0., image_attention_scale=1.,
        learned_IP_parameters=340787200, conditioning='raw brain -> pretrained A -> pretrained P -> learned IP K/V',
        lossless_output='WebP; decoded pixels must exactly equal original PIL RGB',
        single_image_per_query=True, train_seed=1729, sampling_seed_rule=protocol['query_seed_rule'],
        diffusers_version=diffusers.__version__, all_models_frozen=True,
        loader_adaptations=['pinned local single-file checkpoint', 'optional absent Q/K norms assigned None'],
        brain_model_trained_from_scratch=True, generation_gradient_or_extra_steps=False))
    current = {}
    processor_class=type(next(iter(cross_processors.values())))
    original_call=processor_class.__call__
    @functools.wraps(original_call)
    def checked_attention(self,*call_args,**call_kwargs):
        ip=call_kwargs.get('ip_hidden_states')
        assert ip is not None and ip.shape==(1,4,2048)
        assert ip.data_ptr()==current['token_pointer'], 'The actual IP path must receive this query condition'
        current['IP_conditioned_attention_calls']+=1
        return original_call(self,*call_args,**call_kwargs)
    processor_class.__call__=checked_attention

    def unet_hook(module, inputs, result):
        current['UNet_NFE'] += inputs[0].shape[0]
        assert torch.isfinite(result[0]).all()

    def vae_hook(module, inputs, result):
        assert torch.isfinite(result).all(), 'Nonfinite decoded VAE output'
        current['VAE_decoder_calls'] += 1
        current['VAE_float_min'] = float(result.min())
        current['VAE_float_max'] = float(result.max())

    handles = [pipe.unet.register_forward_hook(unet_hook), pipe.vae.decoder.register_forward_hook(vae_hook)]
    manifest = output/'inference_manifest.jsonl'
    saved = [json.loads(line) for line in manifest.read_text().splitlines()] if manifest.exists() else []
    done = {row['query_id']:row for row in saved}
    assert len(done)==len(saved)
    for row in saved:
        assert sha(row['generated_path']) == row['generated_sha256']
    # Disk reservation uses a conservative lossless-image estimate; never delete results to proceed.
    free = __import__('shutil').disk_usage(output).free
    assert free > (expected_queries-len(done))*450000 + (512<<20), 'Insufficient persistent image reservation'
    scope = classes()
    subjects = allowed_inputs['subjects']
    with manifest.open('a', buffering=1) as log:
        for subject in subjects:
            folder = parent_root/subject
            if args.split=='Tdiag' and args.intervention=='oracle':
                conditions = torch.load(WORK/'data/fused_train_targets.pt', weights_only=True)
                lookup = {i:i for i in range(16540)}
            elif args.split=='Tdiag':
                model = scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval()
                model.load_state_dict(torch.load(folder/'brain_final.pt',weights_only=True)['model'],strict=True)
                raw = torch.load(bindings['subjects'][subject]['train_path'],weights_only=True,mmap=True)
                with torch.inference_mode():
                    conditions = model(raw[[r['array_row'] for r in native_inputs]][:,bindings['selected_indices'],:].cuda()).float().cpu()
                lookup = {r['array_row']:i for i,r in enumerate(native_inputs)}
                del model, raw
            else:
                state = torch.load(folder/'conditions.pt',weights_only=True)
                assert state['model_sha256']==sha(folder/'brain_final.pt')
                conditions = state['condition']
                lookup = {int(row):i for i,row in enumerate(state['array_rows'])}
            permutation = list(range(len(native_inputs)))
            rng = random.Random(1729)
            for i in range(len(permutation)-1,0,-1):
                j=rng.randrange(i)
                permutation[i],permutation[j]=permutation[j],permutation[i]
            assert all(i!=j for i,j in enumerate(permutation))
            if args.intervention=='missing':
                model = scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval()
                model.load_state_dict(torch.load(folder/'brain_final.pt',weights_only=True)['model'],strict=True)
                with torch.inference_mode():
                    missing = model(torch.zeros(1,17,250,device='cuda')).float().cpu()
                del model
            for index, query in enumerate(native_inputs):
                query_id=f"{subject}/{query['stimulus_id']}/{repeats}"
                if query_id in done:
                    continue
                source_query = native_inputs[permutation[index]] if args.intervention=='permuted' else query
                condition = missing if args.intervention=='missing' else conditions[lookup[source_query['array_row']]:lookup[source_query['array_row']]+1]
                seed=int(hashlib.sha256(f"1729|{subject}|{query['stimulus_id']}|{repeats}|{protocol['version']}".encode()).hexdigest()[:16],16) % (2**63-1)
                current.clear()
                current.update(UNet_NFE=0,VAE_decoder_calls=0,IP_conditioned_attention_calls=0)
                with torch.inference_mode():
                    native = condition.to('cuda',torch.float16)
                    adapted = fusion_adapter(native)
                    tokens = projector(adapted)
                    assert tokens.shape==(1,4,2048) and torch.isfinite(tokens).all()
                    current['token_pointer']=tokens.data_ptr()
                    image = pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(seed),
                        prompt_embeds=prompt,pooled_prompt_embeds=pooled,
                        cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,guidance_scale=0.,output_type='pil').images[0].convert('RGB')
                assert current['UNet_NFE']==4 and current['VAE_decoder_calls']==1
                assert current['IP_conditioned_attention_calls']==4*len(cross_processors)
                current.pop('token_pointer')
                path=output/subject/(Path(query['stimulus_id']).stem+'.webp')
                path.parent.mkdir(exist_ok=True)
                assert not path.exists(), 'Preserve orphan image; explicit repair needed instead of silent regeneration'
                pixels=np.asarray(image)
                image.save(path,format='WEBP',lossless=True,quality=90,method=4)
                with Image.open(path) as decoded:
                    assert np.array_equal(np.asarray(decoded.convert('RGB')),pixels)
                row=dict(query_id=query_id,subject=subject,stimulus_id=query['stimulus_id'],array_row=query['array_row'],
                    repeat_protocol=repeats,intervention=args.intervention,method='BrainHIVE_'+args.run_name,
                    EEG_source_stimulus_id='MISSING_ZERO_EEG' if args.intervention=='missing' else source_query['stimulus_id'],
                    image_seed=seed,generated_path=str(path),generated_sha256=sha(path),
                    condition_sha256=hashlib.sha256(condition.contiguous().numpy().tobytes()).hexdigest(),
                    condition_norm=float(condition.norm()),adapted_norm=float(adapted.float().norm()),
                    target_or_visual_features_read_for_condition=args.intervention=='oracle',
                    oracle_diagnostic_only=args.intervention=='oracle',lossless_pixel_roundtrip=True,**current)
                log.write(json.dumps(row)+'\n')
                done[query_id]=row
                if len(done)%100==0 or len(done)==expected_queries:
                    print(json.dumps(dict(stage='BRAINHIVE_ACTUAL_IMAGES',run=args.run_name,completed=len(done),expected=expected_queries,seconds=time.monotonic()-started)),flush=True)
    for handle in handles:
        handle.remove()
    assert len(done)==expected_queries
    processor_class.__call__=original_call
    write(output/'completion.json',dict(status='ALL_ACTUAL_LOSSLESS_IMAGES_SAVED_INFERENCE_ONLY',queries=len(done),
        intervention=args.intervention,role=args.role,split=args.split,seconds=time.monotonic()-started,
        inference_manifest_sha256=sha(manifest),images_per_query=1,selected_best_seed=False,
        image_quality_success_established=False,component_gain_established=False))
    print('BRAINHIVE_IMAGE_ARM_COMPLETE',args.run_name,flush=True)


if __name__=='__main__':
    main()
