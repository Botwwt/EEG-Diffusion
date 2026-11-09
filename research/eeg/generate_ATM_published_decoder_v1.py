"""Published EEG features + matching author prior + original Turbo generation.

This reproduces the released decoder branch, not its unavailable EEG encoder.
All ground-truth image access belongs to the separate evaluation process.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    a = p.parse_args()
    registration = json.loads(a.registration.read_text())
    assert registration['scope'] == 'ALL2000_PUBLISHED_FEATURE_DECODER_REPRODUCTION_ONLY'
    assert registration['seed'] == 1729 and registration['query_count'] == 2000
    for path, digest in registration['inference_source_sha256'].items():
        assert sha(path) == digest, path
    sys.path[:0] = registration['python_paths']
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TQDM_DISABLE'] = '1'
    import torch
    import numpy as np
    import diffusers
    import transformers
    import inspect
    from diffusers import StableDiffusionXLPipeline
    from safetensors import safe_open
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    start = time.monotonic()
    out = Path(registration['output_root'])
    assert not (out/'inference_manifest.jsonl').exists(), 'Never overwrite or silently reuse images'
    assert not (out/'completion.json').exists()
    targets = [json.loads(x) for x in Path(registration['inference_manifest']).read_text().splitlines()]
    assert len(targets) == 2000 and len({x['query_id'] for x in targets}) == 2000
    permitted = {'query_id','subject','stimulus_id','array_row','image_seed','prior_seed'}
    assert all(set(x) == permitted for x in targets), 'Inference manifest must contain no target labels or pointers'
    source = Path(registration['author_source_root'])/'Generation'
    prior_module = load(source/'diffusion_prior.py', 'published_author_prior')
    generator_module = load(source/'custom_pipeline.py', 'published_author_generator')
    for item in registration['generator_assets']:
        assert sha(item['path']) == item['sha256']
    assets = Path(registration['generator_asset_root'])
    # Loader adaptation only: the pinned author generation methods below are unchanged.
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(assets/'sdxl_turbo_fp16.safetensors'), config=str(assets/'sdxl_config'),
        torch_dtype=torch.float16, local_files_only=True, add_watermarker=False)
    pipe.load_ip_adapter(str(assets), subfolder='', weight_name='adapter.safetensors',
                         image_encoder_folder=None, torch_dtype=torch.float16,
                         local_files_only=True)
    assert pipe.unet.config.sample_size*pipe.vae_scale_factor == 512
    assert type(pipe.scheduler).__name__ == 'EulerAncestralDiscreteScheduler'
    assert pipe.scheduler.config.timestep_spacing == 'trailing'
    pipe.set_ip_adapter_scale(1.0)
    pipe.to('cuda')
    for module in pipe.components.values():
        if isinstance(module, torch.nn.Module):
            module.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    pipe.generate_ip_adapter_embeds = generator_module.generate_ip_adapter_embeds.__get__(pipe)
    generator = generator_module.Generator4Embeds.__new__(generator_module.Generator4Embeds)
    generator.num_inference_steps = 4
    generator.dtype = torch.float16
    generator.device = 'cuda'
    generator.pipe = pipe

    def tensor_sha(x):
        return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

    projection = pipe.unet.encoder_hid_proj.image_projection_layers[0]
    actual = {k: tensor_sha(v) for k, v in projection.state_dict().items()}
    key_map = {'image_embeds.weight': 'image_proj.proj.weight',
               'image_embeds.bias': 'image_proj.proj.bias',
               'norm.weight': 'image_proj.norm.weight', 'norm.bias': 'image_proj.norm.bias'}
    with safe_open(str(assets/'adapter.safetensors'), framework='pt', device='cpu') as f:
        assert actual == {k: tensor_sha(f.get_tensor(v).to(torch.float16)) for k, v in key_map.items()}
        attention_hashes = {}
        source_attention_keys = set()
        key_id = 1
        attention_parameters = 0
        # This mapping is the installed diffusers0.30 conversion path, inspected
        # before registration; verify every deployed image-attention K/V tensor.
        for name, processor in pipe.unet.attn_processors.items():
            if name.endswith('attn1.processor') or 'motion_modules' in name:
                continue
            state = processor.state_dict()
            assert set(state) == {'to_k_ip.0.weight','to_v_ip.0.weight'}
            assert processor.scale == [1.0]
            for kind in ['to_k_ip','to_v_ip']:
                public_key = f'ip_adapter.{key_id}.{kind}.weight'
                deployed = state[kind+'.0.weight']
                public = f.get_tensor(public_key).to(torch.float16)
                assert deployed.shape == public.shape and tensor_sha(deployed) == tensor_sha(public)
                attention_hashes[name+'.'+kind] = tensor_sha(deployed)
                source_attention_keys.add(public_key)
                attention_parameters += deployed.numel()
            key_id += 2
        assert source_attention_keys == {k for k in f.keys() if k.startswith('ip_adapter.')}
        assert attention_parameters == 340787200
    model_receipt = dict(actual_projection_sha256=actual, projection_parameters=sum(x.numel() for x in projection.parameters()),
                         actual_image_attention_sha256=attention_hashes,image_attention_parameters=attention_parameters,
                         generator='stabilityai/sdxl-turbo', precision='float16', resolution=512,
                         steps=4, CFG=0.0, IP_scale=1.0, scheduler=dict(pipe.scheduler.config),
                         author_methods_modified=False, loader='local pinned public single-file checkpoint',
                         unused_image_encoder_omitted=True, watermarker_disabled=True,
                         torch_version=torch.__version__,diffusers_version=diffusers.__version__,
                         transformers_version=transformers.__version__,all_networks_frozen_for_inference=True,
                         actual_IP_loader_source_sha256=sha(inspect.getsourcefile(type(pipe.unet)._convert_ip_adapter_attn_to_diffusers)),
                         registration_sha256=sha(a.registration))
    (out/'model_receipt.json').write_text(json.dumps(model_receipt, indent=2)+'\n')
    current = {}

    def unet_hook(module, inputs, output):
        current['UNet_NFE'] += len(inputs[0])
        value = output[0]
        assert torch.isfinite(value).all(), 'Nonfinite UNet prediction'

    def projection_hook(module, inputs, output):
        assert inputs[0].numel() == 1024
        assert tensor_sha(inputs[0]) == current['deployed_condition_sha256']
        assert tuple(output.shape) == (1, 4, 2048) and torch.isfinite(output).all()
        current['projection_calls'] += 1

    def decoder_hook(module, inputs, output):
        assert torch.isfinite(output).all(), 'Nonfinite VAE RGB output'
        current['VAE_decoder_calls'] += 1
        current['decoded_float_min'] = float(output.min())
        current['decoded_float_max'] = float(output.max())

    handles = [pipe.unet.register_forward_hook(unet_hook),
               projection.register_forward_hook(projection_hook),
               pipe.vae.decoder.register_forward_hook(decoder_hook)]
    written = 0
    with (out/'inference_manifest.jsonl').open('x') as manifest:
        for subject in registration['subjects']:
            asset = registration['subjects'][subject]
            ce = torch.load(asset['features'], weights_only=True, map_location='cpu')
            state = torch.load(asset['prior'], weights_only=True, map_location='cpu')
            assert sha(asset['features']) == asset['features_sha256']
            assert sha(asset['prior']) == asset['prior_sha256']
            assert tuple(ce.shape) == (200, 1024) and torch.isfinite(ce).all()
            prior = prior_module.DiffusionPriorUNet(cond_dim=1024, dropout=0.1).cuda().eval().requires_grad_(False)
            prior.load_state_dict(state, strict=True)
            prior_pipe = prior_module.Pipe(prior, device='cuda')
            assert prior_pipe.scheduler.config.clip_sample
            def prior_hook(module, inputs, output):
                assert torch.isfinite(output).all(), 'Nonfinite prior noise prediction'
                current['prior_NFE'] += len(inputs[0])
            prior_handle = prior.register_forward_hook(prior_hook)
            selected = sorted([x for x in targets if x['subject'] == subject], key=lambda x: x['array_row'])
            assert [x['array_row'] for x in selected] == list(range(200))
            for target in selected:
                row = int(target['array_row'])
                condition = ce[row:row+1].float().cuda()
                prior_rng = torch.Generator(device='cuda').manual_seed(target['prior_seed'])
                image_rng = torch.Generator(device='cuda').manual_seed(target['image_seed'])
                current.clear()
                current.update(UNet_NFE=0, projection_calls=0, VAE_decoder_calls=0, prior_NFE=0)
                with torch.inference_mode():
                    h = prior_pipe.generate(c_embeds=condition, num_inference_steps=50,
                                            guidance_scale=5.0, generator=prior_rng)
                    assert torch.isfinite(h).all()
                    current['deployed_condition_sha256'] = tensor_sha(h.to(torch.float16))
                    image = generator.generate(h, generator=image_rng)
                assert current['prior_NFE'] == 100
                assert current['UNet_NFE'] == 4 and current['projection_calls'] == 4
                assert current['VAE_decoder_calls'] == 1 and image.size == (512, 512)
                pixels = np.asarray(image.convert('RGB'))
                assert pixels.dtype == np.uint8 and pixels.shape == (512, 512, 3)
                filename = Path(target['stimulus_id']).stem+'.png'
                assert Path(filename).name == filename
                saved = out/'images'/subject/filename
                saved.parent.mkdir(parents=True, exist_ok=True)
                assert not saved.exists()
                image.convert('RGB').save(saved, format='PNG')
                record = target | dict(generated_path=str(saved), generated_sha256=sha(saved),
                    EEG_native_condition_sha256=tensor_sha(condition), prior_output_sha256=tensor_sha(h),
                    prior_output_norm=float(h.norm()), deployed_condition_sha256=current['deployed_condition_sha256'],
                    actual_deployed_projection_tensor_sha256=actual, **{k:v for k,v in current.items() if k!='deployed_condition_sha256'},
                    prior_steps=50, prior_CFG=5.0, prior_clip_sample=True,
                    steps=4, CFG=0.0, IP_scale=1.0, resolution=512,
                    target_image_or_visual_feature_read_at_inference=False,
                    independent_EEG_feature_row_identity_reconstruction=False,
                    pretrained_prior_sha256=asset['prior_sha256'],
                    pretrained_EEG_feature_sha256=asset['features_sha256'],
                    registration_sha256=sha(a.registration))
                manifest.write(json.dumps(record)+'\n')
                manifest.flush()
                written += 1
                if written % 25 == 0:
                    progress=dict(status='RUNNING',completed=written,expected=2000,seconds=time.monotonic()-start)
                    (out/'progress.json').write_text(json.dumps(progress)+'\n')
                    print(json.dumps(progress),flush=True)
            prior_handle.remove()
            del prior,prior_pipe,state,ce
            torch.cuda.empty_cache()
    for handle in handles:
        handle.remove()
    assert written == 2000
    receipt=dict(status='COMPLETED_ALL2000_PUBLISHED_FEATURE_DECODER_IMAGES',queries=written,
                 inference_manifest_sha256=sha(out/'inference_manifest.jsonl'),seconds=time.monotonic()-start,
                 full_EEG_encoder_reproduction=False,component_gain_proven=False,
                 image_selection_or_regeneration=False,UNet_sample_evaluations=8000,
                 prior_sample_evaluations=200000)
    (out/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
