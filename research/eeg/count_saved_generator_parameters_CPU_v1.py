"""Count exact saved generator architectures without allocating their weights."""
from pathlib import Path
import hashlib
import inspect
import json
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    started=time.monotonic()
    output=ROOT/'audit/current_generator_parameter_count_CPU_20261020_v1.json'
    assert not output.exists()
    from profile_standard_generator import MODEL, ASSETS, torch, UNet2DConditionModel, AutoencoderKL, CLIPTextModel, CLIPTextModelWithProjection
    from transformers import CLIPTextConfig
    from safetensors import safe_open
    torch.set_num_threads(2)
    snapshot=json.loads((MODEL/'completion.json').read_text())
    assert snapshot['revision']=='462165984030d82259a11f4367a4eed129e94a7b'
    records={}
    for folder,cls in [('unet',UNet2DConditionModel),('vae',AutoencoderKL),('text_encoder',CLIPTextModel),('text_encoder_2',CLIPTextModelWithProjection)]:
        config_path=MODEL/folder/'config.json'
        cfg=json.loads(config_path.read_text())
        with torch.device('meta'):
            model=cls(CLIPTextConfig(**cfg)) if folder.startswith('text_encoder') else cls.from_config(cfg)
        assert all(x.device.type=='meta' for x in model.parameters())
        assets=[x for x in snapshot['files'] if Path(x['path']).parent==MODEL/folder and str(x['path']).endswith('.safetensors')]
        assert len(assets)==1,(folder,assets)
        item=assets[0]
        with safe_open(item['path'],framework='pt',device='cpu') as file:
            actual={k:tuple(file.get_slice(k).get_shape()) for k in file.keys()}
        expected={k:tuple(x.shape) for k,x in model.state_dict().items()}
        assert actual==expected,(folder,set(actual)^set(expected))
        records[folder]=dict(class_name=cls.__name__,parameters=sum(x.numel() for x in model.parameters()),
            config_path=str(config_path),config_sha256=sha(config_path),checkpoint=item['path'],
            original_snapshot_checkpoint_sha256=item['sha256'],all_saved_tensor_shapes_match_actual_class=True,
            state_tensors=len(actual),class_source_sha256=sha(inspect.getfile(cls)),
            deployed_precision='float32' if folder=='vae' else 'bfloat16',trainable_at_inference=0)
        del model
    with safe_open(str(ASSETS/'adapter.safetensors'),framework='pt',device='cpu') as file:
        import math
        ip={k:math.prod(file.get_slice(k).get_shape()) for k in file.keys()}
    projection=sum(v for k,v in ip.items() if k.startswith('image_proj.'))
    attention=sum(v for k,v in ip.items() if k.startswith('ip_adapter.'))
    assert projection==8400896 and attention==340787200
    result=dict(status='EXACT_CLASS_META_PARAMETER_COUNTS_MATCH_SAVED_WEIGHT_HEADERS',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256=sha(__file__),
        modules=records,IP_projection_parameters=projection,IP_attention_parameters=attention,
        deployed_UNet_including_IP_parameters=records['unet']['parameters']+projection+attention,
        complete_generator_parameters_including_text_VAE_IP=sum(x['parameters'] for x in records.values())+projection+attention,
        image_encoder_used_at_generation=False,pretrained_revision=snapshot['revision'],
        snapshot_receipt_sha256=sha(MODEL/'completion.json'),GPU_used=False,GPU_initialized=torch.cuda.is_initialized(),
        materialized_full_model_weights=False,new_images=0,seconds=time.monotonic()-started,
        limits=['Meta models are used only for architecture counts; no new inference or full weight load.',
                'Saved tensor names/shapes verified; multi-GB weight byte hashes are from the existing verified snapshot receipt, not rehashed here.',
                'Generator total includes IP projection; do not add projection rows again.',
                'Counts do not establish capacity insufficiency or convergence.'])
    assert not result['GPU_initialized']
    output.write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
        file.write(json.dumps(dict(event_id='current_generator_parameter_count_CPU_20261020_v1',GPU_used=False,
            seconds=result['seconds'],receipt=str(output),receipt_sha256=sha(output)))+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
