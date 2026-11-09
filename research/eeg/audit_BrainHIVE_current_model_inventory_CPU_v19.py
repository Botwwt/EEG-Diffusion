"""Inspect actual native checkpoints and modules without another GPU job."""
import csv
import json
import math
import os
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_native_models_v1 import *
from brainhive_bounded_component_v1 import BoundedConditionAdapter


def count(model):return sum(p.numel() for p in model.parameters())


def main():
    torch.set_num_threads(2)
    output=ROOT/'audit/BrainHIVE_current_model_inventory_CPU_v19.json'
    assert not output.exists()
    binding=WORK/'data/actual_input_bindings.json';inputs=json.loads(binding.read_text())
    proof=json.loads((WORK/'audit/all40_parent_training_CPU_verification_v1.json').read_text())
    assert proof['status']=='ALL40_CHECKPOINTS_LOGS_AND_CONDITION_IDENTITIES_VERIFIED_CPU'
    scope=classes(include_adapter=True);rows=[]
    for entry in proof['models']:
        checkpoint=Path(entry['checkpoint'])
        assert sha(checkpoint)==entry['checkpoint_sha256']
        state=torch.load(checkpoint,weights_only=True,map_location='cpu')
        model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False)
        model.load_state_dict(state['model'],strict=True)
        assert count(model)==6452224 and state['seed']==1729 and state['epochs']==25
        assert state['updates']==entry['updates']
        rows.append(dict(module='EEG',role=entry['role'],subject=entry['subject'],
            class_name='main.models_brain.BrainMLP',parameters=count(model),
            structure='4250->1024; residual bias-free1024->1024->1024 GELU; LayerNorm',
            input='17x250 physical mean4; published evaluation physical mean80',output='1024 raw LayerNorm coordinates',
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),epochs=25,updates=state['updates'],
            origin='This campaign from scratch; source architecture, not ATM or a reduced ATM',frozen=True))
    del model,state
    models=prior_models('cpu')
    native=[('visual_target_F',models['fusion_encoder'].vision_model,'FusionEncoder',
        'three Linear projections512/1024/1024->1024 summed, residual1024->4096->1024, LayerNorm',
        'public B32(512),H14(1024),VAE-view(1024) features','1024; target creation, not normal EEG inference','fusion_encoder'),
        ('native_A',models['fusion_encoder'].adapter,'ResNet1DLayer',
        'residual bias-free1024->4096->1024 GELU; norm=False','1024 raw EEG prediction','1024','fusion_encoder'),
        ('native_P',models['proj'],'IPProjectionModel','Linear1024->8192, reshape4x2048, LayerNorm2048',
        '1024','4x2048 native IP tokens','proj')]
    for label,model,cls,structure,inp,out,directory in native:
        checkpoint=ASSETS/'prior_H14'/directory/'diffusion_pytorch_model.safetensors'
        rows.append(dict(module=label,role='shared',subject='all',class_name=cls,parameters=count(model),
            structure=structure,input=inp,output=out,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            epochs=None,updates=None,origin='Existing published Brain-HIVE pretrained weights; original training updates unknown',frozen=True))
    contract=json.loads((WORK/'data/component_input_contract_v1.json').read_text())
    source=next(x for x in contract['subjects'][0]['sources'] if x['role'].startswith('oof'))
    cache=torch.load(source['condition_cache_path'],weights_only=True,map_location='cpu',mmap=True)
    position=source['groups']['component_train'][0]['cache_row'];c=cache['condition'][position:position+1]
    with torch.inference_mode():a=models['fusion_encoder'].adapter(c);p=models['proj'](a)
    assert c.shape==a.shape==(1,1024) and p.shape==(1,4,2048)
    interface=dict(actual_OOF_condition_source=source['condition_cache_path'],actual_cache_row=position,
        c_shape=list(c.shape),A_shape=list(a.shape),P_shape=list(p.shape),
        c_norm=float(c.norm()),A_norm=float(a.norm()),P_norm=float(p.norm()),CPU_float32=True,
        retrieval_or_generation_quality_claimed=False)
    from safetensors import safe_open
    attention=ASSETS/'prior_H14/attn_adapter/diffusion_pytorch_model.safetensors'
    with safe_open(str(attention),framework='pt',device='cpu') as f:
        learned_IP_parameters=sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())
    assert learned_IP_parameters==340787200
    rows.append(dict(module='native_IP',role='shared',subject='all',class_name='IPAttnAdapterModel/IPAttnProcessor2_0',
        parameters=learned_IP_parameters,structure='70 native UNet cross-attention processors, separate trained K/V',
        input='4x2048 IP tokens',output='UNet cross-attention residual',checkpoint=str(attention),
        checkpoint_sha256=sha(attention),epochs=None,updates=None,
        origin='Existing published Brain-HIVE pretrained weights; original training updates unknown',frozen=True))
    reg=json.loads((ROOT/'configs/ATM_public_decoder_registration_20261020_v1.json').read_text())
    config_root=Path(reg['generator_asset_root'])/'sdxl_config'
    from diffusers import UNet2DConditionModel,AutoencoderKL
    from transformers import CLIPTextConfig,CLIPTextModel,CLIPTextModelWithProjection
    definitions=[('Turbo_UNet',UNet2DConditionModel,config_root/'unet/config.json'),
        ('Turbo_VAE',AutoencoderKL,config_root/'vae/config.json'),
        ('Turbo_text1',CLIPTextModel,config_root/'text_encoder/config.json'),
        ('Turbo_text2',CLIPTextModelWithProjection,config_root/'text_encoder_2/config.json')]
    payload=next(x for x in reg['generator_assets'] if 'sdxl_turbo_fp16' in x['path'])
    for label,cls,config in definitions:
        with torch.device('meta'):
            model=cls(CLIPTextConfig.from_json_file(str(config))) if label.startswith('Turbo_text') else cls.from_config(json.loads(config.read_text()))
        rows.append(dict(module=label,role='shared',subject='all',class_name=cls.__name__,parameters=count(model),
            structure='Pinned native SDXL-Turbo config, exact meta parameter count; no weight allocations',
            input='native pipeline',output='native pipeline',checkpoint=payload['path'],
            checkpoint_sha256=payload['sha256'],epochs=None,updates=None,
            origin='Existing stabilityai/sdxl-turbo weights; config measured now, full payload hash references launch verification',frozen=True))
    with safe_open(payload['path'],framework='pt',device='cpu') as f:
        header={prefix:sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys() if k.startswith(prefix))
            for prefix in ['model.diffusion_model.','first_stage_model.']}
    by_name={r['module']:r for r in rows}
    assert header['model.diffusion_model.']==by_name['Turbo_UNet']['parameters']
    assert header['first_stage_model.']==by_name['Turbo_VAE']['parameters']
    recipe=json.loads((ROOT/'configs/BrainHIVE_bounded_main_recipe_v1.json').read_text())
    for risk in [False,True]:
        arm='risk' if risk else 'ordinary';model=BoundedConditionAdapter(predict_risk=risk)
        assert count(model)==recipe['trainable_parameters'][arm]
        checkpoint=WORK/'components/bounded_v1'/arm/'component_final.pt'
        progress=WORK/'components/bounded_v1'/arm/'progress.json'
        live=json.loads(progress.read_text()) if progress.exists() else None
        rows.append(dict(module='component_'+arm,role='new',subject='all',class_name='BoundedConditionAdapter',
            parameters=count(model),structure='1024->256 GELU->1024 zero-init direction; risk arm adds256->1',
            input='EEG raw1024 + own learned output affine, no image target at inference',output='same raw1024 support; max radius0.25',
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint) if checkpoint.exists() else None,
            epochs=3,updates=live.get('updates') if live else 0,
            origin='This campaign from scratch; final checkpoint exists only after fixed24807 updates',frozen=False))
    record=dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        status='ACTUAL40_PARENT_WEIGHTS_PRIOR_MODULES_TURBO_HEADER_AND_COMPONENT_CLASSES_AUDITED_CPU',
        source_sha256=sha(__file__),models=rows,input_binding_sha256=sha(binding),
        EEG=dict(selected_channels=inputs['selected_channels'],electrodes=17,time_samples=250,
            train_repeats=4,protected_V_repeats=4,official_native_benchmark_repeats=80,
            preprocessing=inputs['input_preprocessing'],added_zscore=inputs['additional_per_channel_zscore']),
        native_interface=interface,native_generator=dict(model='stabilityai/sdxl-turbo',revision=payload['revision'],
            resolution=512,steps=4,CFG=0,IP_scale=1,weights_dtype='float16; training autocast bfloat16',
            native_IP_parameters_are_additional_to_plain_UNet=True,T_only_visual_F_is_not_used_at_normal_inference=True),
        actual_sampler_parameters=sum(by_name[n]['parameters'] for n in ['Turbo_UNet','Turbo_VAE','Turbo_text1','Turbo_text2','native_IP']),
        target_F_not_in_sampler=True,table_rows_not_to_be_summed_across_40_subject_role_alternatives=True,
        actual_Turbo_header_numel=header,legacy_REF_PRIOR_DiffusionPriorUNet_is_not_current_BrainHIVE=True,
        controlled_capacity_evidence=False,new_GPU_jobs=0,new_images=0,new_candidate_configurations=0,
        scientific_goal_complete=False)
    write(output,record)
    csv_path=ROOT/'audit/BrainHIVE_current_model_inventory_CPU_v19.csv'
    assert not csv_path.exists()
    with csv_path.open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status=record['status'],models=len(rows),module_counts={r['module']:r['parameters'] for r in rows if r['module']!='EEG'},sampler_parameters=record['actual_sampler_parameters'])),flush=True)


if __name__=='__main__':main()
