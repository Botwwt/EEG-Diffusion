"""D20 full source/coverage/control/noise/held-error and IP weight-bound replay."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_error_pair_lora_20261027_v2 as V
from brainhive_bounded_component_v1 import BoundedConditionAdapter
from fit_BrainHIVE_error_pair_lora_GPU_20261027_v1 import wrong_positions
R=Path(__file__).resolve().parent
Q=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/rapid_iteration_20261027_v1');B=Q/'candidates/D20_error_pair_lora';A=R/'audit/error_pair_lora_20261027_v1'

def main():
    started=time.monotonic();out=A/'fit_CPU_audit';assert not out.exists();out.mkdir();torch.set_num_threads(2)
    spec_path=R/'configs/error_pair_lora_20261027_v1/fit_spec.json';spec=U.read(spec_path)
    for x in spec['source_pins']:U.checked(x)
    profile=U.read(U.checked(spec['profile']));assert profile['nonzero_module_counts']==[140]*12 and profile['frozen_weights_unchanged'] and profile['native_and_zero_pixel_parity']
    probe=profile['precision_probe'];assert len(probe['legacy_FP16_zero_gradient_modules'])==1 and probe['direct_BF16_zero_gradient_modules']==[]
    old=U.read(U.checked(spec['D19_fit']));prep=U.read(U.checked(old['prepared_inputs']));mapping=U.read(U.checked(prep['mapping']))
    expected_perm=wrong_positions(mapping['T400']);raw=np.load(U.checked(prep['conditions']['T400']),allow_pickle=False)
    contract=U.read(U.checked(spec['native_input_contract']));s=next(x for x in contract['subjects'] if x['subject']=='sub-01')
    caches={};models={}
    for src in s['sources']:
        if src['role'] not in ('oof0','oof1'):continue
        role=int(src['role'][-1]);caches[role]=torch.load(U.checked({'path':src['condition_cache_path'],'sha256':src['condition_cache_sha256']}),weights_only=True,map_location='cpu',mmap=True)
        models[role]=torch.load(U.checked({'path':src['checkpoint_path'],'sha256':src['checkpoint_sha256']}),weights_only=True,map_location='cpu')['model']
    positions={fold:{int(r):i for i,r in enumerate(cache['array_rows'].tolist())} for fold,cache in caches.items()}
    for i,q in enumerate(mapping['T400']):assert np.array_equal(raw[i],caches[q['fold']]['condition'][positions[q['fold']][q['array_row']]].numpy())
    saved=torch.load(U.checked(spec['ordinary_checkpoint']),weights_only=False,map_location='cpu');model=BoundedConditionAdapter().eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True)
    w=torch.stack([models[q['fold']]['block.layers.0.norm.weight'] for q in mapping['T400']]);b=torch.stack([models[q['fold']]['block.layers.0.norm.bias'] for q in mapping['T400']])
    with torch.no_grad():pred=model(torch.from_numpy(raw.copy()),w,b)[0].numpy()
    exported=np.load(U.checked(profile['ordinary_condition_bank']),allow_pickle=False)
    ordinary_error=float(np.max(np.linalg.norm(pred-exported,axis=1)/np.linalg.norm(exported,axis=1)));assert ordinary_error<1e-6
    math_check=V.cpu_math_check()
    import brainhive_paired_study_contract_20261025_v1 as C
    cfg=C.read(R/'configs/paired_effect_20261025_v1/study_review_v3.json');C.setup_runtime(cfg)
    from safetensors.torch import load_file
    attn=Path('/root/brainhive_reproduction_20261020_v1/assets/prior_H14/attn_adapter');config=U.read(attn/'config.json');weights=load_file(str(attn/'diffusion_pytorch_model.safetensors'))
    index={name:str(i) for i,name in enumerate(config['attn_processor_keys']) if name.endswith('attn2.processor')};assert len(index)==70
    norms={};full_states={};all_logs={};proofs=[];same_initial=[];reference_errors=None;noise_logs=None;normrows=[]
    for arm in V.ARMS:
        p=B/('fit_'+arm);completion=U.read(p/'completion.json');assert completion['spec']==U.ref(spec_path) and completion['coverage_all'] and completion['updates']==200 and completion['unique_T_queries']==400
        assert U.read(A/('fit_round_'+arm)/'completion.json')['all_registered_stages_closed']
        cp=torch.load(U.checked(completion['model']),weights_only=False,map_location='cpu');assert cp['updates']==200 and cp['query_exposures']==400 and cp['coverage'].all() and cp['coverage'].shape==(400,)
        assert len(cp['optimizer']['state'])==280 and all(int(x['step'])==200 for x in cp['optimizer']['state'].values())
        assert cp['ordinary_checkpoint']==spec['ordinary_checkpoint'] and cp['spec']==U.ref(spec_path)
        logs=[json.loads(line) for line in (p/'training_log.jsonl').read_text().splitlines()];train=[x for x in logs if x['stage']=='ONE_COMPLETE_T400_PASS'];assert len(train)==200
        observed=[i for event in train for i in event['indices']];assert sorted(observed)==list(range(400))
        order=np.load(p/'training_order.npy',allow_pickle=False);assert observed==order.tolist()==torch.randperm(400,generator=torch.Generator().manual_seed(20201729)).tolist()
        stream=[(x['indices'],x['t'],x['noise_sha256']) for x in train]
        if noise_logs is None:noise_logs=stream
        else:assert noise_logs==stream
        assert np.array_equal(np.load(p/'sameconcept_negative_permutation.npy',allow_pickle=False),expected_perm)
        assert completion['original_weights_unchanged'] and completion['original_UNet_sha256']==profile['original_UNet_parameter_sha256']
        assert len(cp['model'])==140 and sum(x.numel() for s in cp['model'].values() for x in s.values())==906240
        for name,state in cp['model'].items():
            processor,kind=name.rsplit('.',1);key='attn_processors.'+index[processor]+'.'+kind+'.weight'
            original=weights[key].to(torch.float16).float();delta=state['up']@state['down'];assert delta.shape==original.shape and torch.isfinite(delta).all()
            ratio=float(delta.norm()/original.norm());assert ratio<=.05+1e-6
            normrows.append(dict(arm=arm,module=name,relative_weight_change=ratio,original_shape=list(original.shape)))
        held=[]
        for label in ['initial','fixed_final']:
            e=np.load(p/(label+'_Dfit_errors.npy'),allow_pickle=False);r=np.load(p/(label+'_Dfit_reference_errors.npy'),allow_pickle=False);assert e.shape==r.shape==(200,2) and np.isfinite(e).all()
            if reference_errors is None:reference_errors=r
            else:assert np.array_equal(reference_errors,r)
            if label=='initial':assert np.array_equal(e,r)
            de=torch.from_numpy(e-r);pref=torch.nn.functional.softplus(100*(de[:,0]-de[:,1]))
            event=next(x for x in logs if x['stage']=='FIXED_ALL200_DFIT_LOG_ONLY' and x['label']==label)
            assert abs(event['preference_loss']-float(pref.mean()))<1e-6
            held.append(dict(label=label,positive_MSE=float(e[:,0].mean()),negative_MSE=float(e[:,1].mean()),delta_positive=float(de[:,0].mean()),delta_negative=float(de[:,1].mean()),preference_loss=float(pref.mean())))
        same_initial.append(cp['initial_state_sha256']);full_states[arm]=cp;all_logs[arm]=logs
        proofs.append(dict(arm=arm,completion=U.ref(p/'completion.json'),all400_queries_once=True,optimizer_parameters=280,all_steps200=True,maximum_weight_ratio=max(x['relative_weight_change'] for x in normrows if x['arm']==arm),held_all200=held))
    assert len(set(same_initial))==1
    first=full_states[V.ARMS[0]]
    for cp in full_states.values():
        assert torch.equal(cp['initial_noise_rng'],first['initial_noise_rng']) and torch.equal(cp['final_noise_rng'],first['final_noise_rng'])
    with (out/'all420_low_rank_projection_bounds.csv').open('x',newline='') as f:
        wcsv=csv.DictWriter(f,fieldnames=['arm','module','relative_weight_change','original_shape']);wcsv.writeheader();wcsv.writerows(normrows)
    result=dict(status='D20_ALL_THREE_COMPLETE_T400_FITS_COMMON_RNG_CONTROLS_AND_ALL420_IP_WEIGHT_BOUNDS_CPU_VERIFIED',source=U.ref(__file__),spec=U.ref(spec_path),profile=spec['profile'],proofs=proofs,
        all400_original_OOF_c_exactly_replayed=True,all400_old_ordinary_CPU_relative_error=ordinary_error,ordinary_checkpoint=spec['ordinary_checkpoint'],CPU_math_check=math_check,
        profile_actual_positive_query_positions=[0,10],profile_registered_loaded_positions=[0,10,20,30],profile_last_two_loaded_positions_not_used_for_optimization=True,
        precision_comparison='same actual model/input/noise/loss: one FP16 zero-gradient module, none via direct BF16 master path; not an effect result',
        all200_orders_timesteps_noise_hashes_equal_across_arms=True,common_initial_and_final_noise_RNG=True,all600_held_error_pairs_recomputed=True,
        CPU_UNet_full_forward_replayed=False,CPU_VAE_reencoding_repeated=False,existing_D19_source_latent_audit_reused=True,held_scores_used_for_model_selection=False,new_GPU_jobs=0,new_images=0,
        formal_D_resumed=False,seconds=time.monotonic()-started,independent_confirmation=False,SOTA_established=False)
    U.atomic(out/'summary.json',result);print(json.dumps({k:result[k] for k in ['status','seconds','all400_old_ordinary_CPU_relative_error']}))

if __name__=='__main__':main()
