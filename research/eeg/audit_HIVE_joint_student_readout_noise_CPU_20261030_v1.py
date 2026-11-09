"""Independent CPU state/EEG forward/readout geometry/T-critic arithmetic.

Does not replay the CUDA optimizer trajectory or independently re-encode G.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path


def ref(path):
    path=Path(path)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest)


def main(protocol_path,binding_path):
    s=json.loads(Path(protocol_path).read_bytes());bound=json.loads(Path(binding_path).read_bytes());root=Path(s['output_root'])
    done=json.loads((root/'completion.json').read_bytes());assert done['status']=='COMPLETE_JOINT_STUDENT_READOUT_NOISE_T8_ENGINEERING'
    sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']));sys.path.insert(0,str(Path(protocol_path).parent))
    import numpy as np
    import torch
    import brainhive_native_models_v1 as N
    from brainhive_conditional_vmf_sampler_CPU_20261030_v1 import sample_conditional_mixture
    torch.set_num_threads(2);started=time.monotonic()
    for item in s['payloads']+bound['input_pins']:assert ref(item['path'])['sha256']==item['sha256']
    files={path.name:ref(path) for path in root.iterdir() if path.is_file()}
    models=torch.load(root/'ENGINEERING_ONLY_all3_working_models.pt',map_location='cpu',weights_only=True)
    snapshot=torch.load(root/'ENGINEERING_ONLY_correct_state_roundtrip.pt',map_location='cpu',weights_only=True)
    assert snapshot['role']=='ENGINEERING_ONLY_REJECT_IN_FORMAL_TRAINING_OR_CONDITION_BANKS'
    assert snapshot['temporary_updates']==4 and len(snapshot['optimizer']['state'])==5
    assert all(int(value['step'])==4 and torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all() for value in snapshot['optimizer']['state'].values())
    assert snapshot['CPU_RNG'].dtype==torch.uint8 and len(snapshot['CUDA_RNG'])==1 and snapshot['CUDA_RNG'][0].dtype==torch.uint8
    inputs=np.load(s['T8_inputs']['path'],allow_pickle=False);prior=dict(np.load(s['prior']['path'],allow_pickle=False));data=np.load(root/'all3_predictive_and_noise_engineering.npz',allow_pickle=False)
    raw=torch.load(s['EEG']['path'],map_location='cpu',weights_only=True,mmap=True);binding=json.loads(Path(s['actual_input_bindings']['path']).read_bytes())
    eeg=raw[torch.from_numpy(inputs['array_rows'])][:,binding['selected_indices'],:].clone()
    original=torch.load(s['parent']['path'],map_location='cpu',weights_only=True,mmap=True)['model']
    h=np.ones(1024)/32;h[-1]-=1;h/=np.linalg.norm(h);H=np.eye(1024)-2*np.outer(h,h)
    errors=[];moments=[]
    for record in done['arms']:
        arm=record['arm'];working=models[arm]
        assert working['role']=='ENGINEERING_ONLY_REJECT_IN_FORMAL_TRAINING_OR_CONDITION_BANKS' and working['temporary_updates']==4
        assert working['source_checkpoint_SHA256']==s['parent']['sha256']
        model=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval();model.load_state_dict(working['model'],strict=True)
        for key in ('block.layers.0.norm.weight','block.layers.0.norm.bias'):assert torch.equal(original[key],working['model'][key])
        if arm=='correct_EEG':
            assert torch.equal(snapshot['log_scale'],working['log_scale'])
            for name,value in snapshot['model'].items():assert torch.equal(value,working['model'][name])
        with torch.no_grad():forward=model(eeg).numpy()
        raw_error=float(np.max(abs(forward-data[arm+'_raw_condition'])));assert raw_error<3e-5
        # Replay conditional draw from the EXACT stored GPU raw condition,
        # separating independent EEG-forward rounding from discrete sampling.
        c=data[arm+'_raw_condition'];u=(c.astype(np.float64)-prior['F_bias'])/prior['F_weight'];u-=u.mean(axis=1,keepdims=True)
        coordinates=(u@H)[:,:1023];mu=coordinates/np.linalg.norm(coordinates,axis=1,keepdims=True)
        log_scale=float(working['log_scale']);assert log_scale==float(data[arm+'_log_scale'])
        scale=float(np.exp(log_scale));sample_errors=[];condition_errors=[]
        for j,predicted in enumerate(record['predictive_engineering']):
            sampled,draw=sample_conditional_mixture(prior['weights'],prior['means'],prior['concentrations'],mu[j],scale,1,np.random.default_rng(predicted['condition']['condition_seed']))
            assert int(draw['component'][0])==predicted['condition']['component'] and draw['proposals']==predicted['condition']['proposals']
            error=float(np.max(abs(sampled[0]-data[arm+'_latent_sample'][j])));assert error<1e-10;sample_errors.append(error)
            pre=np.concatenate((sampled,np.zeros((1,1))),axis=1)@H
            condition=prior['F_bias']+prior['F_weight']*float(prior['reference_pre_affine_radius'])*pre
            error=float(np.max(abs(condition-data[arm+'_native_condition'][j])));assert error<3e-5;condition_errors.append(error)
        errors.append(dict(arm=arm,independent_EEG_forward_maxabs=raw_error,independent_predictive_latent_maxabs=max(sample_errors),independent_native_F_geometry_maxabs=max(condition_errors)))
    maximum_critic_error=0.
    for record in done['T_only_fixed_q_noise_diagnostics']:
        features=data[f"{record['arm']}_{record['position']}_{record['version']}_features"]
        cosines=features@data['target_features'][record['position']]
        maximum_critic_error=max(maximum_critic_error,float(np.max(abs(cosines-record['T_only_CLIP_cosines']))))
    assert maximum_critic_error<1e-6 and not torch.cuda.is_initialized()
    assert all(ref(root/name)==item for name,item in files.items())
    result=dict(status='PASS_JOINT_STUDENT_T8_INDEPENDENT_CPU_STATE_FORWARD_READOUT_AND_NOISE_ARITHMETIC',queries=8,arms=3,temporary_updates_per_arm=4,formal_models_or_images=0,errors=errors,maximum_T_training_critic_arithmetic_error=maximum_critic_error,original_files_unchanged=True,CUDA_initialized=False,new_GPU_calls=0,not_replayed_CUDA_optimizer_trajectory_or_G_features=True,seconds=time.monotonic()-started)
    path=root/'independent_CPU_audit.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--binding',required=True);args=parser.parse_args();main(args.protocol,args.binding)
