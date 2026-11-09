"""Four discarded real-T8 optimizer steps, CPU state/RNG/readout checks."""
import argparse,datetime,hashlib,json,sys,time
from pathlib import Path


def main(protocol_path):
    p=json.loads(Path(protocol_path).read_bytes());assert p['host']=='old' and p['new_GPU_calls']==0
    def checked(item):
        path=Path(item['path'])
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256']
        return path
    for item in p['input_pins']:checked(item)
    sys.path.insert(0,p['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=p['runtime_manifest']));sys.path.insert(0,str(Path(protocol_path).parent))
    import numpy as np
    import torch
    import brainhive_native_models_v1 as N
    from brainhive_joint_student_engineering_Torch_20261030_v1 import make_optimizer,student_step,capture_engineering_state,restore_engineering_state,assert_tree_equal,posterior_conditions
    torch.set_num_threads(2);torch.manual_seed(1729);started=time.monotonic()
    prior=dict(np.load(checked(p['prior']),allow_pickle=False));inputs=np.load(checked(p['T8_inputs']),allow_pickle=False)
    binding=json.loads(checked(p['actual_input_bindings']).read_bytes());assert binding['selected_indices']==list(range(46,63))
    raw=torch.load(checked(p['EEG']),map_location='cpu',weights_only=True,mmap=True)
    eeg=raw[torch.from_numpy(inputs['array_rows'])][:,binding['selected_indices'],:].clone()
    original=torch.load(checked(p['parent']),map_location='cpu',weights_only=True)
    def fresh():
        model=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval();model.load_state_dict(original['model'],strict=True)
        model.block.layers[0].norm.requires_grad_(False)
        log_scale=torch.nn.Parameter(torch.tensor(np.log(float(inputs['subject_strengths'][0])),dtype=torch.float32))
        optimizer=make_optimizer(model,log_scale);return model,log_scale,optimizer
    model,log_scale,optimizer=fresh();initial_log_scale=log_scale.detach().clone();affine={name:value.detach().clone() for name,value in model.block.layers[0].norm.named_parameters()}
    weight=torch.from_numpy(prior['F_weight']);bias=torch.from_numpy(prior['F_bias']);radius=float(prior['reference_pre_affine_radius'])
    teacher=torch.from_numpy(inputs['visual_latent_mean']*(100000./101022.)).requires_grad_(True)
    steps=[]
    for unused in range(4):steps.append(student_step(model,log_scale,optimizer,eeg,teacher,prior,weight,bias))
    assert teacher.grad is None and all(torch.equal(affine[name],value.detach()) for name,value in model.block.layers[0].norm.named_parameters())
    assert any(not torch.equal(model.state_dict()[name],value) for name,value in original['model'].items())
    assert not torch.equal(log_scale.detach(),initial_log_scale)
    states=list(optimizer.state.values());assert len(states)==5 and all(int(state['step'])==4 for state in states)
    snapshot=capture_engineering_state(model,log_scale,optimizer,4,p['parent']['sha256'])
    path=Path(p['output_root'])/'ENGINEERING_ONLY_state_roundtrip.pt';torch.save(snapshot,path)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    second,second_scale,second_optimizer=fresh();restore_engineering_state(restored,second,second_scale,second_optimizer)
    again=capture_engineering_state(second,second_scale,second_optimizer,4,p['parent']['sha256']);assert_tree_equal(snapshot,again)
    # State identity, not future CPU/CUDA optimizer-trajectory identity.
    model.eval();second.eval()
    with torch.no_grad():condition=model(eeg);second_condition=second(eeg)
    assert torch.equal(condition,second_condition)
    torch_RNG=torch.get_rng_state().clone();numpy_RNG=np.random.get_state();seeds=[1729+int(row) for row in inputs['array_rows']]
    prediction,samples,records=posterior_conditions(condition,log_scale,prior,weight,bias,radius,seeds)
    replay,replay_samples,replay_records=posterior_conditions(second_condition,second_scale,prior,weight,bias,radius,seeds)
    assert torch.equal(prediction,replay) and torch.equal(samples,replay_samples) and records==replay_records
    assert torch.equal(torch_RNG,torch.get_rng_state())
    after=np.random.get_state();assert numpy_RNG[0]==after[0] and np.array_equal(numpy_RNG[1],after[1]) and numpy_RNG[2:]==after[2:]
    assert not torch.cuda.is_initialized()
    checked(p['parent']);checked(p['EEG'])
    result=dict(status='PASS_REAL_T8_JOINT_STUDENT_TEMPORARY_TRANSACTION_CPU',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),queries=8,temporary_EEG_optimizer_updates=4,formal_EEG_optimizer_updates=0,optimizer_tensor_states=5,source_original_logit_scale=float(original['logit_scale']),engineering_initial_scale_from_frozenT_MLE=float(inputs['subject_strengths'][0]),engineering_initial_scale_after_float32_log_parameterization=float(np.exp(float(initial_log_scale.double()))),student_scale_after_four_temporary_steps=float(np.exp(float(log_scale.detach().double()))),steps=steps,teacher_moment='privilegedF_PowerSpherical_engineering_input_not_realGoptimizedq',original_fullT_parent_init_not_full_chain_OOF=True,original_output_affine_frozen=True,original_source_checkpoint_unchanged=True,serialized_model_optimizer_temperature_CPU_RNG_roundtrip_exact=True,future_optimizer_trajectory_equality_not_claimed=True,all8_private_latent_draws_and_native_F_conditions_replayed_exactly=True,global_Torch_and_numpy_RNG_unchanged_by_readout=True,checkpoint_role='ENGINEERING_ONLY_REJECT_IN_FORMAL_TRAINING_OR_CONDITION_BANKS',checkpoint_bytes=path.stat().st_size,checkpoint_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),new_scientific_models=0,new_research_images=0,new_GPU_calls=0,CUDA_initialized=False,NEW_contacted=False,seconds=time.monotonic()-started)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);main(parser.parse_args().protocol)
