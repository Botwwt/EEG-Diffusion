"""Temporary original-EEG/scalar optimizer and predictive readout engineering.

Import only after the pinned runtime is configured. No adapter is introduced.
Snapshots are explicitly engineering-only, never formal training checkpoints.
"""
import copy
import numpy as np
import torch
from brainhive_EEG_joint_density_M_step_CPU_20261030_v1 import Torch_joint_first_order_loss
from brainhive_semantic_teacher_Torch_20261029_v1 import F_native_coordinates,native_F_condition
from brainhive_conditional_vmf_sampler_CPU_20261030_v1 import sample_conditional_mixture


def make_optimizer(model,log_scale,network_lr=1e-4,scale_lr=1e-3):
    assert all(not parameter.requires_grad for parameter in model.block.layers[0].norm.parameters())
    return torch.optim.AdamW([dict(params=[parameter for parameter in model.parameters() if parameter.requires_grad],lr=network_lr,weight_decay=.01),dict(params=[log_scale],lr=scale_lr,weight_decay=0.)],betas=(.9,.999),eps=1e-8)


def student_step(model,log_scale,optimizer,eeg,teacher_moment,prior,F_weight,F_bias):
    model.train();optimizer.zero_grad(set_to_none=True)
    condition=model(eeg)
    coordinates=F_native_coordinates(condition,F_weight,F_bias)
    values=Torch_joint_first_order_loss(coordinates,teacher_moment,log_scale,prior['weights'],prior['means'],prior['concentrations'])
    loss=values.mean();assert torch.isfinite(loss)
    loss.backward()
    trainable=[parameter for parameter in model.parameters() if parameter.requires_grad]
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
    assert log_scale.grad is not None and torch.isfinite(log_scale.grad)
    network_norm=float(torch.sqrt(sum(parameter.grad.double().square().sum() for parameter in trainable)))
    scale_gradient=float(log_scale.grad.detach())
    assert network_norm>0
    norm=float(torch.nn.utils.clip_grad_norm_(trainable+[log_scale],1.))
    optimizer.step()
    assert torch.isfinite(log_scale) and all(torch.isfinite(parameter).all() for parameter in trainable)
    return dict(theta_dependent_cross_entropy=float(loss.detach()),network_gradient_norm=network_norm,log_scale_gradient=scale_gradient,total_gradient_norm_before_clipping=norm,scale_after_update=float(np.exp(float(log_scale.detach().double()))))


def _CPU_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:_CPU_tree(item) for key,item in value.items()}
    if isinstance(value,list):return [_CPU_tree(item) for item in value]
    if isinstance(value,tuple):return tuple(_CPU_tree(item) for item in value)
    return copy.deepcopy(value)


def capture_engineering_state(model,log_scale,optimizer,updates,source_SHA):
    state=np.random.get_state()
    return dict(role='ENGINEERING_ONLY_REJECT_IN_FORMAL_TRAINING_OR_CONDITION_BANKS',source_checkpoint_SHA256=source_SHA,temporary_updates=updates,model=_CPU_tree(model.state_dict()),log_scale=log_scale.detach().cpu().clone(),optimizer=_CPU_tree(optimizer.state_dict()),model_training=bool(model.training),CPU_RNG=torch.get_rng_state().clone(),CUDA_RNG=[value.cpu().clone() for value in torch.cuda.get_rng_state_all()] if next(model.parameters()).device.type=='cuda' else None,numpy_global_RNG=dict(kind=state[0],state=state[1].tolist(),position=state[2],has_gaussian=state[3],cached_gaussian=state[4]))


def restore_engineering_state(state,model,log_scale,optimizer):
    assert state['role']=='ENGINEERING_ONLY_REJECT_IN_FORMAL_TRAINING_OR_CONDITION_BANKS'
    model.load_state_dict(state['model'],strict=True)
    with torch.no_grad():log_scale.copy_(state['log_scale'])
    optimizer.load_state_dict(state['optimizer']);model.train(state['model_training'])
    torch.set_rng_state(state['CPU_RNG'])
    if state['CUDA_RNG'] is not None:
        assert next(model.parameters()).device.type=='cuda'
        torch.cuda.set_rng_state_all(state['CUDA_RNG'])
    state=state['numpy_global_RNG'];np.random.set_state((state['kind'],np.asarray(state['state'],dtype=np.uint32),state['position'],state['has_gaussian'],state['cached_gaussian']))


def assert_tree_equal(first,second):
    assert type(first) is type(second),(type(first),type(second))
    if torch.is_tensor(first):assert first.dtype==second.dtype and first.shape==second.shape and torch.equal(first,second)
    elif isinstance(first,dict):
        assert first.keys()==second.keys()
        for key in first:assert_tree_equal(first[key],second[key])
    elif isinstance(first,(tuple,list)):
        assert len(first)==len(second)
        for a,b in zip(first,second):assert_tree_equal(a,b)
    else:assert first==second


def posterior_conditions(raw_condition,log_scale,prior,F_weight,F_bias,radius,seeds):
    """Normal readout inputs contain no visual target or per-image q table."""
    assert len(raw_condition)==len(seeds)
    with torch.no_grad():
        coordinates=F_native_coordinates(raw_condition,F_weight,F_bias)
        unit=(coordinates/coordinates.norm(dim=-1,keepdim=True)).cpu().numpy()
    scale=float(np.exp(float(log_scale.detach().double())));samples=[];records=[]
    for direction,seed in zip(unit,seeds):
        generator=np.random.default_rng(int(seed));before=copy.deepcopy(generator.bit_generator.state)
        value,information=sample_conditional_mixture(prior['weights'],prior['means'],prior['concentrations'],direction,scale,1,generator)
        samples.append(value[0]);records.append(dict(condition_seed=int(seed),component=int(information['component'][0]),proposals=information['proposals'],log_evidence=information['log_evidence'],private_RNG_before=before,private_RNG_after=copy.deepcopy(generator.bit_generator.state)))
    value=torch.as_tensor(np.stack(samples),device=raw_condition.device,dtype=torch.float64)
    condition=native_F_condition(value,F_weight,F_bias,radius).float()
    assert torch.isfinite(condition).all()
    return condition,value,records
