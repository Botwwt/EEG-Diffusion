"""Actual T8 EEG forward and parameter-gradient check; no optimizer or CUDA.

The fixed teacher moment below is an engineering input, not a fitted semantic
teacher. The real generator E-step remains a separate prerequisite.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time


def checked(item):
    path=Path(item['path'])
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256']
    return path


def main(protocol_path):
    specification=json.loads(Path(protocol_path).read_bytes())
    assert specification['host']=='old' and specification['new_GPU_calls']==0
    for item in specification['input_pins']:
        checked(item)
    sys.path.insert(0,specification['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=specification['runtime_manifest']))
    sys.path.insert(0,str(Path(protocol_path).parent))
    import numpy as np
    import torch
    import brainhive_native_models_v1 as N
    from brainhive_semantic_teacher_Torch_20261029_v1 import F_native_coordinates,exact_moment_M_step
    torch.set_num_threads(2)
    started=time.monotonic()
    prior=np.load(checked(specification['prior']),allow_pickle=False)
    inputs=np.load(checked(specification['T8_inputs']),allow_pickle=False)
    metadata=json.loads(checked(specification['T8_metadata']).read_bytes())
    raw=torch.load(checked(specification['EEG']),map_location='cpu',weights_only=True,mmap=True)
    bindings=json.loads(checked(specification['actual_input_bindings']).read_bytes())
    selected=specification['selected_channel_indices']
    assert selected==list(range(46,63)) and len(selected)==17
    # Bind the explicit selection to the original physical-channel registration.
    assert bindings['selected_indices']==selected
    assert bindings['selected_channels']==specification['selected_channel_names']
    assert bindings['subjects']['sub-01']['train_sha256']==specification['EEG']['sha256']
    assert specification['selected_channel_names']==['P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2']
    weight=torch.from_numpy(prior['F_weight'])
    bias=torch.from_numpy(prior['F_bias'])
    scale=float(inputs['subject_strengths'][0])
    teacher_moments=torch.from_numpy(inputs['visual_latent_mean']*(100000./101022.))
    models={}
    records=[]
    for j,item in enumerate(metadata):
        source=item['subject01_source']
        pin=dict(path=source['checkpoint_path'],sha256=source['checkpoint_SHA256'])
        if source['role'] not in models:
            payload=torch.load(checked(pin),map_location='cpu',weights_only=True)
            model=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval()
            model.load_state_dict(payload['model'],strict=True)
            models[source['role']]=model
        model=models[source['role']]
        row=item['array_row']
        eeg=raw[row:row+1,selected,:]
        # GPU-exported cache versus an independent original CPU float32 forward.
        model.float()
        with torch.no_grad():
            condition=model(eeg)
        cache=torch.from_numpy(inputs['individual_raw_condition'][0,j:j+1]).float()
        cache_error=float((condition-cache).abs().max())
        assert cache_error<3e-5
        # FP64 is used only for the derivative check, never saved as a new model.
        model.double().zero_grad(set_to_none=True)
        model.block.layers[0].norm.requires_grad_(False)
        condition=model(eeg.double())
        coordinates=F_native_coordinates(condition,weight,bias)
        loss=exact_moment_M_step(coordinates,teacher_moments[j:j+1],prior['weights'],prior['means'],prior['concentrations'],scale).sum()
        loss.backward()
        gradients={name:parameter.grad for name,parameter in model.named_parameters() if parameter.requires_grad}
        assert all(value is not None and torch.isfinite(value).all() for value in gradients.values())
        norm=float(torch.sqrt(sum(value.square().sum() for value in gradients.values())))
        assert norm>0
        parameter=model.proj_in.weight
        direction=parameter.grad.detach()/parameter.grad.detach().norm()
        analytic=float((parameter.grad*direction).sum())
        original=parameter.detach().clone()
        differences=[]
        for step in (1e-4,1e-5,1e-6):
            values=[]
            for sign in (1.,-1.):
                with torch.no_grad():
                    parameter.copy_(original+sign*step*direction)
                    value=exact_moment_M_step(F_native_coordinates(model(eeg.double()),weight,bias),teacher_moments[j:j+1],prior['weights'],prior['means'],prior['concentrations'],scale).sum()
                values.append(float(value))
            numeric=(values[0]-values[1])/(2*step)
            differences.append(dict(step=step,numeric=numeric,error=abs(numeric-analytic)))
        with torch.no_grad():
            parameter.copy_(original)
        numeric=differences[-1]['numeric']
        error=differences[-1]['error']
        print(json.dumps(dict(diagnostic_row=row,analytic=analytic,finite_difference_steps=differences)),file=sys.stderr,flush=True)
        assert error<1e-5 and error/max(abs(analytic),1e-12)<1e-5
        assert error<=max(1e-7,differences[0]['error']*.1)
        assert all(parameter.grad is None for parameter in model.block.layers[0].norm.parameters())
        records.append(dict(array_row=row,OOF_role=source['role'],cached_original_float32_forward_maxabs=cache_error,existing_EEG_trainable_gradient_norm=norm,input_projection_directional_derivative=analytic,independent_finite_difference=numeric,finite_difference_error=error,finite_difference_steps=differences,original_h1e4_strict_absolute_guard_pass=differences[0]['error']<1e-5))
        assert time.monotonic()-started<90
    for item in metadata:
        checked(dict(path=item['subject01_source']['checkpoint_path'],sha256=item['subject01_source']['checkpoint_SHA256']))
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_ACTUAL_T8_OOF_EEG_PARAMETER_M_STEP_CPU',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=records,queries=8,OOF_models=2,teacher='fixed_actual_F_PowerSpherical_moment_for_engineering_only_not_semantic_teacher',existing_EEG_parameter_gradients_checked=True,new_EEG_optimizer_updates=0,new_models_saved=0,real_semantic_teacher_fitted=False,all_source_checkpoint_SHA_unchanged=True,CUDA_initialized=False,new_GPU_calls=0,NEW_contacted=False,seconds=time.monotonic()-started)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    main(parser.parse_args().protocol)
