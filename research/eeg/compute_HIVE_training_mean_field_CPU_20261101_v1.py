"""Average frozen raw MLP responses over all real165400 TRAIN OOF rows."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import brainhive_training_mean_field_CPU_20261101_v1 as M

def run(protocol):
    started=time.monotonic();s,parent,F=M.checked_setup(protocol)
    import numpy as np
    import torch
    import brainhive_exposed_fullfit_component_inputs_CPU_20261101_v1 as I
    torch.set_num_threads(2);torch.manual_seed(1729)
    assert not torch.cuda.is_initialized()
    from check_HIVE_training_mean_field_CPU_20261101_v1 import math_checks
    local_math=math_checks(torch)
    M.write(M.OUT/'actual_worker_synthetic_math_checks.json',local_math)
    inputs=I.load_verified(parent['input_bank'],parent['asset_protocol'])
    assert inputs['own_source_affines_verified'] and inputs['join_cells_verified']==165400*1024
    ids=inputs['affine_ids'];assert np.array_equal(np.bincount(ids,minlength=20),np.full(20,8270))
    trace_indices=sorted(int(j) for number in range(20) for j in np.flatnonzero(ids==number)[:2])
    assert len(trace_indices)==len(set(trace_indices))==40
    results={}
    for role in M.ROLES:
        root=M.OUT/role;assert not root.exists();root.mkdir()
        source=M.source_head(s,parent,role)
        value=torch.load(M.checked(source),map_location='cpu',weights_only=True)
        assert value['arm']==role and value['seed']==1729 and value['updates']==31014
        assert value.get('strength',1.0)==1.0 and value.get('radius',0.25)==0.25
        state=value['model'];assert set(state)=={'trunk.0.weight','trunk.0.bias','direction.weight','direction.bias'}
        assert sum(v.numel() for v in state.values())==525568
        assert all(v.device.type=='cpu' and v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values())
        total=torch.zeros(1024,dtype=torch.float64);squares=total.clone()
        digest=hashlib.sha256();chunks=[];traces={};count=0
        with torch.no_grad():
            for lo in range(0,165400,s['chunk']):
                hi=min(lo+s['chunk'],165400);aid=ids[lo:hi]
                c=torch.from_numpy(np.array(inputs['condition'][lo:hi],copy=True))
                weight=torch.from_numpy(inputs['weight'][aid].copy());bias=torch.from_numpy(inputs['bias'][aid].copy())
                raw=M.raw_output(c,weight,bias,state,torch)
                assert raw.shape==(hi-lo,1024) and raw.dtype==torch.float32 and torch.isfinite(raw).all()
                part=raw.double().sum(0);total+=part;squares+=raw.double().square().sum(0)
                digest.update(raw.contiguous().numpy().tobytes());count+=hi-lo
                chunks.append(dict(lo=lo,hi=hi,raw_output_FP32_SHA256=hashlib.sha256(raw.numpy().tobytes()).hexdigest()))
                for index in trace_indices:
                    if lo<=index<hi:
                        offset=index-lo
                        traces[index]=dict(condition=c[offset].clone(),weight=weight[offset].clone(),bias=bias[offset].clone(),
                                           source_raw=raw[offset].clone(),affine_id=int(aid[offset]))
        assert count==165400 and len(chunks)==324 and chunks[-1]['hi']-chunks[-1]['lo']==24
        assert len(traces)==40
        mean,variance=M.mean_and_variance(total,squares,count,torch)
        mean32=mean.float()
        derived=M.constant_state(state,mean32,torch)
        assert M.ref(source['path'])==source
        np.save(root/'mean_FP64.npy',mean.numpy(),allow_pickle=False)
        np.save(root/'mean_FP32.npy',mean32.numpy(),allow_pickle=False)
        np.save(root/'source_raw_variance_FP64.npy',variance.numpy(),allow_pickle=False)
        with (root/'component_mean_field.pt').open('xb') as stream:
            torch.save(dict(model=derived,arm=role+'_training_mean_field',source_component=source,
                source_training_updates=31014,new_training_updates=0,parameters=525568,
                effective_raw_field_coefficients=1024,training_seed=1729,strength=1.,radius=.25,
                mean_rows=165400,input_bank=parent['input_bank'],asset_protocol=parent['asset_protocol'],
                parent_source_training_cost_inherited=True,derivation_protocol=M.ref(protocol),
                no_R80_inputs_or_image_score_payloads_used=True),stream)
        trace=dict(indices=trace_indices,affine_ids=[traces[i]['affine_id'] for i in trace_indices],
                   **{key:torch.stack([traces[i][key] for i in trace_indices]) for key in ('condition','weight','bias','source_raw')})
        with (root/'fixed40_native_OOF_trace.pt').open('xb') as stream:torch.save(trace,stream)
        result=dict(status='COMPLETE_ALL165400_TRAIN_ONLY_RAW_FIELD_MEAN',role=role,
                    protocol=M.ref(protocol),source_component=source,input_bank=parent['input_bank'],asset_protocol=parent['asset_protocol'],
                    queries=count,subjects=10,concepts=1654,images=16540,affine_groups=20,
                    affine_group_counts=np.bincount(ids,minlength=20).tolist(),
                    mean_FP64=M.ref(root/'mean_FP64.npy'),mean_FP32=M.ref(root/'mean_FP32.npy'),
                    source_raw_variance_FP64=M.ref(root/'source_raw_variance_FP64.npy'),
                    mean_component=M.ref(root/'component_mean_field.pt'),fixed40_trace=M.ref(root/'fixed40_native_OOF_trace.pt'),
                    raw_FP32_outputs_stream_SHA256=digest.hexdigest(),chunks=chunks,
                    accumulation='FP32 CPU raw MLP, FP64 sequential512-row sums, fixed row order',
                    source_training_updates_inherited=31014,new_training_updates=0,
                    no_R80_inputs_or_image_score_payloads_used=True,GPU_initialized=False,
                    GPU_generation_or_method_efficacy_admitted=False,seconds=time.monotonic()-started)
        M.write(root/'completion.json',result);results[role]=M.ref(root/'completion.json')
    assert not torch.cuda.is_initialized()
    M.write(M.OUT/'means_completion.json',dict(status='COMPLETE_BOTH_TRAIN_ONLY_RAW_MEANS_NOT_GPU_OR_EFFICACY',
        protocol=M.ref(protocol),roles=results,new_training_updates=0,new_GPU_jobs=0,new_reconstruction_images=0,
        no_R80_inputs_or_image_score_payloads_used=True,seconds=time.monotonic()-started))
    print(json.dumps(dict(status='COMPLETE_BOTH_TRAIN_ONLY_RAW_MEANS_NOT_GPU_OR_EFFICACY',queries_per_head=165400,seconds=time.monotonic()-started)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
