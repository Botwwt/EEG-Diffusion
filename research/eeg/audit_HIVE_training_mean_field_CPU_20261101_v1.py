"""Independent full TRAIN mean replay and bounded-interface CPU audit; no images."""
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
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    torch.set_num_threads(2);torch.manual_seed(1729)
    inputs=I.load_verified(parent['input_bank'],parent['asset_protocol'])
    means=M.read(M.OUT/'means_completion.json')
    assert means['protocol']==M.ref(protocol) and set(means['roles'])==set(M.ROLES)
    ids=inputs['affine_ids'];results={}
    for role in M.ROLES:
        folder=M.OUT/role;done=M.read(M.checked(means['roles'][role]))
        assert done['queries']==165400 and done['affine_group_counts']==[8270]*20
        assert done['source_component']==M.source_head(s,parent,role)
        original=torch.load(M.checked(done['source_component']),weights_only=True,map_location='cpu')
        derivative=torch.load(M.checked(done['mean_component']),weights_only=True,map_location='cpu')
        assert derivative['new_training_updates']==0 and derivative['source_training_updates']==31014
        assert derivative['mean_rows']==165400 and derivative['no_R80_inputs_or_image_score_payloads_used']
        assert torch.equal(derivative['model']['trunk.0.weight'],original['model']['trunk.0.weight'])
        assert torch.equal(derivative['model']['trunk.0.bias'],original['model']['trunk.0.bias'])
        assert torch.count_nonzero(derivative['model']['direction.weight'])==0
        mean64=np.load(M.checked(done['mean_FP64']),allow_pickle=False)
        mean32=np.load(M.checked(done['mean_FP32']),allow_pickle=False)
        assert mean64.shape==mean32.shape==(1024,) and mean64.dtype==np.float64 and mean32.dtype==np.float32
        assert np.array_equal(mean64.astype(np.float32),mean32)
        assert np.array_equal(derivative['model']['direction.bias'].numpy(),mean32)
        model=BoundedConditionAdapter().eval();model.load_state_dict(original['model'],strict=True)
        control=BoundedConditionAdapter().eval();control.load_state_dict(derivative['model'],strict=True)
        trace=torch.load(M.checked(done['fixed40_trace']),map_location='cpu',weights_only=True)
        assert len(trace['indices'])==len(set(trace['indices']))==40
        assert sorted(trace['affine_ids'])==sorted(list(range(20))*2)
        for j,index in enumerate(trace['indices']):
            assert np.array_equal(trace['condition'][j].numpy(),inputs['condition'][index])
            assert trace['affine_ids'][j]==int(ids[index])
            assert np.array_equal(trace['weight'][j].numpy(),inputs['weight'][ids[index]])
            assert np.array_equal(trace['bias'][j].numpy(),inputs['bias'][ids[index]])
        raw_digest=hashlib.sha256();native_mean=torch.zeros(1024,dtype=torch.float64)
        largest_norm_error=largest_angle=0.0;rows=0
        with torch.no_grad():
            for number,part in enumerate(done['chunks']):
                lo,hi=part['lo'],part['hi'];assert lo==number*512 and hi==min(lo+512,165400)
                aid=ids[lo:hi]
                c=torch.from_numpy(np.array(inputs['condition'][lo:hi],copy=True))
                gamma=torch.from_numpy(inputs['weight'][aid].copy());beta=torch.from_numpy(inputs['bias'][aid].copy())
                u=(c-beta)/gamma
                # Original nn.Linear/GELU module, independently of raw_output.
                raw=model.direction(model.trunk(u))
                raw_bytes=raw.numpy().tobytes();raw_digest.update(raw_bytes)
                assert hashlib.sha256(raw_bytes).hexdigest()==part['raw_output_FP32_SHA256']
                native_mean+=raw.double().sum(0);rows+=hi-lo
                m=torch.from_numpy(mean32).expand_as(u)
                centered=m-m.mean(1,keepdim=True);r2=u.square().sum(1,keepdim=True)
                tangent=centered-(centered*u).sum(1,keepdim=True)/r2*u
                delta=.25*r2.sqrt()*tangent/(r2+tangent.square().sum(1,keepdim=True)).sqrt()
                scale=r2.sqrt()/(u+delta).norm(dim=1,keepdim=True)
                updated=scale*(u+delta)
                norm_error=((updated.norm(dim=1)-u.norm(dim=1)).abs()/u.norm(dim=1)).max()
                angle=torch.atan(delta.norm(dim=1)/u.norm(dim=1)).max()
                largest_norm_error=max(largest_norm_error,float(norm_error));largest_angle=max(largest_angle,float(angle))
            assert rows==165400 and raw_digest.hexdigest()==done['raw_FP32_outputs_stream_SHA256']
            assert np.array_equal((native_mean/165400).numpy(),mean64)
            assert largest_norm_error<2e-6 and largest_angle<=float(np.arctan(.25))+2e-6
            actual_raw=model.direction(model.trunk((trace['condition']-trace['bias'])/trace['weight']))
            assert torch.allclose(actual_raw,trace['source_raw'],**s['numerical_guard'])
            control_raw=control.direction(control.trunk((trace['condition']-trace['bias'])/trace['weight']))
            assert torch.equal(control_raw,torch.from_numpy(mean32).expand_as(control_raw))
            disabled=control(trace['condition'],trace['weight'],trace['bias'],enabled=False)[0]
            assert torch.equal(disabled,trace['condition'])
            original_bias=control.direction.bias.clone();control.direction.bias.zero_()
            zero=control(trace['condition'],trace['weight'],trace['bias'])[0]
            assert torch.equal(zero,trace['condition']);control.direction.bias.copy_(original_bias)
            # Independent FP64/native-model pass in AFFINE-GROUP order, with
            # a different384-row partition and NumPy accumulation.
            model=model.double();group_sums=[]
            for group in range(20):
                where=np.flatnonzero(ids==group);assert len(where)==8270
                group_sum=np.zeros(1024,np.float64)
                for lo in range(0,8270,384):
                    index=where[lo:lo+384]
                    c=torch.from_numpy(np.array(inputs['condition'][index],copy=True)).double()
                    gamma=torch.from_numpy(inputs['weight'][ids[index]]).double()
                    beta=torch.from_numpy(inputs['bias'][ids[index]]).double()
                    raw=model.direction(model.trunk((c-beta)/gamma))
                    group_sum+=raw.numpy().sum(axis=0,dtype=np.float64)
                group_sums.append(group_sum)
            independent_mean=np.stack(group_sums).sum(axis=0)/165400
        absolute=np.abs(independent_mean-mean64);tolerance=s['numerical_guard']['atol']+s['numerical_guard']['rtol']*np.abs(mean64)
        assert np.all(absolute<=tolerance)
        assert M.ref(done['source_component']['path'])==done['source_component']
        assert not torch.cuda.is_initialized()
        results[role]=dict(source_component=done['source_component'],mean_component=done['mean_component'],
            all165400_native_FP32_raw_outputs_and_chunk_hashes_replayed=True,
            FP32_mean_accumulation_bit_exact=True,all165400_independent_FP64_group_order_forward_replayed=True,
            independent_FP64_mean_maximum_absolute_error=float(absolute.max()),
            independent_FP64_mean_maximum_tolerance_fraction=float(np.max(absolute/tolerance)),
            all20_source_affines_and40_trace_rows_checked=True,all165400_manual_mean_field_norms_checked=True,
            maximum_relative_norm_error=largest_norm_error,maximum_retraction_angle_radians=largest_angle,
            fixed40_actual_native_constant_raw_and_zero_disabled_paths_pass=True,
            not_all165400_mean_controls_run_through_native_adapter=True,
            no_UNet_or_images_or_R80_inputs=True)
    result=dict(status='PASS_BOTH_FULL_TRAIN_RAW_MEANS_AND_CPU_ONLY_INTERFACE_AUDIT',protocol=M.ref(protocol),
                input_bank=parent['input_bank'],roles=results,queries_per_role=165400,source_affines=20,
                new_training_updates=0,new_GPU_jobs=0,new_reconstruction_images=0,
                no_R80_inputs_or_image_score_payloads_used=True,GPU_generation_admitted=False,
                no_new_innovation_or_SOTA_or_CCF_A_claim=True,seconds=time.monotonic()-started)
    M.write(M.OUT/'full_CPU_audit.json',result)
    print(json.dumps(dict(status=result['status'],queries_per_role=165400,seconds=result['seconds'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
