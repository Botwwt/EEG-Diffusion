"""Lock all400 EEG-only JVP directions and matched controls before image scores."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import brainhive_temporal_image_screen_20261028_v1 as U


def run(spec_path):
    start=time.monotonic();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    spec=U.read(spec_path);assert spec['status']=='D37_FIXED_THREE_ARM_IMAGE_SCREEN_REGISTERED'
    for item in spec['source_pins']:U.check(item)
    output=U.B/'prepared_image_v1';assert not output.exists();output.mkdir(parents=True)
    import brainhive_paired_study_contract_20261025_v1 as C
    manifest=U.ROOT/'audit/diagnostic_memory_recovery_20261025_v3/isolated_runtime_manifest.json'
    C.setup_runtime(dict(runtime_manifest=dict(path=str(manifest),sha256=U.sha(manifest))))
    from brainhive_native_models_v1 import prior_models
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_temporal_conditional_attention_20261028_v1 import temporal_tangents,frozen_prior_temporal_tokens
    def memory():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
    initial=memory()
    def budget():assert time.monotonic()-start<550 and memory()-initial<10<<30
    panel=U.read(U.check(spec['panel']));queries,groups=U.queries(panel)
    mapping=U.read(U.check(spec['EEG_mapping']));assert mapping['Dcheck_queries']==queries
    eeg=np.load(U.check(spec['EEG400']),allow_pickle=False);assert eeg.shape==(400,17,250) and eeg.dtype==np.float32 and np.isfinite(eeg).all()
    old=np.load(U.check(spec['baseline_conditions']),allow_pickle=False);assert old.shape==(400,3,1024) and old.dtype==np.float32
    rows=[json.loads(x) for x in U.check(spec['baseline_manifest']).read_text().splitlines()];assert len(rows)==1200
    for i,q in enumerate(queries):
        for a,arm in enumerate(U.BASE_ARMS):
            row=rows[i*3+a];assert all(row[k]==q[k] for k in U.IDENTITY) and row['arm']==arm
            assert U.byte_sha(old[i,a])==row['condition_sha256']
    parent=torch.load(U.check(spec['parent_checkpoint']),weights_only=True,map_location='cpu');state=parent['model']
    frozen=U.read(U.check(spec['ordinary_frozen_spec']));trained=torch.load(U.check(frozen['checkpoint']),weights_only=True,map_location='cpu')
    assert trained['calibration']['selected_strength']==frozen['selected_strength']==1
    ordinary=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False);ordinary.load_state_dict(trained['model'],strict=True)
    gamma=state['block.layers.0.norm.weight'][None];beta=state['block.layers.0.norm.bias'][None]
    models=prior_models('cpu',include_attention=False)
    def prior(c):return models['proj'](models['fusion_encoder'].adapter(ordinary(c,gamma.expand(len(c),-1),beta.expand(len(c),-1),gate_mode='fixed')[0]))
    pieces=[];native_max=ordinary_max=relative_max=center_max=budget_max=0.
    for lo in range(0,400,32):
        budget();x=torch.tensor(eeg[lo:lo+32].reshape(-1,4250));c,tangent,_=temporal_tangents(x,state)
        cached=torch.tensor(old[lo:lo+32,0])
        native_max=max(native_max,float((c-cached).abs().max()));relative_max=max(relative_max,float(((c-cached).norm(dim=1)/cached.norm(dim=1)).max()))
        with torch.no_grad():current=ordinary(cached,gamma.expand(len(cached),-1),beta.expand(len(cached),-1),gate_mode='fixed')[0]
        ordinary_max=max(ordinary_max,float((current-torch.tensor(old[lo:lo+len(x),1])).abs().max()))
        tokens,directions,_=frozen_prior_temporal_tokens(cached,tangent.detach(),prior)
        assert torch.isfinite(directions).all()
        center_max=max(center_max,float(directions.sum(2).abs().max()))
        ratio=directions.square().sum(-1).mean(-1).sqrt()/tokens.norm(dim=-1)
        budget_max=max(budget_max,float(ratio.max()));pieces.append(directions.numpy())
    # Existing native replay engineering tolerance, never a score-based fit.
    assert native_max<=1e-3 and relative_max<=1e-4 and ordinary_max<=1e-3
    assert budget_max<=.250002 and center_max<2e-4
    directions=np.concatenate(pieces);assert directions.shape==(400,4,5,2048)
    donor=U.control_map(groups);rms=np.sqrt(np.square(directions,dtype=np.float64).sum(-1).mean(-1))
    scale=np.divide(rms,rms[donor],out=np.zeros_like(rms),where=rms[donor]>1e-12).astype(np.float32)
    wrong=directions[donor]*scale[:,:,None,None]
    wrong_rms=np.sqrt(np.square(wrong,dtype=np.float64).sum(-1).mean(-1));matched_error=float(np.abs(wrong_rms-rms).max())
    assert matched_error<1e-4 and all(queries[i]['concept_id']==queries[int(j)]['concept_id'] for i,j in enumerate(donor))
    for name,value in [('ordinary_conditions400.npy',old[:,1].copy()),('native_conditions400.npy',old[:,0].copy()),
                       ('temporal_tokens400.npy',directions),('same_concept_donor400.npy',donor),('wrong_RMS_scale400.npy',scale)]:U.array(output/name,value)
    U.write(output/'query_and_control_map.json',dict(queries=queries,donor_query_ids=[queries[int(j)]['query_id'] for j in donor],
        control='Same-concept stimulus derangement, each native token matched to query correct-direction temporal RMS; original ordinary condition retained'))
    value=dict(status='D37_ALL400_TEMPORAL_DIRECTIONS_AND_MATCHED_CONTROLS_LOCKED_BEFORE_IMAGES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        spec=U.ref(spec_path),preparation_source=U.ref(__file__),execution_repair=U.ref(U.ROOT/'configs/temporal_conditional_attention_20261028_v1/CPU_preparation_repair_v2.json'),query_ids=[q['query_id'] for q in queries],arms=list(U.ARMS),native_parent_role='full_T',
        EEG_physical_repeats=4,raw_EEG_shape=list(eeg.shape),all1200_original_condition_row_SHAs_replayed=True,
        native_CPU_vs_cached_condition_max_error=native_max,native_relative_max_error=relative_max,
        ordinary_CPU_vs_cached_max_error=ordinary_max,actual_frozen_ordinary_A_P_directional_derivatives=2000,
        maximum_direction_time_sum=center_max,maximum_relative_token_RMS_budget=budget_max,
        wrong_vs_correct_RMS_max_absolute_error=matched_error,fixed_radius=.25,added_learning_parameters=0,SGD_updates=0,
        no_Dfit_or_Dcheck_visual_features_loaded=True,no_image_effect_scores_read=True,new_GPU_jobs=0,new_images=0,
        original_ordinary_SHA=frozen['checkpoint']['sha256'],ordinary_strength=1,
        files={p.stem:U.ref(p) for p in output.iterdir() if p.is_file()},seconds=time.monotonic()-start,peak_extra_CPU_bytes=memory()-initial)
    budget();assert not torch.cuda.is_initialized();U.write(output/'completion.json',value)
    print(json.dumps({k:value[k] for k in ['status','seconds','native_CPU_vs_cached_condition_max_error','ordinary_CPU_vs_cached_max_error','maximum_relative_token_RMS_budget','peak_extra_CPU_bytes']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();run(a.spec)
