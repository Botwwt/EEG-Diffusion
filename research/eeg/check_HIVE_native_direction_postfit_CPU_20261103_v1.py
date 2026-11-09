"""Actual legacy encoder probes plus independent fixed-gallery and mixed-metric checks."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,signal,time
from pathlib import Path
import numpy as np
from PIL import Image
import brainhive_native_direction_image_contract_20261103_v1 as I
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();signal.alarm(350);started=time.monotonic();s,train,e,old,base,G,H,C=I.setup(a.protocol);out=Path(s['output_root'])/'CPU_preflight';assert not out.exists();out.mkdir(parents=True);before=I.resources(s,train,e,C)
 import torch
 from brainhive_bounded_component_v1 import BoundedConditionAdapter
 import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
 import score_low_level as L
 from audit_HIVE_native_direction_final_inputs_CPU_20261103_v1 import recompute_native_and_ordinary
 from audit_HIVE_native_direction_final_scores_CPU_20261103_v1 import independent_ranks,independent_low
 import brainhive_native_direction_readout_20261103_v1 as V
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();prep=C.read(Path(e['output_root'])/'inputs/completion.json');arrays=dict(np.load(C.checked(prep['arrays']),allow_pickle=False));head_state=torch.load(C.checked(base['ordinary']),weights_only=True,map_location='cpu')['model'];head=BoundedConditionAdapter().eval().requires_grad_(False);head.load_state_dict(head_state,strict=True);native_error=0.;head_error=0.
 for si,mp in enumerate(e['full_models']):
  state=torch.load(C.checked(mp['model']),weights_only=True,map_location='cpu')['model'];x=torch.from_numpy(arrays['EEG'][8*si]);actual,changed=recompute_native_and_ordinary(x,state,head_state);expected=torch.from_numpy(arrays['conditions_CPU'][8*si]);native_error=max(native_error,float((actual-expected).abs().max()));assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
  with torch.inference_mode():other=head(actual,state['block.layers.0.norm.weight'].expand(4,-1),state['block.layers.0.norm.bias'].expand(4,-1))[0]
  head_error=max(head_error,float((changed-other).abs().max()));assert torch.allclose(changed,other,atol=1e-6,rtol=1e-6)
 gallery=[dict(stimulus_id=str(i),concept_id=str(i//10)) for i in range(400)];eye=np.eye(400,768,dtype=np.float32);rng=np.random.default_rng(1729);random=rng.normal(size=(400,768)).astype(np.float32)
 for n in (80,400):
  for features in (eye[:n],random[:n]):
   lhs=S.metric_arrays(features,eye,gallery[:n],gallery);rhs=independent_ranks(features,eye,gallery[:n],gallery);assert all(np.max(abs(lhs[key]-rhs[:,j]))<1e-12 for j,key in enumerate(V.DINO_METRICS))
 fixture=np.zeros((80,8));allvalues={arm:fixture.copy() for arm in I.OLD_ARMS+I.ARMS};zero,_=V.summarize(allvalues,gallery[:80]);assert zero['decision']['action']=='EXPAND400'
 allvalues['directional'][:,1]=-.03;allvalues['directional'][:,4]=.02;partial,_=V.summarize(allvalues,gallery[:80]);assert partial['decision']['action']=='EXPAND400'
 allvalues['directional'][:,4]=-.02;harm,_=V.summarize(allvalues,gallery[:80]);assert harm['decision']['action']=='STOP_EXPANSION_JOINT_HARM'
 all400={arm:np.zeros((400,8)) for arm in I.OLD_ARMS+I.ARMS};all400['directional'][:,1]=.02;all400['directional'][:,4]=.01;good,_=V.summarize(all400,gallery);assert good['decision']['action']=='PROMISING_MATCHED_INCREMENT_REQUIRES_MORE_SUBJECTS_AND_FULL_METRICS'
 low_fixture=np.random.default_rng(1729).integers(0,256,size=(512,512,3),dtype=np.uint8);Image.fromarray(low_fixture).save(out/'synthetic_pixel_fixture.png');rgb,y=L.pixels(out/'synthetic_pixel_fixture.png');expected=np.array([L.pixcorr(rgb,rgb),L.ssim_y(y,y),0.]);observed=independent_low(out/'synthetic_pixel_fixture.png',rgb,y);assert np.max(abs(expected-observed))<1e-10
 asset=I.assets(s,base,C);assert len(asset['panel'])==400 and len(asset['old_manifest'])==1200
 dependency=I.dependency(s,train,C);assert dependency['status'] in ('WAIT_SAME_ORIGINAL_TRAINING_OWNER','READY_ALL30_FIXED_FINALS_AND_ACTUAL_OWNER_EXIT')
 done=dict(status='PASS_CPU_SOURCE_MATH_REAL10_PARENT_PROBES_AND_FIXED_GALLERY_POLICY_NOT_POSTFIT_GPU_RESULT',UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),protocol=C.ref(a.protocol),source=C.ref(__file__),
  real10_parent_T_probe_cells=40,maximum_manual_native_abs_error=native_error,maximum_manual_ordinary_abs_error=head_error,
  independent_rank_fixtures=4,zero_case_expands=True,single_metric_harm_with_positive_detail_expands=True,joint_harm_stops_expansion_only=True,positive_matched_case_preserves_future_confirmation_requirement=True,
  independent_lowlevel_fixture_error=float(np.max(abs(expected-observed))),actual400_panel_join_and1200_baseline_contract=True,actual_dependency=dependency,
  scope='Real old encoder T inputs and pure scoring/policy math; final trained encoders, postfit host pixels and actual evaluation remain pending. Synthetic fixture is engineering data, not a reconstruction.',
  new_GPU_jobs=0,new_research_images=0,NEW_contacted=False,seconds=time.monotonic()-started,resources_before=before,resources_after=I.resources(s,train,e,C))
 assert not torch.cuda.is_initialized() and time.monotonic()-started<s['CPU_stage_seconds'];G.write(out/'completion.json',done);print(json.dumps(done),flush=True)
if __name__=='__main__':main()
