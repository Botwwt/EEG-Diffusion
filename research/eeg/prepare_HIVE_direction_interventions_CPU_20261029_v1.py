"""Export all400 two-arm conditions, no CUDA, target F, image or score read."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
from brainhive_equal_angle_direction_CPU_20261029_v1 import interventions,mathematical_checks
def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def read(p):return json.loads(Path(p).read_bytes())
def checked(x):assert ref(x['path'])['sha256']==x['sha256'];return Path(x['path'])
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def run(path):
    start=time.monotonic();s=read(path);assert s['status']=='REGISTERED_ORDINARY_EQUAL_ANGLE_DIRECTION_INTERVENTION' and s['host']=='old'
    for x in s['source_pins']:checked(x)
    out=Path(s['output_root'])/'prepared';assert not out.exists();out.mkdir(parents=True);checks=mathematical_checks()
    sys.path.insert(0,s['repository_root']);import brainhive_native_models_v1 as N
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    import torch
    torch.set_num_threads(2)
    old=read(checked(s['D09_completion']));inputs=read(checked(old['inputs']));panel=read(checked(inputs['panel']));queries=panel['Dcheck'];assert len(queries)==400
    bank=np.load(checked(old['check_conditions']),allow_pickle=False);assert bank.shape==(400,3,1024) and bank.dtype==np.float32
    cache=torch.load(checked(inputs['native_cache']),weights_only=True,map_location='cpu',mmap=True);parent=torch.load(checked(inputs['native_checkpoint']),weights_only=True,map_location='cpu')['model'];lookup={int(v):i for i,v in enumerate(cache['array_rows'])}
    native=cache['condition'][[lookup[q['array_row']] for q in queries]];assert np.array_equal(native.numpy(),bank[:,0])
    gamma=parent['block.layers.0.norm.weight'];beta=parent['block.layers.0.norm.bias'];state=torch.load(checked(s['ordinary']),weights_only=True,map_location='cpu')
    model=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False);model.load_state_dict(state['model'],strict=True);assert state['calibration']['selected_strength']==1
    mean_state=torch.load(checked(s['T_mean_direction']),weights_only=True,map_location='cpu');assert mean_state['queries']==132300 and mean_state['checkpoint']==s['ordinary']
    u=(native-beta)/gamma
    with torch.inference_mode():
        d=model.direction(model.trunk(u));d-=d.mean(-1,keepdim=True)
        replay=model(native,gamma.expand_as(native),beta.expand_as(native),gate_mode='fixed')[0].numpy()
    drift=float(np.linalg.norm(replay.astype(np.float64)-bank[:,1],axis=1).max());relative=float((np.linalg.norm(replay-bank[:,1],axis=1)/np.linalg.norm(bank[:,1],axis=1)).max());assert relative<1e-5
    result,proof=interventions(bank[:,0],bank[:,1],gamma.numpy(),beta.numpy(),d.numpy(),mean_state['centered_pre_tangent_direction_mean'].numpy())
    # Independent explicit normalized-step formula reconstructs every output.
    ud=(bank[:,0].astype(np.float64)-beta.numpy().astype(np.float64))/gamma.numpy().astype(np.float64);md=mean_state['centered_pre_tangent_direction_mean'].numpy();dd=d.numpy().astype(np.float64);error=0.
    for j,direction in enumerate((np.broadcast_to(md,dd.shape),dd-md)):
        q=direction-ud*np.einsum('ij,ij->i',direction,ud)[:,None]/np.einsum('ij,ij->i',ud,ud)[:,None]
        tangent_norm=np.linalg.norm(ud,axis=1)*np.tan(proof['original_angles_radians']);step=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)*tangent_norm[:,None]
        check_u=(ud+step)*np.linalg.norm(ud,axis=1,keepdims=True)/np.linalg.norm(ud+step,axis=1,keepdims=True);check=beta.numpy()+gamma.numpy()*check_u
        fallback=proof['reports'][('population_direction','varying_direction')[j]]['zero_component_or_zero_angle_fallback_queries'];check[fallback]=bank[fallback,1]
        error=max(error,float(np.abs(check-result[:,j]).max()))
    assert error<2e-6 and not torch.cuda.is_initialized() and time.monotonic()-start<120
    np.save(out/'conditions400x2.npy',result,allow_pickle=False);write(out/'queries.json',queries)
    completion=dict(status='COMPLETE_ALL400_EQUAL_ANGLE_DIRECTION_CONDITIONS_CPU_VERIFIED',protocol=ref(path),conditions=ref(out/'conditions400x2.npy'),queries=ref(out/'queries.json'),arms=proof['arms'],mathematical_checks=checks,equal_angle_and_norm_proof=proof,independent_normalized_step_max_error=error,CPU_original_head_vs_cached_GPU_relative_error=relative,CPU_original_head_vs_cached_GPU_absolute_rowL2=drift,mean_uses_only_T=True,query_target_F_image_or_score_access=False,new_GPU_jobs=0,new_model_updates=0,normal_inference_other_query_EEG=False,normal_inference_still_uses_old_head_amplitude_and_query_geometry=True,seconds=time.monotonic()-start)
    write(out/'completion.json',completion);print(json.dumps(completion))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
