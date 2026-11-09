"""Independent CPU native formula, global Procrustes value, and full saved forwards."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_native_coordinate_gauge_20261028_v1 as V

def main():
    p=argparse.ArgumentParser()
    for k in ('completion','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(2)
    done=U.read(a.completion);assert done['status']=='D24_ALL9_NATIVE_FOLDED_COORDINATE_MODELS_FULL_ROLE_T_COMPLETE';spec=U.read(U.checked(done['spec']))
    for item in spec['source_pins']:U.checked(item)
    states=torch.load(U.checked(done['model']),weights_only=True,map_location='cpu')['models'];rot=torch.load(U.checked(done['rotations']),weights_only=True,map_location='cpu')
    vg=rot['visual_weight'];vb=rot['visual_bias']
    from safetensors import safe_open
    with safe_open(str(U.checked(spec['visual_prior_weights'])),framework='pt',device='cpu') as f:
        assert torch.equal(vg,f.get_tensor('vision_model.block.layers.0.norm.weight')) and torch.equal(vb,f.get_tensor('vision_model.block.layers.0.norm.bias'))
    raw=torch.from_numpy(np.array(np.load(U.checked(spec['T_mean4']),mmap_mode='r'),copy=True).reshape(13230,4250));truth=torch.from_numpy(np.load(U.checked(spec['targets']),allow_pickle=False));visual=V.remove_affine(truth,vg,vb)
    prepared=U.read(U.checked(spec['prepared_EEG']));mapping=U.read(U.checked(prepared['mapping']));dev=torch.from_numpy(np.array(np.load(U.checked(prepared['Dfit']),mmap_mode='r')[0],copy=True).reshape(820,4250));dc=torch.from_numpy(np.array(np.load(U.checked(prepared['Dcheck']),mmap_mode='r'),copy=True).reshape(400,4250))
    oof=np.load(U.checked(done['OOF_T_conditions']),allow_pickle=False);df=np.load(U.checked(done['Dfit_conditions']),allow_pickle=False);lock=U.read(U.checked(done['condition_lock']));checkbank=np.load(U.checked(lock['conditions']),allow_pickle=False)
    assert lock['ordinary_applied'] is False and lock['model']==done['model'] and lock['Dcheck_target_F_or_images_used'] is False
    assert lock['query_ids']==[q['query_id'] for q in mapping['Dcheck_queries']] and tuple(lock['arms'])==V.ARMS
    index=[json.loads(s) for s in U.checked(spec['index']).read_text().splitlines()];b=V.zero_mean_basis(1024)
    maxima=dict(weight_abs=0.,forward_relative=0.,orthogonality_abs=0.,fit_objective_optimality_gap=0.)
    verified=0;proofs=[]
    def infer(e,s):
        with torch.inference_mode():return torch.cat([V.predict_from_state(v,s) for v in e.split(256)])
    def compare(e,s,expected):
        nonlocal verified
        got=infer(e,s);reference=torch.from_numpy(expected);error=float(((got-reference).double().norm(dim=1)/reference.double().norm(dim=1).clamp_min(1e-12)).max());assert error<3e-5
        maxima['forward_relative']=max(maxima['forward_relative'],error);verified+=len(got)
    for role,item in spec['roles'].items():
        old=torch.load(U.checked(item['checkpoint']),weights_only=True,map_location='cpu')['model'];native=infer(raw,old);u=V.remove_affine(native,old[V.PREFIX+'norm.weight'],old[V.PREFIX+'norm.bias']);ids=item['fit_positions'];held=item['held_positions'];x=u[ids];yy=visual[ids];perm=V.same_concept_derangement([index[i]['concept_id'] for i in ids])
        assert {index[i]['concept_id'] for i in ids}.isdisjoint({index[i]['concept_id'] for i in held})
        for ai,arm in enumerate(V.ARMS):
            r=rot['rotations'][role][arm];fold=V.fold_native(old,r,vg,vb);actual=states[role][arm];assert set(fold)==set(actual)
            we=max(float((fold[k]-actual[k]).abs().max()) for k in fold);assert we<3e-7;maxima['weight_abs']=max(maxima['weight_abs'],we)
            orth=float((r@r.T-torch.eye(1024)).abs().max());assert orth<1e-8;maxima['orthogonality_abs']=max(maxima['orthogonality_abs'],orth)
            if ai<2:
                y=yy if ai==0 else yy[perm];cross=(y@b).T@(x@b);best=torch.linalg.svdvals(cross).sum();attained=torch.trace((b.T@r@b).T@cross);gap=float((best-attained).abs()/best.abs().clamp_min(1.));assert gap<1e-7;maxima['fit_objective_optimality_gap']=max(maxima['fit_objective_optimality_gap'],gap)
            else:assert torch.equal(r,torch.eye(1024,dtype=torch.float64))
            if held:compare(raw[held],actual,oof[held,ai])
            if role=='full':compare(dev,actual,df[:,ai]);compare(dc,actual,checkbank[:,ai])
            proofs.append(dict(role=role,arm=arm,fit_stimuli=len(ids),held_stimuli=len(held),folded_weight_abs_error=we))
    assert verified==43350 and not torch.cuda.is_initialized()
    for item in spec['source_pins']:U.checked(item)
    result=dict(status='D24_ALL43350_NATIVE_FOLDED_CONDITIONS_AND9_MODELS_CPU_VERIFIED',source=U.ref(__file__),completion=U.ref(a.completion),all_sources_unchanged=True,all9_folded_native_weights_verified=True,all6_closed_form_objectives_globally_optimal_to_tolerance=True,all43350_saved_native_forwards_verified=True,normal_inference_other_subject_EEG=False,normal_inference_target_access=False,ordinary_applied=False,proofs=proofs,maxima=maxima,new_fit=0,new_images=0,GPU_initialized=False,seconds=time.monotonic()-start)
    U.atomic(a.output/'summary.json',result);print(json.dumps({k:result[k] for k in ['status','seconds','maxima']}))

if __name__=='__main__':main()
