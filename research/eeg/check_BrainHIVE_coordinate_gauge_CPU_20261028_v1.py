"""Critical algebra and failure-mode checks before any actual D24 fitting."""
import argparse,json
from pathlib import Path
import torch
import brainhive_native_coordinate_gauge_20261028_v1 as V

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(1729)
    d=9;n=120;dtype=torch.float64;b=V.zero_mean_basis(d);eye=torch.eye(d,dtype=dtype)
    results={}
    def check(name,value):assert value,name;results[name]=True
    check('zero_mean_basis_orthonormal_and_perp_one',torch.allclose(b.T@b,eye[:-1,:-1],atol=1e-12) and b.sum(0).abs().max()<1e-12)
    q,_=torch.linalg.qr(torch.randn(d-1,d-1,dtype=dtype));r=b@q@b.T+torch.ones(d,d,dtype=dtype)/d
    x=torch.randn(n,d,dtype=dtype);x-=x.mean(1,keepdim=True);y=x@r.T
    recovered,info=V.procrustes_rotation(x,y)
    check('known_full_rank_rotation_recovered',torch.allclose(recovered,r,atol=1e-10) and info['objective_after']<1e-20)
    noisy=y+.2*torch.randn_like(y);rr,inf=V.procrustes_rotation(x,noisy)
    check('procrustes_global_solution_beats_identity_on_fit',inf['objective_after']<=inf['objective_before']+1e-12)
    check('all_pair_gram_and_norm_preserved',torch.allclose((x@rr.T)@(x@rr.T).T,x@x.T,atol=1e-10))
    s={'proj_in.weight':torch.randn(d,13,dtype=dtype),'proj_in.bias':torch.randn(d,dtype=dtype),V.PREFIX+'ff.w1.weight':torch.randn(17,d,dtype=dtype),V.PREFIX+'ff.w2.weight':torch.randn(d,17,dtype=dtype),V.PREFIX+'norm.weight':torch.rand(d,dtype=dtype)+.5,V.PREFIX+'norm.bias':torch.randn(d,dtype=dtype)}
    e=torch.randn(80,13,dtype=dtype);g=torch.rand(d,dtype=dtype)+.7;bb=torch.randn(d,dtype=dtype)
    native=V.predict_from_state(e,s);u=V.remove_affine(native,s[V.PREFIX+'norm.weight'],s[V.PREFIX+'norm.bias'])
    folded=V.fold_native(s,r,g,bb);expected=(u@r.T)*g+bb;actual=V.predict_from_state(e,folded)
    check('all_coupled_weights_native_forward_equivalent',torch.allclose(actual,expected,atol=1e-10,rtol=1e-10))
    check('identity_and_original_affine_exact_state',all(torch.equal(v,s[k]) for k,v in V.fold_native(s,eye,s[V.PREFIX+'norm.weight'],s[V.PREFIX+'norm.bias']).items()))
    broken={k:v.clone() for k,v in folded.items()};broken[V.PREFIX+'ff.w1.weight']=s[V.PREFIX+'ff.w1.weight']
    check('uncoupled_residual_change_detectably_wrong',(V.predict_from_state(e,broken)-expected).norm()>1.)
    random,_=torch.linalg.qr(torch.randn(d,d,dtype=dtype))
    try:V.fold_native(s,random,g,bb);accepted=True
    except AssertionError:accepted=False
    check('non_mean_preserving_rotation_rejected',not accepted)
    perm=V.same_concept_derangement([j//10 for j in range(30)])
    check('control_stimuli_distinct_and_same_concept',all(i//10==v//10 and i!=v for i,v in enumerate(perm)))
    result=dict(status='D24_ALL9_CRITICAL_COORDINATE_GAUGE_CPU_CHECKS_PASSED',checks=results,GPU_initialized=torch.cuda.is_initialized(),actual_models_fitted=0,new_images=0)
    assert not result['GPU_initialized'];a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
