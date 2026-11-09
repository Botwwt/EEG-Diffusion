"""Independent scalar reductions and CPU64 native fixture; not all-T neural replay."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]=os.environ["MKL_NUM_THREADS"]=os.environ["OPENBLAS_NUM_THREADS"]="2"
os.environ["MPLBACKEND"]="Agg"
import json, sys, time
from pathlib import Path
import numpy as np
import atm_relative_P_diagnostic_core_20261105_v183 as F
plan=json.loads(Path(sys.argv[1]).read_bytes());root=Path(plan["parent_project"])
sys.path.insert(0,str(root/"code"))
import atm_deployment_training_contract_20261105_v179 as C
c=C.setup(root/"code/policy.json");q,K,pq=c["q"],c["K"],c["physical"];out=Path(plan["output_root"])
assert K.ref(__file__)==plan["sources"]["CPU"];before=C.resources(c,True);started=time.monotonic()
done=K.read(out/"GPU_completion.json");assert done["status"]=="COMPLETE_FULL_T_RELATIVE_P_DIAGNOSTIC"
f=np.load(K.checked(done["fixture"]));s=np.load(K.checked(done["statistics"]))
assert np.all(s["coverage"]==1) and np.array_equal(s["T"],np.asarray(q["T_array_rows"]))
order=np.random.default_rng(1739).permutation(1323);pos=(order[:16,None]*10+np.arange(10)).ravel()
assert np.array_equal(s["T"][pos],f["ids"])
num,den=s["numerators"],s["denominators"]
np.testing.assert_allclose(num.sum(-1),np.repeat(den[...,None],2,-1),atol=2e-15,rtol=1e-12)
np.testing.assert_allclose(num[:,:,0],num[:,:,1]+num[:,:,2],atol=2e-15,rtol=1e-12)
assert np.isfinite(num).all() and np.all(num>=0) and np.all(den>0)
# Independent scalar-row orientation and split reduction, without calling repeat_mass.
expected=np.zeros_like(num[pos]);expected_den=np.zeros_like(den[pos]);n=len(pos)
for space in range(2):
    z=f["eval_logits"][space];r4=z[4]
    for orient in range(2):
        s4=r4 if orient==0 else r4.T
        weights=np.exp(s4-s4.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True);weights/=2*n
        for i in range(n):
            for region in range(3):
                selected=np.array([j for j in range(n) if j!=i and (region==0 or (i//10==j//10)==(region==1))])
                expected_den[i,space,region]+=weights[i,selected].sum()
                for control in range(2):
                    p=np.arange(n) if control==0 else (np.arange(n)//10)*10+(np.arange(n)%10+1)%10
                    source=z[:4][:,:,p];source=source if orient==0 else source.transpose(0,2,1)
                    margins=source[:,i,i,None]-source[:,i,selected]
                    for left,right in (((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))):
                        a=margins[list(left)].mean(0);b=margins[list(right)].mean(0)
                        cats=((a>0)&(b>0),(a<0)&(b<0),a*b<0,(a==0)|(b==0))
                        for k,cat in enumerate(cats):expected[i,space,region,control,k]+=weights[i,selected][cat].sum()/3
np.testing.assert_allclose(expected,num[pos],atol=2e-15,rtol=1e-12)
np.testing.assert_allclose(expected_den,den[pos],atol=2e-15,rtol=1e-12)
sys.path[:0]=pq["existing_runtime_paths"]+["/root/atm_official_20261016"]
import torch
from models.atms import ATMS
torch.set_num_threads(2)
base=torch.load(K.checked(q["checkpoint"]),map_location="cpu",weights_only=True,mmap=True)
net=ATMS().double().eval();net.load_state_dict(base["model"],strict=True)
z=np.load(K.checked(q["projector"]))
w,b,g,be=[torch.from_numpy(z[k].astype(np.float64)) for k in
          ("image_embeds.weight","image_embeds.bias","norm.weight","norm.bias")]
def project(x):
    a=(x@w.T+b).reshape(-1,4,2048);d=a-a.mean(-1,keepdim=True)
    return d/torch.sqrt((d*d).mean(-1,keepdim=True)+1e-5)*g+be
def independent(c,y):
    sf=net.logit_scale.exp().clamp_max(100)*torch.nn.functional.normalize(c,dim=1)@torch.nn.functional.normalize(y,dim=1).T
    p,t=project(c),project(y).detach()
    pu=p/torch.linalg.vector_norm(p,dim=-1,keepdim=True);tu=t/torch.linalg.vector_norm(t,dim=-1,keepdim=True)
    sp=net.logit_scale.exp().clamp_max(100).detach()*sum(pu[:,k]@tu[:,k].T for k in range(4))/4
    ce=lambda v:((torch.logsumexp(v,1)-v.diag()).mean()+(torch.logsumexp(v,0)-v.diag()).mean())/2
    return (ce(sf),(p-t).square().mean(),ce(sp)),(sf,sp)
y=torch.from_numpy(f["y"]).double();all_logits=[];outputs=[]
with torch.no_grad():
    for ri in range(5):
        x=torch.from_numpy(f["physical"][:,ri] if ri<4 else f["x"]).double()
        cv=net(x,torch.ones(n,dtype=torch.long));outputs.append(cv.numpy())
        ll,ss=independent(cv,y);all_logits.append(np.stack([v.numpy() for v in ss]))
logerr=float(np.max(abs(np.stack(all_logits,axis=1)-f["eval_logits"])))
prederr=float(np.max(abs(np.stack(outputs)-f["eval_outputs"])))
assert logerr<.001 and prederr<.0001,(logerr,prederr)
cv=net(torch.from_numpy(f["x"]).double(),torch.ones(n,dtype=torch.long))
ll,_=independent(cv,y);params=tuple(net.parameters());gradient_errors=[];norm_errors=[]
for k,loss in enumerate(ll):
    grads=torch.autograd.grad(loss,params,retain_graph=k<2,allow_unused=True)
    v=torch.cat([(torch.zeros_like(p) if a is None else a).flatten() for p,a in zip(params,grads)]).detach().numpy()
    err=float(np.max(abs(v[f["sample_indices"]]-f["eval_gradient_samples"][k])))
    rel=abs(float(v@v)-float(f["eval_gradient_norms"][k]))/max(float(f["eval_gradient_norms"][k]),1e-30)
    assert err<.0001 and rel<.003,(k,err,rel)
    gradient_errors.append(err);norm_errors.append(rel)
assert time.monotonic()-started<850
means=num.sum(0)/den.sum(0)[...,None,None]
gram=s["batch_grams"];batchcos=np.array([F.cosine(v) for v in gram])
finite=K.read(out/"finite_steps.json");assert len(finite)==6
trace=K.read(out/"batch_trace.json");assert len(trace)==83 and sum(v["queries"] for v in trace)==13230
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(15,4.5))
cos=F.cosine(s["global_mapper_gram"])
im=ax[0].imshow(cos,vmin=-1,vmax=1,cmap="coolwarm")
ax[0].set_xticks(range(3),F.LOSSES,rotation=25);ax[0].set_yticks(range(3),F.LOSSES)
for i in range(3):
    for j in range(3):ax[0].text(j,i,f"{cos[i,j]:.3f}",ha="center",va="center")
ax[0].set_title("Full-T mapper gradient cosine")
for j,space in enumerate(("F","P")):
    ax[1].bar(np.arange(3)+.35*j,means[j,:,0,2],width=.35,label=space)
ax[1].set_xticks(np.arange(3)+.175,["all","same concept","other concept"])
ax[1].set_ylabel("Negative-logit derivative mass");ax[1].set_title("Split-repeat sign disagreement");ax[1].legend()
for k,route in enumerate(("F_only","F_plus_P_MSE","F_plus_P_NCE")):
    entries=[v for v in finite if v["route"]==route]
    dif=np.mean([np.array(v["eval_after"])-v["eval_BNmatched_before"] for v in entries],axis=0)
    ax[2].plot(range(3),dif,marker="o",label=route)
ax[2].axhline(0,color="grey",lw=.8);ax[2].set_xticks(range(3),F.LOSSES,rotation=25)
ax[2].set_title("Two fixed temporary Adam steps\nBN-matched eval loss change");ax[2].legend(fontsize=7)
fig.tight_layout();fig.savefig(out/"diagnosis.png",dpi=170);plt.close(fig)
result=dict(status="PASS_FULL_T_ACCOUNTING_FIXED800_CPU64_AND_THREE_PARAMETER_GRADIENTS",
    T_queries=13230,train_batches=83,full_T_global_mapper_cosines=cos.tolist(),
    batch_mapper_cosine_median=np.median(batchcos,axis=0).tolist(),
    batch_negative_count=(batchcos<0).sum(0).tolist(),
    derivative_mass_fractions=means.tolist(),coordinate_order=["F","P"],region_order=["all","same_concept","different_concept"],
    control_order=["correct","same_concept_cyclic_target"],category_order=["both_positive","both_negative","opposite_sign","zero_boundary"],
    CPU64_max_logit_error=logerr,CPU64_max_prediction_error=prederr,
    CPU64_gradient_sample_max_errors=gradient_errors,CPU64_gradient_squared_norm_relative_errors=norm_errors,
    all_T_forward_backward_NOT_independently_replayed=True,temporary_steps_NOT_CPU_replayed=True,
    evidence_scope="Seen T, fixed strong NCE, native stochastic parameter gradients and frozen eval repeat readouts; no heldout effectiveness claim",
    limits=["Logit derivative mass is not parameter-gradient norm or physiological noise fraction",
            "Repeated R1 margins averaged; not f(physical R2)",
            "Disagreement need not be harmful; hard negatives naturally near boundary",
            "P-InfoNCE and cross-repeat diagnostics alone do not establish novelty",
            "Six temporary steps fully restored; no new fitted checkpoint or images"],
    finite_steps=finite,statistics=done["statistics"],fixture=done["fixture"],figure=K.ref(out/"diagnosis.png"),
    seconds=time.monotonic()-started)
K.write(out/"analysis.json",result);C.resources(c)
