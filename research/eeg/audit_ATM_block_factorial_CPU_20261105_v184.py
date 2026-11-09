"""CPU64 composed donor forwards, complete stored-query metric/CI recomputation."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]=os.environ["MKL_NUM_THREADS"]=os.environ["OPENBLAS_NUM_THREADS"]="2"
import json,resource,sys,time
from pathlib import Path
import numpy as np
plan=json.loads(Path(sys.argv[1]).read_bytes());root=Path(plan["parent_project"]);out=Path(plan["output_root"])
sys.path.insert(0,str(root/"image_code"))
import atm_deployment_image_contract_20261105_v179 as C
c=C.setup(root/"image_code/policy.json");K=c["K"];q=c["train"]["q"];pq=c["physical"]
assert K.ref(__file__)==plan["sources"]["CPU"];C.resources(c,True)
sys.path[:0]=pq["existing_runtime_paths"]+["/root/atm_official_20261016"]
import torch
from models.atms import ATMS
import inspect
assert Path(inspect.getfile(ATMS)).resolve()==Path(K.checked(plan["actual_ATMS_source"])).resolve()
torch.set_num_threads(2);started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
gpu=K.read(out/"GPU_completion.json");assert gpu["status"]=="FULL_T_AND_600_QUERY_FROZEN_BLOCK_FACTORIAL_GPU_COMPLETE"
def guard():
    assert time.monotonic()-started<850
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<pq["maximum_extra_CPU_bytes"]
    assert K.allocated(out)<plan["maximum_output_bytes"]
    C.resources(c)
nets=[]
for pin in plan["parents"]:
    st=torch.load(K.checked(pin),map_location="cpu",weights_only=True,mmap=True)["model"]
    net=ATMS();net.load_state_dict(st,strict=True);net=net.double().eval();nets.append(net)
raw=torch.load(K.checked(pq["EEG"]),map_location="cpu",weights_only=True,mmap=True)
test=torch.load(K.checked(c["train"]["old"]["train"]["R80"]["EEG_R80"]),map_location="cpu",weights_only=True,mmap=True)
values=np.load(out/"conditions.npy");assert values.shape==(600,4,1024)
fix=np.load(out/"T_fixture.npz");met=np.load(out/"per_query.npz");rows=K.read(out/"queries.json")
assert rows==c["queries"];ids=[r["array_row"] for r in rows["R4"]]
errors={};count=0
with torch.inference_mode():
    for domain,x,stored in [("T32",raw[fix["T_ids"]],fix["T_conditions"]),("R4",raw[ids],values[:400]),("R80",test,values[400:])]:
        for ai,(ti,hi) in enumerate(((0,0),(0,1),(1,0),(1,1))):
            err=0.
            for lo in range(0,len(x),32):
                guard();xx=x[lo:lo+32].double();subject=torch.ones(len(xx),dtype=torch.long)
                # Compose independent original donors, rather than loading a hybrid state.
                features=nets[ti].enc_eeg(nets[ti].encoder(xx,None,subject))
                pred=nets[hi].proj_eeg(features).numpy()
                err=max(err,float(abs(pred-stored[lo:lo+len(xx),ai]).max()));count+=len(xx)
            assert err<5e-5,(domain,ai,err);errors[domain+"__"+gpu["arms"][ai]]=err
wp=np.load(K.checked(q["projector"]));W=torch.from_numpy(wp["image_embeds.weight"].astype(np.float64))
B=torch.from_numpy(wp["image_embeds.bias"].astype(np.float64));G=torch.from_numpy(wp["norm.weight"].astype(np.float64));BE=torch.from_numpy(wp["norm.bias"].astype(np.float64))
def p(x):
    z=torch.nn.functional.linear(torch.from_numpy(x.astype(np.float64)),W,B).reshape(-1,4,2048)
    return torch.nn.functional.layer_norm(z,(2048,),G,BE,1e-5).numpy()
def norm(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
metric_error=0.
for domain,y,offset in [("R4",np.load(K.checked(plan["R4_teacher"])),0),("R80",torch.load(K.checked(plan["R80_teacher"]),map_location="cpu",weights_only=True,mmap=True).numpy(),400)]:
    y=y.astype(np.float64);py=p(y);groups={}
    for i,row in enumerate(rows[domain]):groups.setdefault(row["concept_id"],[]).append(i)
    for ai,arm in enumerate(gpu["arms"]):
        guard();x=values[offset:offset+len(y),ai].astype(np.float64);px=p(x)
        expected={"F_cosine":np.sum(norm(x)*norm(y),axis=1),"F_MSE":np.mean((x-y)**2,axis=1),
          "condition_norm":np.linalg.norm(x,axis=1),"P_cosine":np.mean(np.sum(norm(px)*norm(py),axis=-1),axis=1),
          "P_MSE":np.mean((px-py)**2,axis=(1,2))}
        for label,xx,yy,div in [("F",norm(x),norm(y),1),("P",norm(px).reshape(len(x),-1),norm(py).reshape(len(y),-1),4)]:
            s=np.einsum("ik,jk->ij",xx,yy,optimize=True)/div
            top=[];top5=[];afc=[];wt=[];wa=[]
            for i in range(len(y)):
                neg=np.delete(s[i],i);rank=np.lexsort((np.arange(len(y)),-s[i]))
                top.append(float(rank[0]==i));top5.append(float(i in rank[:5]))
                afc.append(float(((neg<s[i,i]).sum()+.5*(neg==s[i,i]).sum())/len(neg)))
                if domain=="R4":
                    g=groups[rows[domain][i]["concept_id"]];nn=[j for j in g if j!=i]
                    wt.append(float(g[np.argmax(s[i,g])]==i))
                    wa.append(float(((s[i,nn]<s[i,i]).sum()+.5*(s[i,nn]==s[i,i]).sum())/9))
            expected.update({label+"_gallery_top1":top,label+"_gallery_top5":top5,label+"_gallery_AFC":afc})
            if domain=="R4":expected.update({label+"_within10_top1":wt,label+"_within10_AFC":wa})
        for key,a in expected.items():
            er=float(np.max(abs(np.asarray(a)-met[domain+"__"+arm+"__"+key])))
            assert er<1e-10,(domain,arm,key,er);metric_error=max(metric_error,er)
ci_error=0.;ci_count=0
for domain in ("T","R4","R80"):
    n=13230 if domain=="T" else 400 if domain=="R4" else 200
    if domain=="T":groups=[list(range(i,i+10)) for i in range(0,n,10)]
    else:
        gg={}
        for i,row in enumerate(rows[domain]):gg.setdefault(row["concept_id"],[]).append(i)
        groups=[gg[k] for k in sorted(gg)]
    ds=gpu["domains"][domain];arms=gpu["arms"];draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)))
    for name,stats in ds["contrasts"].items():
        if name=="interaction_PP-PN-NP+NN":weights=[1,-1,-1,1]
        else:
            aa,bb=name.split("-");weights=[int(a==aa)-int(a==bb) for a in arms]
        for key,stat in stats.items():
            delta=sum(w*met[domain+"__"+a+"__"+key] for a,w in zip(arms,weights))
            assert len(delta)==n and np.isfinite(delta).all()
            groupmeans=np.asarray([np.mean(delta[ix]) for ix in groups])
            ci=np.percentile(np.mean(groupmeans[draw],axis=1),[2.5,97.5])
            er=max(abs(float(delta.mean())-stat["difference"]),float(abs(ci-stat["descriptive_concept95"]).max()))
            assert er<1e-12;ci_error=max(ci_error,er);ci_count+=1
assert not torch.cuda.is_initialized()
for pin in plan["parents"]:K.checked(pin)
guard()
K.write(out/"analysis.json",dict(status="PASS_FULL_BLOCK_FACTORIAL_2528_CPU64_FORWARDS_AND_ALL_STORED_METRICS_CIS",
 domains=gpu["domains"],CPU_forward_queries=count,CPU_condition_max_errors=errors,
 all600_F_P_metrics_max_error=metric_error,all_T_R4_R80_CI_count=ci_count,CI_max_error=ci_error,
 CPU_replay_scope="all600x4 + first16/last16Tx4; no fullT neural replay; allT saved metric accounting and CIs",
 original_models_unchanged=True,new_models=0,new_updates=0,new_images=0,NEW_contacted=False,
 seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0))
