"""OLD-only frozen native-state interventions; never save modified weights."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["OMP_NUM_THREADS"]=os.environ["MKL_NUM_THREADS"]=os.environ["OPENBLAS_NUM_THREADS"]="2"
import fcntl,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_block_factorial_core_20261105_v184 as F
plan=json.loads(Path(sys.argv[1]).read_bytes());root=Path(plan["parent_project"]);out=Path(plan["output_root"])
sys.path.insert(0,str(root/"image_code"))
import atm_deployment_image_contract_20261105_v179 as C
c=C.setup(root/"image_code/policy.json");K=c["K"];q=c["train"]["q"];pq=c["physical"]
for pin in plan["sources"].values():K.checked(pin)
assert K.ref(__file__)==plan["sources"]["GPU"]
before=C.resources(c,True);assert before["own_remaining_bytes"]>plan["maximum_output_bytes"]+(4<<20)
repo=Path(pq["GPU_wrapper"]["path"]).parent
j=K.read(repo/"audit"/(plan["job_name"]+"_job.json"));assert j["status"]=="RUNNING" and j["pid"]==os.getpid()
lock=(repo/"audit/TRIALSET_GPU_INTERLEAVE.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()==pq["GPU_UUID"]
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
sys.path[:0]=pq["existing_runtime_paths"]+["/root/atm_official_20261016"]
import torch
from models.atms import ATMS
import inspect
assert Path(inspect.getfile(ATMS)).resolve()==Path(K.checked(plan["actual_ATMS_source"])).resolve()
torch.set_num_threads(2);torch.manual_seed(1729);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(pq["maximum_GPU_bytes"]/torch.cuda.get_device_properties(0).total_memory)
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
def guard():
    assert time.monotonic()-started<850
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<pq["maximum_extra_CPU_bytes"]
    assert torch.cuda.max_memory_allocated()<pq["maximum_GPU_bytes"]
    assert K.allocated(out)<plan["maximum_output_bytes"]
    C.resources(c)
try:
    parents=[torch.load(K.checked(pin),map_location="cpu",weights_only=True,mmap=True)["model"] for pin in plan["parents"]]
    assert parents[0].keys()==parents[1].keys()
    net=ATMS().cuda();names=list(net.state_dict());assert set(names)==set(parents[0])
    trunk=[k for k in names if k.startswith(("encoder.","enc_eeg."))]
    head=[k for k in names if k.startswith("proj_eeg.")]
    inactive=[k for k in names if k not in trunk+head]
    assert inactive and all(k=="logit_scale" or k.startswith("subject_wise_linear.") for k in inactive)
    raw=torch.load(K.checked(pq["EEG"]),map_location="cpu",weights_only=True,mmap=True)
    test=torch.load(K.checked(c["train"]["old"]["train"]["R80"]["EEG_R80"]),map_location="cpu",weights_only=True,mmap=True)
    visual=torch.load(K.checked(pq["visual_target"]),map_location="cpu",weights_only=True,mmap=True)
    y80=torch.load(K.checked(plan["R80_teacher"]),map_location="cpu",weights_only=True,mmap=True).numpy()
    teacher4=np.load(K.checked(plan["R4_teacher"]));T=np.asarray(q["T_array_rows"],np.int64)
    assert len(T)==13230 and hashlib.sha256(T.tobytes()).hexdigest()==q["T_rows_sha256"]
    assert np.all(T.reshape(1323,10)//10==T.reshape(1323,10)[:,:1]//10)
    ids4=np.asarray([v["array_row"] for v in c["queries"]["R4"]]);assert not set(ids4)&set(T)
    oldrows=K.read(K.checked(plan["R4_queries"]))
    identity=("query_id","subject_id","stimulus_id","concept_id","array_row","image_seed")
    assert len(oldrows)==400 and all(all(a[k]==b[k] for k in identity) for a,b in zip(c["queries"]["R4"],oldrows))
    weights={k:v.astype(np.float64) for k,v in np.load(K.checked(q["projector"])).items()}
    banks={};arrays={};analyses={};state_hashes={};baseline_errors={};fixtures={}
    for domain,xx,yy,ids in (("T",raw[T],visual[T].numpy(),T),("R4",raw[ids4],teacher4,ids4),("R80",test,y80,np.arange(200))):
        guard();n=len(xx)
        if domain=="T":groups=[list(range(i,i+10)) for i in range(0,n,10)]
        else:
            gg={}
            for i,row in enumerate(c["queries"][domain]):gg.setdefault(row["concept_id"],[]).append(i)
            groups=[gg[k] for k in sorted(gg)]
        perarm={};preds=[]
        for arm,(ti,hi) in zip(F.ARMS,F.FACTORS):
            guard()
            state={k:parents[ti if k in trunk else hi if k in head else 0][k] for k in names}
            net.load_state_dict(state,strict=True);net.eval()
            # BN is part of the trunk donor; no recalibration and no dropout.
            assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in net.state_dict().items())
            chunks=[]
            with torch.inference_mode():
                for lo in range(0,n,100):
                    guard();x=xx[lo:lo+100].cuda()
                    chunks.append(net(x,torch.ones(len(x),dtype=torch.long,device="cuda")).cpu().numpy())
            pred=np.concatenate(chunks);preds.append(pred)
            perarm[arm]=F.measure(pred,yy,groups,weights,domain!="T")
            arrays.update({domain+"__"+arm+"__"+k:v for k,v in perarm[arm].items()})
            state_hashes[domain+"__"+arm]=hashlib.sha256(pred.tobytes()).hexdigest()
            if domain!="T" and arm in (F.ARMS[0],F.ARMS[3]):
                idx=0 if arm==F.ARMS[0] else 2;lo=0 if domain=="R4" else 400
                er=float(abs(pred-c["values"][lo:lo+n,idx]).max());assert er<5e-5
                baseline_errors[domain+"__"+arm]=er
            K.write(out/"progress.json",dict(domain=domain,arm=arm,queries=n,new_models=0,new_images=0))
        pp=np.stack(preds,1)
        if domain=="T":
            fi=np.r_[np.arange(16),np.arange(n-16,n)]
            fixtures=dict(T_positions=fi,T_ids=T[fi],T_conditions=pp[fi],T_targets=yy[fi])
        else:banks[domain]=pp
        analyses[domain]=F.summarize(perarm,groups)
    np.save(out/"conditions.npy",np.concatenate([banks["R4"],banks["R80"]]))
    np.savez_compressed(out/"T_fixture.npz",**fixtures)
    np.savez_compressed(out/"per_query.npz",**arrays)
    K.write(out/"queries.json",c["queries"])
    result=dict(status="FULL_T_AND_600_QUERY_FROZEN_BLOCK_FACTORIAL_GPU_COMPLETE",domains=analyses,
      arms=F.ARMS,factors=F.FACTORS,trunk_keys=trunk,head_keys=head,inactive_keys=inactive,
      inactive_always_N=True,BN_donor="trunk",mode="eval; no BN refresh; all weights frozen",
      full_T_conditions=52920,heldout_but_previously_exposed_conditions=2400,
      prediction_SHA256=state_hashes,baseline_errors=baseline_errors,parents=plan["parents"],
      no_training_no_model_selection=True,new_models=0,new_updates=0,new_images=0,
      effect_interpretation="Posthoc block intervention; coadaptation/gauge mismatch can confound hybrids; not independent confirmation or images.",
      GPU_seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,
      GPU_peak_bytes=torch.cuda.max_memory_allocated(),resource_before=before,resource_after=C.resources(c))
    for pin in plan["parents"]:K.checked(pin)
    guard();K.write(out/"GPU_completion.json",result)
except BaseException as e:
    K.write(out/"GPU_failure.json",dict(error=repr(e),traceback=traceback.format_exc()))
    raise
