import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import importlib.util,json,subprocess,sys,time,traceback
from pathlib import Path
plan=json.loads(Path(sys.argv[1]).read_bytes());root=Path(plan["parent_project"])
sys.path.insert(0,str(root/"code"))
import atm_deployment_training_contract_20261105_v179 as C
c=C.setup(root/"code/policy.json");q,K,pq=c["q"],c["K"],c["physical"];out=Path(plan["output_root"])
assert K.ref(__file__)==plan["sources"]["controller"]
sp=importlib.util.spec_from_file_location("owned",K.checked(pq["owned_process"]));P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
started=time.monotonic();owner=child=None;events=[]
while not (out/"actual_launch.json").exists():
    assert time.monotonic()-started<15
    time.sleep(.05)
launch=K.read(out/"actual_launch.json");assert launch["PID"]==os.getpid()
repo=Path(pq["GPU_wrapper"]["path"]).parent
try:
    for key in ("GPU","CPU"):
        C.resources(c,True);assert time.time()+935<launch["absolute_deadline_epoch"]
        cmd=[sys.executable,"-B",K.checked(plan["sources"][key]),sys.argv[1]]
        receipt=repo/"audit"/(plan["job_name"]+"_job.json")
        if key=="GPU":
            assert not receipt.exists()
            cmd=[sys.executable,"-B",K.checked(pq["GPU_wrapper"]),"--name",plan["job_name"],"--max-seconds","900","--",*cmd]
        env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]="0" if key=="GPU" else ""
        with (out/(key+".log")).open("xb") as log:
            owner=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=K.process(owner.pid);child=P.capture_wrapper_child(owner,receipt) if key=="GPU" else None
            K.write(out/"stage_owner.json",dict(stage=key,owner=identity,receipt=str(receipt) if key=="GPU" else None))
            rc=owner.wait(timeout=935)
        assert rc==0,(key,rc)
        assert not K.process(identity["PID"],identity["start_ticks"])["live"]
        if key=="GPU":
            j=K.read(receipt);assert j["status"]=="COMPLETED" and j["exit_code"]==0
            assert not K.process(j["pid"])["live"] and not K.process(j["wrapper_pid"])["live"]
        events.append(dict(stage=key,receipt=K.ref(receipt) if key=="GPU" else None));owner=child=None
    a=K.read(out/"analysis.json");assert a["status"].startswith("PASS_FULL_BLOCK_FACTORIAL")
    K.write(out/"completion.json",dict(status="COMPLETE_FROZEN_EEG_BLOCK_FACTORIAL",analysis=K.ref(out/"analysis.json"),
        events=events,retained_updates=0,new_images=0,seconds=time.monotonic()-started))
except BaseException as e:
    cleanup=P.stop_owned(owner,child)
    K.write(out/"failure.json",dict(error=repr(e),traceback=traceback.format_exc(),events=events,cleanup=cleanup))
    raise
