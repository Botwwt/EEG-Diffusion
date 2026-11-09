"""Execute one source-pinned, finite research round without manual stage gaps.

This runner adds no candidates, retries, tuning, D resume, or new GPU wrapper.
Production GPU stages use the unchanged run_gpu_job_v2 lock/receipt/ledger.
Cleanup signals only a receipt-bound direct child after /proc identity checks;
the wrapper stays alive to record its exit and ledger. No killpg is used here.
Private operations injection supports CPU-only fixtures, never production CLI.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

import brainhive_paired_supervision_20261026_v3 as S

ROOT=Path(__file__).resolve().parent
SCHEMA='BrainHIVE_rapid_fixed_round_20261027_v1'
MAX_SECONDS=900
MIN_CLEANUP_RESERVE=40
DEAD=frozenset(('Z','X','x'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    first=Path(path).read_bytes();second=Path(path).read_bytes()
    assert first==second,'JSON changed during read: '+str(path)
    result=json.loads(second);assert isinstance(result,dict)
    return result


def atomic(path,value):
    path=Path(path);pending=path.with_name(path.name+'.pending')
    with pending.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def field(document,key):
    value=document
    for part in key.split('.'):value=value[part]
    return value


def finite(value,positive=False):
    assert type(value) in (float,int) and math.isfinite(value)
    assert value>0 if positive else value>=0
    return float(value)


def is_live(identity):return identity is not None and identity['state'] not in DEAD


def same_process(bound,current):
    if current is None:return False
    for key in ('pid','start_ticks','pgid','sid'):
        assert current[key]==bound[key],'Owned PID identity changed: '+key
    if is_live(current):assert current['argv']==bound['argv'],'Owned command changed'
    return is_live(current)


class LinuxOps:
    def __init__(self):assert sys.platform.startswith('linux'),'Live runner requires Linux /proc; use CPU fixtures elsewhere'
    monotonic=staticmethod(time.monotonic)
    sleep=staticmethod(time.sleep)
    inspect=staticmethod(S.inspect_linux_process)
    signal=staticmethod(os.kill)
    def spawn(self,command,root,log,env):
        return subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                env=env,start_new_session=True)


def validate_plan(plan_path,root=ROOT):
    root=Path(root).resolve();plan=read(plan_path)
    assert plan['schema']==SCHEMA
    assert isinstance(plan['fixed_candidate'],str) and plan['fixed_candidate']
    assert plan['automatic_hyperparameter_search'] is False and plan['D_resume'] is False
    maximum=finite(plan['maximum_seconds'],True);assert maximum<=MAX_SECONDS
    reserve=finite(plan.get('cleanup_reserve_seconds',60),True);assert reserve>=MIN_CLEANUP_RESERVE
    output=Path(plan['output_dir']).resolve()
    assert output.is_relative_to(root) and output!=root and not output.exists(),'Use a fresh output directory inside this project'
    assert plan['sources'] and isinstance(plan['sources'],list)
    sources={}
    for item in plan['sources']:
        path=Path(item['path']).resolve();assert path.is_file() and sha(path)==item['sha256']
        assert str(path) not in sources;sources[str(path)]=item['sha256']
    assert sources.get(str(Path(__file__).resolve()))==sha(__file__),'Pin this exact runner source'
    assert sources.get(str((ROOT/'brainhive_paired_supervision_20261026_v3.py').resolve()))==sha(ROOT/'brainhive_paired_supervision_20261026_v3.py')
    stages=plan['stages'];assert isinstance(stages,list) and 1<=len(stages)<=8
    names=set();jobs=set();guard_paths=set();caps=0.
    for stage in stages:
        name=stage['name'];assert isinstance(name,str) and re.fullmatch('[A-Za-z0-9_-]+',name) and name not in names;names.add(name)
        assert stage['kind'] in ('gpu','cpu')
        cap=finite(stage['max_seconds'],True);caps+=cap
        command=stage['command'];assert isinstance(command,list) and len(command)>=2 and all(type(v) is str and v and '\0' not in v for v in command)
        assert re.fullmatch(r'python(?:\d+(?:\.\d+)*)?(?:\.exe)?',Path(command[0]).name), 'Use an explicit Python executable'
        index=2 if command[1]=='-B' else 1
        assert len(command)>index and command[index] not in ('-c','-m')
        source=Path(command[index]).resolve();assert str(source) in sources,'Stage entry source must be pinned'
        assert source!=Path(__file__).resolve(),'No recursive runner stages'
        guard=stage['completion'];assert isinstance(guard,dict)
        path=Path(guard['path']).resolve();assert path.is_relative_to(root) and str(path) not in guard_paths and not path.exists()
        guard_paths.add(str(path))
        accepted=guard['accepted_status'];assert isinstance(accepted,list) and accepted and all(type(v) is str and v for v in accepted)
        early=guard.get('early_stop_statuses',[]);assert isinstance(early,list) and all(type(v) is str and v for v in early)
        assert not set(early)&set(accepted)
        for name in ('boolean_equals','early_stop_boolean_equals'):
            value=guard.get(name,{})
            assert isinstance(value,dict) and all(type(k) is str and k and type(v) is bool for k,v in value.items())
        if early:assert guard.get('early_stop_boolean_equals'),'A scientific stop requires explicit boolean evidence'
        for name in ('accepted_child_exit_codes','early_stop_child_exit_codes'):
            values=guard.get(name,[0]);assert isinstance(values,list) and values and all(type(v) is int for v in values)
        for path in stage.get('new_paths',[]):
            path=Path(path).resolve();assert path.is_relative_to(root) and path!=root and not path.exists()
        if stage['kind']=='gpu':
            name=stage['job_name'];assert re.fullmatch('[A-Za-z0-9_.-]+',name) and name not in jobs;jobs.add(name)
            assert not (root/'audit'/(name+'_job.json')).exists(),'Preserve all GPU attempts: use a new job name'
            wrapper=(ROOT/'run_gpu_job_v2.py').resolve()
            assert sources.get(str(wrapper))==sha(wrapper),'Pin the unchanged GPU wrapper'
    assert caps+reserve<=maximum,'Stage caps must leave the global cleanup reserve'
    return plan,sources


def check_pins(plan_path,plan_hash,sources):
    assert sha(plan_path)==plan_hash,'Plan changed after launch'
    for path,expected in sources.items():assert sha(path)==expected,'Pinned source changed: '+path


def check_completion(guard,child_exit):
    document=read(guard['path']);status=document.get('status')
    if status in guard['accepted_status']:
        assert child_exit in guard.get('accepted_child_exit_codes',[0]),'Successful output conflicts with child exit'
        booleans=guard.get('boolean_equals',{});result='ACCEPTED'
    elif status in guard.get('early_stop_statuses',[]):
        assert child_exit in guard.get('early_stop_child_exit_codes',[0]),'Scientific stop conflicts with child exit'
        booleans=guard['early_stop_boolean_equals'];result='SCIENTIFIC_EARLY_STOP'
    else:raise AssertionError('Unaccepted completion status: '+repr(status))
    for key,expected in booleans.items():assert field(document,key) is expected,'Boolean output guard failed: '+key
    return dict(result=result,status=status,booleans_verified=booleans,completion=ref(guard['path']))


class OwnedStage:
    def __init__(self,stage,process,argv,root,ops):
        self.stage,self.process,self.argv,self.root,self.ops=stage,process,argv,Path(root),ops
        self.wrapper_identity=None;self.child_identity=None;self.initial_receipt=None;self.closure=None
        identity=ops.inspect(process.pid)
        if identity is not None:
            if is_live(identity):assert identity['argv']==argv
            self.wrapper_identity=identity

    def receipt_path(self):return self.root/'audit'/(self.stage['job_name']+'_job.json')

    def inspect_receipt(self):
        if self.stage['kind']!='gpu':return
        path=self.receipt_path()
        if not path.exists():return
        try:record=read(path)
        except (json.JSONDecodeError,AssertionError):return  # non-atomic wrapper write; retry, never use a partial record
        assert record['name']==self.stage['job_name'] and record['wrapper_pid']==self.process.pid
        assert record['command']==self.stage['command'] and float(record['max_seconds'])==float(self.stage['max_seconds'])
        assert type(record['pid']) is int and record['pid']>1 and record['pid']!=self.process.pid
        if self.initial_receipt is None:
            self.initial_receipt={key:value for key,value in record.items() if key not in ('status','exit_code','charged_seconds','end_utc','termination_reason')}
        else:
            for key,value in self.initial_receipt.items():assert record.get(key)==value,'Receipt ownership/provenance changed'
        wrapper=self.ops.inspect(self.process.pid)
        if self.wrapper_identity is not None:same_process(self.wrapper_identity,wrapper)
        elif is_live(wrapper):
            assert wrapper['argv']==self.argv;self.wrapper_identity=wrapper
        child=self.ops.inspect(record['pid'])
        if self.child_identity is not None:same_process(self.child_identity,child)
        elif is_live(child):
            assert child['argv']==self.stage['command'] and child['ppid']==self.process.pid
            assert child['pgid']==child['sid']==child['pid']
            assert self.wrapper_identity is not None and child['start_ticks']>=self.wrapper_identity['start_ticks']
            self.child_identity=child
        return record

    def signal_child(self,number):
        if self.stage['kind']=='gpu':
            record=self.inspect_receipt();assert record is not None,'No complete wrapper receipt: refusing an unbound child signal'
            bound=self.child_identity
            current=self.ops.inspect(record['pid'])
            if not is_live(current):return dict(signal_sent=False,reason='bound child already dead')
            assert bound is not None,'No initial live /proc child identity: refusing signal'
            assert same_process(bound,current)
            wrapper=self.ops.inspect(self.process.pid)
            assert self.wrapper_identity is not None and same_process(self.wrapper_identity,wrapper),'Wrapper must remain live for ledger closure'
            assert current['ppid']==self.process.pid and record['wrapper_pid']==self.process.pid
            self.ops.signal(bound['pid'],number)
            return dict(signal_sent=True,pid=bound['pid'],start_ticks=bound['start_ticks'],signal=int(number),receipt=ref(self.receipt_path()),identity_verified=True)
        bound=self.wrapper_identity;current=self.ops.inspect(self.process.pid)
        if not is_live(current):return dict(signal_sent=False,reason='owned CPU process already dead')
        assert bound is not None and same_process(bound,current)
        self.ops.signal(bound['pid'],number)
        return dict(signal_sent=True,pid=bound['pid'],start_ticks=bound['start_ticks'],signal=int(number),identity_verified=True)

    def cleanup(self):
        evidence=[]
        if self.process.poll() is None:
            evidence.append(self.signal_child(signal.SIGTERM))
            until=self.ops.monotonic()+8
            while self.process.poll() is None and self.ops.monotonic()<until:
                if self.stage['kind']=='gpu':self.inspect_receipt()
                self.ops.sleep(.1)
            if self.process.poll() is None:
                evidence.append(self.signal_child(signal.SIGKILL))
                until=self.ops.monotonic()+25
                while self.process.poll() is None and self.ops.monotonic()<until:
                    if self.stage['kind']=='gpu':self.inspect_receipt()
                    self.ops.sleep(.1)
        assert self.process.poll() is not None,'Owned wrapper did not close; retained for attention, never killed before ledger closure'
        return evidence

    def closed(self):
        code=self.process.poll();assert type(code) is int
        if self.stage['kind']=='cpu':return dict(child_exit_code=code,CPU_direct_child=True)
        record=self.inspect_receipt();assert record is not None,'GPU wrapper exited without its own complete receipt'
        child_exit=record.get('exit_code');S.validate_wrapper_exit(child_exit,code)
        expected='COMPLETED' if child_exit==0 else 'FAILED_OR_JOB_TIMEOUT_RESUMABLE'
        assert record.get('status')==expected and record.get('end_utc')
        finite(record['charged_seconds'])
        current=self.ops.inspect(record['pid'])
        if self.child_identity is not None:assert not same_process(self.child_identity,current),'Bound GPU child still live after wrapper exit'
        else:assert not is_live(current),'Unbound child remains live; refusing successful closure'
        ledger=self.root/'audit/compute_ledger.jsonl'
        found=[json.loads(line) for line in ledger.read_text().splitlines() if line.strip() and json.loads(line).get('name')==self.stage['job_name']]
        assert len(found)==1 and found[0]==record,'Receipt must match exactly one closed ledger record'
        self.closure=dict(child_exit_code=child_exit,wrapper_exit_code=code,receipt=ref(self.receipt_path()),
            charged_seconds=record['charged_seconds'],ledger_path=str(ledger),ledger_record_verified=True,
            wrapper_PID=self.process.pid,child_PID=record['pid'])
        return self.closure


def run_plan(plan_path,*,_ops=None,_root=ROOT):
    ops=LinuxOps() if _ops is None else _ops
    started=ops.monotonic()
    root=Path(_root).resolve();plan,sources=validate_plan(plan_path,root)
    plan_hash=sha(plan_path);output=Path(plan['output_dir']).resolve();output.mkdir(parents=True)
    deadline=started+float(plan['maximum_seconds'])
    stopped=[False]
    if _ops is None:
        signal.signal(signal.SIGTERM,lambda number,frame:stopped.__setitem__(0,True))
        signal.signal(signal.SIGINT,lambda number,frame:stopped.__setitem__(0,True))
    events=[];results=[];active=None;cleanup=[]
    def event(event_name,**details):
        item=dict(event=event_name,elapsed_seconds=ops.monotonic()-started,**details);events.append(item)
        with (output/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    def state(status,**details):
        atomic(output/'progress.json',dict(schema=SCHEMA,status=status,fixed_candidate=plan['fixed_candidate'],
            elapsed_seconds=ops.monotonic()-started,maximum_seconds=plan['maximum_seconds'],completed_stages=len(results),
            expected_stages=len(plan['stages']),stages=results,**details))
    try:
        atomic(output/'provenance.json',dict(schema=SCHEMA,plan=ref(plan_path),driver=ref(__file__),sources=sources,
            D_resume=False,automatic_hyperparameter_search=False,fixed_candidate=plan['fixed_candidate'],
            timing_scope='initial plan/source preflight, all stage execution, interstage validation and cleanup; excludes coding/review/deployment before this driver'))
        event('ROUND_STARTED');state('RUNNING')
        for index,stage in enumerate(plan['stages']):
            check_pins(plan_path,plan_hash,sources)
            assert not stopped[0] and ops.monotonic()<deadline-float(stage['max_seconds'])-MIN_CLEANUP_RESERVE,'Insufficient finite round budget for the next whole stage'
            assert not Path(stage['completion']['path']).exists()
            for path in stage.get('new_paths',[]):assert not Path(path).exists(),'Stage output appeared before launch'
            command=stage['command'];argv=command
            if stage['kind']=='gpu':
                assert not (root/'audit'/(stage['job_name']+'_job.json')).exists()
                argv=[command[0],'-B',str(ROOT/'run_gpu_job_v2.py'),'--name',stage['job_name'],'--max-seconds',str(stage['max_seconds']),'--',*command]
            env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
            if stage['kind']=='cpu':env['CUDA_VISIBLE_DEVICES']=''
            launch=ops.monotonic();logpath=output/(f'{index:02d}_{stage["name"]}.log')
            with logpath.open('xb') as log:
                process=ops.spawn(argv,root,log,env);active=OwnedStage(stage,process,argv,root,ops)
                event('STAGE_STARTED',name=stage['name'],kind=stage['kind'],PID=process.pid,command=argv,max_seconds=stage['max_seconds'])
                timed_out=False;last_progress=-1.
                while process.poll() is None:
                    if stage['kind']=='gpu':active.inspect_receipt()
                    now=ops.monotonic()
                    if stopped[0] or now-launch>=float(stage['max_seconds']) or now>=deadline-MIN_CLEANUP_RESERVE:
                        timed_out=True;event('STAGE_STOP_REQUESTED',name=stage['name'],signal_requested=bool(stopped[0]))
                        cleanup=active.cleanup();break
                    if now-last_progress>=1:
                        state('RUNNING',active_stage=stage['name'],active_PID=process.pid,stage_elapsed_seconds=now-launch)
                        last_progress=now
                    ops.sleep(.1)
                closure=active.closed();elapsed=ops.monotonic()-launch
                stage_result=dict(name=stage['name'],kind=stage['kind'],seconds=elapsed,process_closure=closure,
                    log=ref(logpath),cleanup=cleanup)
                event('STAGE_PROCESS_CLOSED',**stage_result)
                active=None;cleanup=[]
                assert not timed_out,'Finite stage/round deadline or user signal; no later stage is allowed'
            check_pins(plan_path,plan_hash,sources)
            guard=check_completion(stage['completion'],closure['child_exit_code']);stage_result.update(guard=guard)
            results.append(stage_result);event('STAGE_GUARD_CLOSED',name=stage['name'],guard=guard)
            if guard['result']=='SCIENTIFIC_EARLY_STOP':
                state('SCIENTIFIC_EARLY_STOP',remaining_stages_not_started=True)
                final=dict(status='SCIENTIFIC_EARLY_STOP',schema=SCHEMA,stages=results,
                    fixed_candidate=plan['fixed_candidate'],seconds=ops.monotonic()-started,remaining_stages_not_started=True,
                    full_round=False,no_automatic_retry=True,D_resume=False)
                atomic(output/'completion.json',final);return final
        check_pins(plan_path,plan_hash,sources)
        assert ops.monotonic()<=deadline
        final=dict(status='COMPLETE_RAPID_FIXED_ROUND',schema=SCHEMA,stages=results,fixed_candidate=plan['fixed_candidate'],
            seconds=ops.monotonic()-started,all_registered_stages_closed=True,maximum_seconds=plan['maximum_seconds'],
            interstage_idle_is_included=True,D_resume=False,automatic_hyperparameter_search=False,no_automatic_retry=True)
        state(final['status']);atomic(output/'completion.json',final);event('ROUND_CLOSED',seconds=final['seconds'])
        return final
    except BaseException as error:
        cleanup_error=None
        if active is not None:
            try:
                cleanup=active.cleanup();active.closed()
            except BaseException as failure:cleanup_error=repr(failure)
        failure=dict(status='FAILED_RAPID_ROUND_PRESERVED_NO_AUTOMATIC_RETRY',schema=SCHEMA,error=repr(error),
            traceback=traceback.format_exc(),seconds=ops.monotonic()-started,completed_stages=results,
            cleanup=cleanup,cleanup_attention_required=cleanup_error is not None,cleanup_error=cleanup_error,
            D_resume=False,no_following_stage_started=True)
        atomic(output/'failure.json',failure);state(failure['status'],cleanup_attention_required=cleanup_error is not None)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args(argv)
    if args.preflight_only:
        plan,sources=validate_plan(args.plan)
        print(json.dumps(dict(status='RAPID_FINITE_PLAN_CPU_VERIFIED',stages=len(plan['stages']),
            maximum_seconds=plan['maximum_seconds'],source_count=len(sources),GPU_started=False)),flush=True);return 0
    result=run_plan(args.plan);print(json.dumps(result,allow_nan=False),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
