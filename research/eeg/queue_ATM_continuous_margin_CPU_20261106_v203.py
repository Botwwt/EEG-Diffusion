"""Fixed 830 attempts per arm, complete two-domain images, no effect-based stop."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,subprocess,sys,time,traceback
from pathlib import Path
import atm_continuous_margin_contract_20261106_v203 as C
c=C.setup(sys.argv[1]);q,K,out=c['q'],c['K'],c['out'];assert K.ref(__file__)==q['sources']['controller'];started=time.monotonic();owner=child=None;events=[]
from atm_repeat_image_clean_cache_20261105_v2 import release
spec=importlib.util.spec_from_file_location('owned',K.checked(c['physical']['owned_process']));P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
lp=Path(q['code_root'])/'actual_launch.json'
while not lp.exists():
    assert time.monotonic()-started<15;time.sleep(.05)
launch=K.read(lp);assert launch['PID']==os.getpid() and K.process(os.getpid())['start_ticks']==launch['start_ticks']
def stage(key,mode,args=(),gpu=False,suffix=None):
    global owner,child
    assert time.time()+935<launch['absolute_deadline_epoch']
    K.write(out/('cache_advice_controller_'+str(len(events))+'.json'),release(c['physical']['physical_source']))
    C.resources(c,starting=True);name=q['job_prefix']+'_'+(suffix or mode)
    cmd=[sys.executable,'-B',q['sources'][key]['path'],'--policy',sys.argv[1],'--mode',mode,*args]
    receipt=c['repo']/'audit'/(name+'_job.json')
    if gpu:
        assert not receipt.exists();cmd=[sys.executable,'-B',c['physical']['GPU_wrapper']['path'],'--name',name,'--max-seconds','900','--',*cmd]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
    K.write(out/'progress.json',dict(phase=name,completed_stages=len(events),expected_stages=q['expected_stage_count'],image_effects_read=False))
    with (out/(name+'.log')).open('xb') as f:
        owner=subprocess.Popen(cmd,cwd=c['repo'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        child=P.capture_wrapper_child(owner,receipt) if gpu else None;identity=K.process(owner.pid);rc=owner.wait(timeout=935)
    assert rc==0,(name,rc);assert not K.process(identity['PID'],identity['start_ticks'])['live']
    if gpu:
        j=K.read(receipt);assert j['status']=='COMPLETED' and j['exit_code']==0
        assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
    events.append(dict(stage=name,GPU=gpu,receipt=K.ref(receipt) if gpu else None));owner=child=None;K.write(out/'events.json',events)
try:
    for arm in C.ARMS[2:]:
        block=q['constrained_attempts_per_job'] if arm=='margin_constrained' else 83
        for lo in range(0,830,block):
            hi=min(lo+block,830)
            stage('fit','fit',['--arm',arm,'--start',str(lo),'--end',str(hi)],True,'fit_'+arm+'_'+str(lo)+'_'+str(hi))
    stage('fit','conditions',gpu=True)
    stage('fit_CPU','fit_audit')
    stage('GPU','profile',gpu=True)
    for domain,stops in [('R4',(0,80,160,240,320)),('R80',(0,50,100,150))]:
        for lo in stops:stage('GPU','generate',['--domain',domain,'--start',str(lo)],True,'generate_'+domain+'_'+str(lo))
    stage('CPU','audit')
    for domain in ['R4','R80']:stage('GPU','encode',['--domain',domain],True,'encode_'+domain)
    for arm in C.ARMS[2:]:stage('GPU','paper',['--arm',arm],True,'paper_'+arm)
    stage('CPU','score');assert len(events)==q['expected_stage_count']
    fits={arm:K.read(c['fit']/arm/'fit_10_completion.json') for arm in C.ARMS[2:]}
    assert all(v['attempts']==830 and v['exposures']==132300 for v in fits.values())
    K.write(out/'completion.json',dict(status='COMPLETE_CONTINUOUS_MARGIN_TRAINING_AND_IMAGES',policy=K.ref(sys.argv[1]),events=events,expected_stage_count=q['expected_stage_count'],
        fit_audit=K.ref(c['fit']/'fit_audit_completion.json'),readout=K.ref(out/'readout.json'),image_audit=K.ref(out/'image_audit.json'),
        new_images=1200,reused_images=1200,engineering_images=1,new_training_attempts=1660,accepted_updates={a:v['accepted'] for a,v in fits.items()},new_models=2,
        seconds=time.monotonic()-started,all_registered_metrics_and_contrasts_reported=True,NEW_contacted=False,original_ordinary_unchanged=True,no_automatic_more_fits=True))
    assert time.time()+935<launch['absolute_deadline_epoch'];C.resources(c,True)
    review=Path(q['project_root'])/'terminal_review';review.mkdir()
    cmd=['timeout','--signal=TERM','--kill-after=10s','900',sys.executable,'-B',q['sources']['review']['path'],'--policy',sys.argv[1],'--plan',q['review_plan']['path']]
    with (review/'worker.log').open('xb') as f:rp=subprocess.Popen(cmd,cwd=c['repo'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    ri=K.process(rp.pid);assert ri['live'];K.write(review/'actual_launch.json',dict(PID=rp.pid,start_ticks=ri['start_ticks'],maximum_seconds=900,after_full_images=True,waits_for_original_controller_exit=True,command=cmd))
    K.write(out/'progress.json',dict(phase='COMPLETE_IMAGES_TERMINAL_REVIEW_QUEUED',review_owner=ri))
except BaseException as exc:
    cleanup=P.stop_owned(owner,child);K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),events=events,cleanup=cleanup,seconds=time.monotonic()-started));raise
