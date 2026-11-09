"""Two fixed ideas; isolated queue with honest complete-lifecycle admission."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from brainhive_two_ideas_storage_v1 import ROOT, WORK, DATA, sha, core_sha, read, forecast, verify_candidate_remaining

PYTHON='/root/miniconda3/bin/python'
STATE=WORK/'audit/two_ideas_queue_v1.json'
OWNS_STATE=False
QUEUE_LOCK=None


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def state(status,**kwargs):
    write(STATE,dict(status=status,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),**kwargs))


def gpu_idle():
    with (ROOT/'audit/GPU_JOB.lock').open('a') as file:
        try:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
    return True


def gpu(name,entry,arguments,seconds):
    assert gpu_idle(),'Preserve the exclusive GPU and existing queue order'
    assert not (ROOT/'audit'/(name+'_job.json')).exists(),'Preserve prior attempts'
    argv=[PYTHON,str(ROOT/'run_gpu_job_v2.py'),'--name',name,'--max-seconds',str(seconds),
          '--',PYTHON,'-u',str(ROOT/entry)]+arguments
    state('RUNNING_REGISTERED_JOB',name=name,argv=argv)
    with (WORK/'logs'/(name+'.log')).open('x') as log:
        return subprocess.call(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)


def cpu(entry,arguments):
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    subprocess.run([PYTHON,str(ROOT/entry)]+arguments,check=True,cwd=ROOT,env=env,timeout=7200)


def allocate_recipes(sequence):
    from brainhive_two_ideas_storage_v1 import protected_main
    templates=[read(path) for path in sequence['template_paths']]
    needed,main=protected_main()
    root_free=main['free'][0]-needed[WORK.stat().st_dev]
    data_free=main['free'][1]-needed[DATA.stat().st_dev]
    reserves=len(templates)*(16<<20)
    non_image=sum(c['training_artifact_budget_bytes']+c['feature_budget_bytes']+c['metadata_budget_bytes'] for c in templates)
    root_free-=reserves+non_image
    image_count=16600*len(templates)
    available=max(0,root_free)//330000+max(0,data_free)//330000
    if root_free<0 or available<image_count:
        return None,dict(new_non_image_bytes=non_image,new_image_bytes=image_count*330000,protected_main=main,
            available_after_main_and_new_metadata=[root_free,data_free],
            minimum_total_additional_bytes=max(0,non_image+reserves+image_count*330000-sum(
                f-n for f,n in zip(main['free'],main['remaining_required']))),
            C_D_large_files=False,purchases=False,training_started=False)
    output=WORK/'two_ideas_v1/recipes';assert not output.exists(),'Keep executed recipes'
    output.mkdir(parents=True)
    roots=[str(WORK/'allocated_images_two_ideas_v1'),str(DATA/'images/BrainHIVE_two_ideas_v1')]
    for root in roots:Path(root).mkdir(mode=0o700,parents=True,exist_ok=True)
    capacities=[max(0,root_free)//330000,max(0,data_free)//330000];recipes=[];paths=[]
    for template,template_path in zip(templates,sequence['template_paths']):
        recipe=dict(template);recipe['template_sha256']=sha(template_path);recipe['allowed_private_roots']=roots
        segments=[];start=0
        for index in [1,0]:
            count=min(16600-start,capacities[index])
            if count:
                segments.append(dict(start=start,stop=start+count,root=roots[index]));capacities[index]-=count;start+=count
        assert start==16600
        recipe['image_segments']=segments;recipes.append(recipe);paths.append(output/(recipe['candidate']+'.json'))
    report=forecast(recipes);assert report['funded'],report
    plan=output/'group_plan.json'
    write(plan,dict(recipes=[dict(path=str(p),core_sha256=core_sha(r)) for p,r in zip(paths,recipes)],
                    template_sequence_sha256=sha(sequence['_sequence_path'])))
    for path,recipe in zip(paths,recipes):
        reserve=output/(recipe['candidate']+'.metadata_reserve')
        fd=os.open(reserve,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
        try:os.posix_fallocate(fd,0,16<<20)
        finally:os.close(fd)
        receipt_path=output/(recipe['candidate']+'.admission.json')
        write(receipt_path,dict(status='FULL_TRAIN_GENERATE_SCORE_ADMITTED',candidate=recipe['candidate'],
            recipe_core_sha256=core_sha(recipe),scope=['training','actual_images','per_query_evaluation'],
            output_directory=recipe['output_directory'],max_training_artifact_bytes=recipe['training_artifact_budget_bytes'],
            reservation_paths=[dict(path=str(reserve),minimum_allocated_bytes=16<<20)],
            physical_metadata_reservation_bytes=16<<20,full_lifecycle_is_forecast_not_physical_reservation=True,
            minimum_training_free_bytes=512<<20,group_plan=str(plan),group_plan_sha256=sha(plan),
            lifecycle_reserved_bytes=16<<20,forecast_at_admission=report))
        recipe['storage_admission_receipt']=str(receipt_path);recipe['storage_admission_sha256']=sha(receipt_path)
        write(path,recipe)
    for recipe in recipes:verify_candidate_remaining(recipe,recipe['candidate'])
    return paths,report


def run_candidate(path):
    recipe=read(path);candidate=recipe['candidate'];train=recipe['training_entry']
    if gpu('BrainHIVE_'+candidate+'_engineering_v1',train,['--recipe',str(path),'--profile'],1800):
        return 'Engineering forward/backward failed; no formal training started'
    verify_candidate_remaining(recipe,candidate)
    if gpu('BrainHIVE_'+candidate+'_train_v1',train,['--recipe',str(path)],recipe['training_job_seconds']):
        return 'Training failed or finite timeout; saved artifacts retained, no automatic refit'
    if gpu('BrainHIVE_'+candidate+'_images_v1','generate_BrainHIVE_two_ideas_V_v1.py',
           ['--candidate',candidate,'--recipe',str(path)],14400):return 'Generation failure; saved images preserved'
    cpu('audit_BrainHIVE_two_ideas_image_arm_CPU_v1.py',['--candidate',candidate,'--recipe',str(path)])
    run=recipe['run_name'];features=WORK/'evaluation'/run/'features';manifest=WORK/'images'/run/'manifest.jsonl'
    gallery=WORK/'gallery/V/manifest.jsonl';gf=WORK/'evaluation/bounded_V_gallery_features_v1'
    if not (gf/'completion.json').exists():
        if gpu('BrainHIVE_two_ideas_gallery_features_v1','extract_image_features.py',
               ['--manifest',str(gallery),'--output',str(gf),'--real-gallery'],14400):return 'Shared evaluator failed'
    verify_candidate_remaining(recipe,candidate)
    if gpu('BrainHIVE_'+candidate+'_features_v1','extract_image_features.py',
           ['--manifest',str(manifest),'--output',str(features)],14400):return 'Image feature evaluation failed'
    cpu('score_full_manifest.py',['--manifest',str(manifest),'--features',str(features),'--gallery-manifest',str(gallery),
        '--gallery-features',str(gf),'--output',str(WORK/'evaluation'/run/'scores')])
    cpu('score_low_level.py',['--manifest',str(manifest),'--gallery-manifest',str(gallery),
        '--output',str(WORK/'evaluation'/run/'low_level')])
    cpu('compare_BrainHIVE_two_ideas_CPU_v1.py',['--recipe',str(path)])
    return None


def main():
    global OWNS_STATE,QUEUE_LOCK
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',required=True);args=parser.parse_args()
    QUEUE_LOCK=(WORK/'audit/two_ideas_queue_v1.lock').open('a')
    fcntl.flock(QUEUE_LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists(),'Do not replace a prior queue'
    OWNS_STATE=True
    sequence=read(args.sequence);sequence['_sequence_path']=args.sequence
    assert sequence['fixed_order']==['brainkan_full_v1','response_rank_v1']
    for path,digest in sequence['source_sha256'].items():assert sha(ROOT/path)==digest,path
    for path,digest in sequence['template_sha256'].items():assert sha(path)==digest,path
    started=time.monotonic()
    while True:
        assert time.monotonic()-started<sequence['maximum_wait_seconds'],'Finite admission wait expired'
        previous=read(WORK/'audit/bounded_full_lifecycle_queue_v1.json')
        terminal=previous['status'] in {'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED',
            'FINITE_MAIN_COMPARISON_PARTIAL_WITH_EXPLICIT_FAILURES','FAILED_PRESERVED_NEEDS_EXPLICIT_DIAGNOSIS'}
        if not terminal or not gpu_idle():
            state('WAITING_FOR_EXISTING_MAIN_LIFECYCLE_AND_EXCLUSIVE_GPU',fixed_order=sequence['fixed_order'],
                  previous_status=previous['status'],formal_new_training_started=False,next_check_seconds=600)
        else:
            try:paths,report=allocate_recipes(sequence)
            except AssertionError as error:
                if (WORK/'two_ideas_v1/recipes').exists():
                    raise RuntimeError('Partial admission preserved for explicit diagnosis: '+str(error)) from error
                state('WAITING_FOR_MAIN_STORAGE_SAFETY',error=str(error)[:1800],formal_new_training_started=False)
            else:
                if paths is not None:break
                state('WAITING_FOR_PERSISTENT_STORAGE',storage=report,fixed_order=sequence['fixed_order'],
                      formal_new_training_started=False,next_check_seconds=600)
        time.sleep(600)
    completed=[];failures={}
    for path in paths:
        candidate=read(path)['candidate']
        try:error=run_candidate(path)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)[:1800]
        if error:failures[candidate]=error
        else:completed.append(candidate)
        state('SAVED_CANDIDATE_AND_CONTINUING_FIXED_ORDER',completed=completed,failures=failures)
    state('TWO_REGISTERED_LIFECYCLES_FINISHED' if not failures else 'FINITE_PARTIAL_WITH_PRESERVED_FAILURES',
          completed=completed,failures=failures,no_automatic_hyperparameter_expansion=True,
          mechanism_confirmed=False,independent_generalization_confirmed=False)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OWNS_STATE:state('STOPPED_WITH_EXPLICIT_FAILURE',error_type=type(error).__name__,error=str(error)[:1800])
        raise
