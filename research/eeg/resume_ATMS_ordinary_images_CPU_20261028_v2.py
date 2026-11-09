"""Continue X01 matched images after an observed controller-only log collision.

Uses original immutable trainer, image producers, scorer, policy, learned final
weights, calibration, conditions, noise, and committed image prefix. No fitting.
"""
import argparse,datetime,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import run_ATMS_standard_ordinary_pipeline_CPU_20261028_v1 as V

def progress_compatible(original):
    def report(event,**values):
        if 'stage' in values:values['image_stage']=values.pop('stage')
        return original(event,**values)
    return report

def run(sp):
    began=time.monotonic();s=V.read(sp);assert s['status']=='X01_IMAGE_ONLY_CONTINUATION_AFTER_LOG_KEY_COLLISION_LOCKED'
    for x in s['source_pins']:V.check(x)
    V.check(s['original_failure']);V.check(s['training_completion']);V.check(s['calibration_completion']);policy=V.check(s['image_policy'])
    output=Path(s['image_output']);assert output==V.OUTPUT/'paired_images'
    before={x['path']:V.sha(V.check(x)) for x in s['protected_outputs']}
    original=V.progress;V.progress=progress_compatible(original)
    recovery=V.AUDIT/'image_continuation_v2';assert not recovery.exists();recovery.mkdir()
    common=['--spec',str(policy),'--output',str(output)];steps=[]
    try:
        with (V.AUDIT/'ordinary_pipeline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert not Path('/proc/240115/cmdline').exists(),'Old controller must be terminal, no preemption'
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            assert V.read(output/'prepared/completion.json')['status']=='ALL400_ATMS_NATIVE_ORDINARY_CONDITIONS_NOISE_FIXED_BEFORE_IMAGES'
            for stage,maximum in [(80,16),(400,64)]:
                for serial in range(1,maximum+1):
                    if (output/f'image_closure{stage}.json').exists():break
                    assert time.monotonic()-began<18000
                    for x in s['source_pins']:V.check(x)
                    count=len(list((output/'images').glob('*/*.json')));args=common+['--mode','generate','--stage',str(stage),'--max-seconds','1800']
                    if count:args.append('--resume')
                    name=f'X01_resume_v2_images{stage}_{serial:03d}_20261028'
                    V.gpu_stage(name,V.IMAGE_GENERATOR,args,1800)
                    after=len(list((output/'images').glob('*/*.json')));assert count<after<=2*stage
                    V.progress('MATCHED_ATMS_IMAGE_PREFIX_CONTINUED_WITHOUT_REFIT',stage=stage,images=after,expected_images=2*stage)
                assert V.read(output/f'image_closure{stage}.json')['status']=='COMPLETE_FIXED_ATMS_PAIRED_IMAGE_STAGE'
                V.cpu_stage(f'X01_resume_v2_images{stage}_CPU_audit',V.IMAGE_GENERATOR,common+['--mode','audit','--stage',str(stage),'--max-seconds','600'],660)
                V.gpu_stage(f'X01_resume_v2_features{stage}_20261028',V.IMAGE_SCORER,common+['--mode','features','--stage',str(stage),'--max-seconds','600'],600)
                V.cpu_stage(f'X01_resume_v2_score{stage}',V.IMAGE_SCORER,common+['--mode','score','--stage',str(stage),'--max-seconds','600'],660)
                result_path=output/f'score{stage}/completion.json';v=V.read(result_path);action=v['decision']['action'];steps.append(dict(stage=stage,score=V.ref(result_path),decision=v['decision']))
                assert v['status']=='COMPLETE_ATMS_ORDINARY_PAIRED_DINO_SCREEN'
                if stage==80 and action=='EXPAND400':continue
                assert action in ['STOP_INSTANCE','PROMISING_FOR_FULL_EVALUATION']
                assert all(V.sha(path)==digest for path,digest in before.items())
                V.atomic(V.AUDIT/'actual_image_screen_completion.json',dict(status='X01_MATCHED_ATMS_NATIVE_ORDINARY_ACTUAL_IMAGE_SCREEN_CLOSED',UTC=V.utc(),stage=stage,decision=v['decision'],score=V.ref(result_path),image_policy=V.ref(policy),mean_metrics=v['means'],paired=v['ordinary_minus_native'],independent_confirmation=False,SOTA_established=False,full_evaluation_required_if_promising=action=='PROMISING_FOR_FULL_EVALUATION',continuation=V.ref(sp)))
                V.atomic(recovery/'completion.json',dict(status='X01_EXISTING_FULL_FIT_AND_PAIRED_IMAGES_SCORES_CONTINUED_WITHOUT_REFIT',UTC=V.utc(),spec=V.ref(sp),original_failure=s['original_failure'],steps=steps,new_training_updates=0,existing_model_calibration_noise_and_images_unchanged=True,seconds=time.monotonic()-began));return
    except BaseException as error:
        V.cleanup_owned();V.atomic(recovery/'failure.json',dict(status='X01_IMAGE_CONTINUATION_FAILURE_PRESERVED',error=repr(error),traceback=traceback.format_exc(),steps=steps,new_training_updates=0));raise

if __name__=='__main__':
    def interrupted(*_):raise InterruptedError('Explicit image continuation stop')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
