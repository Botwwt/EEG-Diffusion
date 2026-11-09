"""Both fixed1729 loss controls: profile, all4000 images, audit and all metrics."""
import argparse,datetime,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned


def ref(path):
    path=Path(path)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest)


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def run(study_path):
    study=json.loads(Path(study_path).read_bytes());assert study['host']=='old' and study['training_seed']==1729 and not study['new_training_allowed']
    root=Path(study['output_root']);root.mkdir();logs=root/'queue';logs.mkdir();code=Path(study_path).parent;R=Path(study['repository_root']);started=time.monotonic();events=[];owner=None;pinned=None;phase='INIT'
    def budget():
        assert time.monotonic()-started<21600 and shutil.disk_usage(root).free>2<<30
        assert sum(path.stat().st_size for path in root.rglob('*') if path.is_file())<study['output_cap_bytes']
    def call(name,command,GPU=False,seconds=1800):
        nonlocal owner,pinned
        budget();environment=dict(os.environ);environment.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if GPU:
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==study['GPU_UUID']
            command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command;seconds=900
        else:environment['CUDA_VISIBLE_DEVICES']=''
        with (logs/(name+'.stdout.log')).open('xb') as so,(logs/(name+'.stderr.log')).open('xb') as se:
            owner=subprocess.Popen(command,cwd=R,env=environment,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
            pinned=capture_wrapper_child(owner,R/'audit'/(name+'_job.json')) if GPU else None
            fields=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
            progress=dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),own_PID=owner.pid,start_ticks=int(fields[19]),command=command,events=events,new_training_updates=0,NEW_contacted=False)
            pending=logs/'progress.pending';pending.write_text(json.dumps(progress,indent=2)+'\n');os.replace(pending,logs/'progress.json')
            rc=owner.wait(timeout=seconds+60);owner=None;pinned=None
        event=dict(name=name,returncode=rc,GPU=GPU,seconds_since_start=time.monotonic()-started);events.append(event);write(logs/(name+'.receipt.json'),event);budget();return rc
    def guarded(execution,script,arguments,name):
        return [sys.executable,'-B',str(code/'run_HIVE_R80_evaluation_guarded_20261029_v1.py'),'--execution',str(execution),'--audit-file',str(root/phase.split(':')[0]/'evaluation_resources'/(name+'.json')),'--script',str(script),'--']+arguments
    try:
        for item in study['source_pins']+study['input_pins']:assert ref(item['path'])==item
        assert shutil.disk_usage(root).free>study['reserved_disk_bytes']
        executions={}
        for arm in ('F_only','token_only'):
            phase=arm+':ALL10_ENGINEERING';spec=code/(arm+'_protocol.json');s=json.loads(spec.read_bytes());out=root/arm;job='HIVE_loss_'+arm+'_R80_profile_20261030_v1'
            assert call(job,[sys.executable,'-B',str(code/'profile_HIVE_loss_control_R80_GPU_20261030_v1.py'),'--spec',str(spec),'--job-name',job],True)==0
            profile=out/'profile/completion.json';value=json.loads(profile.read_bytes());assert value['status']=='PASS_ALL10_FIXED_LOSS_CONTROL_R80_NATIVE_PIXELS_INPUT_AFFINES_RNG_AND_RESOURCES'
            assert value['all10_parent_pixels_exact'] and value['all10_loss_control_native_RNG_paired'] and value['extra_CPU_peak']<10<<30 and value['peak_GPU_allocated']<20<<30
            execution=dict(study['execution_template']);execution.update(status='BOUND_FIXED_LOSS_CONTROL_R80_FULL_LIFECYCLE_v1',protocol=ref(spec),profile_completion=ref(profile),source_pins=study['source_pins'],control_arm=arm,training_seed=1729,new_training_allowed=False)
            path=code/(arm+'_execution.json');write(path,execution);executions[arm]=path
        # Complete every declared image before any new effect score is read.
        for arm,execution in executions.items():
            phase=arm+':FULL2000_GENERATION'
            for lo in range(0,2000,200):
                job=f'HIVE_loss_{arm}_R80_sub{lo//200+1:02d}_20261030_v1'
                assert call(job,[sys.executable,'-B',str(code/'generate_HIVE_loss_control_R80_GPU_20261030_v1.py'),'--execution',str(execution),'--lo',str(lo),'--hi',str(lo+200),'--job-name',job],True)==0
                done=json.loads((root/arm/'generation_attempts'/f'{lo:04d}_{lo+200:04d}/completion.json').read_bytes());assert done['queries']==200 and not done['effect_scores_read']
        for arm,execution in executions.items():
            phase=arm+':FULL2000_ARTIFACT_AUDIT'
            assert call('artifact_'+arm,[sys.executable,'-B',str(code/'audit_HIVE_loss_control_R80_CPU_20261030_v1.py'),'--execution',str(execution)],False,1800)==0
        for arm,execution in executions.items():
            phase=arm+':ALL16_METRICS';out=root/arm;manifest=out/'loss_control/manifest.jsonl';features=out/'evaluation/features'
            for index in range(3):
                name=f'HIVE_loss_{arm}_R80_features{index}_20261030_v1'
                rc=call(name,guarded(execution,R/'extract_image_features.py',['--manifest',str(manifest),'--output',str(features)],name),True)
                if rc==0:break
                job=json.loads((R/'audit'/(name+'_job.json')).read_bytes());assert job.get('termination_reason')=='Finite registered job timeout; preserve resumable checkpoints'
                for path in features.rglob('*.tmp'):
                    assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(features.resolve());path.rename(path.with_name(path.name+f'.segment{index}_incomplete_retained'))
            else:raise RuntimeError('Fixed feature timeout segments exhausted')
            assert json.loads((features/'completion.json').read_bytes())['queries']==2000
            gallery=json.loads((code/(arm+'_protocol.json')).read_bytes())['gallery_manifest']['path'];gallery_features=study['execution_template']['gallery_features']
            for name,script,arguments in (
                ('common_'+arm,R/'score_full_manifest.py',['--manifest',str(manifest),'--features',str(features),'--gallery-manifest',gallery,'--gallery-features',gallery_features,'--output',str(out/'evaluation/scores')]),
                ('low_'+arm,R/'score_low_level.py',['--manifest',str(manifest),'--gallery-manifest',gallery,'--output',str(out/'evaluation/low_level')])):
                assert call(name,guarded(execution,script,arguments,name),False,1800)==0
            for index in range(3):
                name=f'HIVE_loss_{arm}_R80_author{index}_20261030_v1'
                rc=call(name,guarded(execution,code/'score_HIVE_loss_control_R80_author_GPU_20261030_v1.py',['--execution',str(execution)],name),True)
                if rc==0:break
                job=json.loads((R/'audit'/(name+'_job.json')).read_bytes());assert job.get('termination_reason')=='Finite registered job timeout; preserve resumable checkpoints'
            else:raise RuntimeError('Fixed author metric timeout segments exhausted')
        phase='ALL4_METHODS_FIXED_PAIRED_COMPARISONS'
        assert call('all4_fixed_comparison',[sys.executable,'-B',str(code/'compare_HIVE_loss_controls_R80_CPU_20261030_v1.py'),'--study',str(study_path)],False,1800)==0
        write(root/'completion.json',dict(status='COMPLETE_SINGLE_SEED_R80_TWO_LOSS_CONTROLS_AND_FIVE_ALL16_COMPARISONS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),study=ref(study_path),events=events,new_research_images=4000,reference_images_reused=4000,new_model_updates=0,training_seed=1729,seconds=time.monotonic()-started,NEW_contacted=False,SOTA_established=False))
    except BaseException as error:
        if owner is not None:stop_owned(owner,pinned)
        write(root/'failure.json',dict(status='LOSS_CONTROL_R80_FAILURE_PRESERVED_NO_SCIENCE_RETRY',phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-started));raise


if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Explicit own single-seed control study stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser();parser.add_argument('--study',required=True);run(parser.parse_args().study)
