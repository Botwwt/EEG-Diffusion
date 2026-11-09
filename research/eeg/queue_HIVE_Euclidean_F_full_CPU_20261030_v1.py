"""One fresh1729 F-objective geometry control, full fit and full R80 lifecycle."""
import argparse,copy,datetime,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_owned_process_CPU_20261029_v1 import capture_wrapper_child,stop_owned


def read(p): return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=digest)
def write(p,v):
    with Path(p).open('x') as f: json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def run(path):
    s=read(path);assert s['status']=='REGISTERED_SINGLE1729_EUCLIDEAN_F_GEOMETRY_FULL_LIFECYCLE'
    assert s['host']=='old' and s['training_seed']==1729 and not s['multiple_training_seeds_allowed']
    root=Path(s['output_root']);assert not root.exists();root.mkdir();logs=root/'queue';logs.mkdir()
    code=Path(path).parent;R=Path(s['repository_root']);started=time.monotonic();phase='INIT';events=[];owner=pinned=None
    def budget():
        assert time.monotonic()-started<21600 and shutil.disk_usage(root).free>2<<30
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<8<<30
    def call(name,command,GPU=False,seconds=1800):
        nonlocal owner,pinned
        budget();env=dict(os.environ);env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if GPU:
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            command=[sys.executable,'-B',str(R/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--']+command;seconds=900
        else: env['CUDA_VISIBLE_DEVICES']=''
        with (logs/(name+'.stdout.log')).open('xb') as so,(logs/(name+'.stderr.log')).open('xb') as se:
            owner=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
            pinned=capture_wrapper_child(owner,R/'audit'/(name+'_job.json')) if GPU else None
            fields=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
            value=dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),own_PID=owner.pid,start_ticks=int(fields[19]),command=command,events=events,training_seed=1729,NEW_contacted=False)
            pending=logs/'progress.pending';pending.write_text(json.dumps(value,indent=2)+'\n');os.replace(pending,logs/'progress.json')
            rc=owner.wait(timeout=seconds+60);owner=pinned=None
        event=dict(name=name,returncode=rc,GPU=GPU,seconds_since_start=time.monotonic()-started);events.append(event);write(logs/(name+'.receipt.json'),event);budget();return rc
    def guarded(execution,script,args,name):
        return [sys.executable,'-B',str(code/'run_HIVE_R80_evaluation_guarded_20261029_v1.py'),'--execution',str(execution),'--audit-file',str(root/'R80/evaluation_resources'/(name+'.json')),'--script',str(script),'--']+args
    try:
        for item in s['source_pins']+s['input_pins']:assert ref(item['path'])==item
        assert shutil.disk_usage(root).free>12<<30
        phase='REAL_F_GRADIENT_PROFILE_TEMPORARY_WEIGHTS_DISCARDED'
        assert call('HIVE_EuclidF_fitprofile_20261030_v1',[sys.executable,'-B',str(code/'fit_HIVE_Euclidean_F_GPU_20261030_v1.py'),'--protocol',str(path),'--mode','profile'],True)==0
        profile=read(root/'fit_profile/completion.json')
        assert profile['status']=='PASS_REAL_F_OBJECTIVE_EUCLIDEAN_COMPONENT_ENGINEERING_ONLY' and profile['scientific_updates']==0
        assert profile['frozen_AP_unchanged'] and profile['real_F_gradient_affine_groups'] and profile['temporary_weights_not_saved']
        phase='FRESH1729_FULL_T_FIXED24807_FIT_AND_DFIT_ONLY_CALIBRATION'
        assert call('HIVE_EuclidF_fullfit_20261030_v1',[sys.executable,'-B',str(code/'fit_HIVE_Euclidean_F_GPU_20261030_v1.py'),'--protocol',str(path),'--mode','train'],True)==0
        done=read(root/'fit/completion.json');assert done['updates']==24807 and done['query_exposures']==396900
        phase='ALL132300_STATE_AND_INDEPENDENT_NUMPY_CPU_AUDIT'
        assert call('HIVE_EuclidF_fit_CPUaudit',[sys.executable,'-B',str(code/'audit_HIVE_Euclidean_F_fit_CPU_20261030_v1.py'),'--protocol',str(path)],False)==0
        audit=read(root/'fit/independent_CPU_audit.json');assert audit['status']=='PASS_FULL132300_EUCLIDEAN_F_STATE_COVERAGE_DFIT_AND_NUMPY_FORWARD_AUDIT'
        # F fitting has no generator calls. Actual generator parity is a separate
        # mandatory postfit gate before any research generation, not a fit claim.
        spec=copy.deepcopy(s['R80_spec_template']);spec.update(status='REGISTERED_FIXED_EUCLIDEAN_F_R80_MATCHED_BENCHMARK_v1',output_root=str(root/'R80'),source_pins=s['source_pins'],control_arm='Euclidean_F',loss_control=done['component'],loss_control_strength=audit['selected_strength'],loss_control_radius=.25,training_seed=1729,new_training_allowed=False,geometry='UNBOUNDED_AFFINE_RESIDUAL',radius_parameter_is_residual_scale_only=True)
        spec_path=code/'Euclidean_F_R80_protocol.json';write(spec_path,spec)
        phase='ALL10_NATIVE_ZERO_DISABLE_TRAINED_GENERATOR_PIXEL_PROFILE'
        name='HIVE_EuclidF_R80_profile_20261030_v1'
        assert call(name,[sys.executable,'-B',str(code/'profile_HIVE_Euclidean_F_R80_GPU_20261030_v1.py'),'--spec',str(spec_path),'--job-name',name],True)==0
        profile_path=root/'R80/profile/completion.json';p=read(profile_path)
        assert p['status']=='PASS_ALL10_FIXED_EUCLIDEAN_F_R80_NATIVE_PIXELS_INPUT_AFFINES_RNG_AND_RESOURCES'
        assert p['all10_zero_and_disabled_pixels_exact'] and p['all10_parent_pixels_exact'] and p['all10_loss_control_native_RNG_paired']
        execution=copy.deepcopy(s['R80_execution_template']);execution.update(status='BOUND_FIXED_EUCLIDEAN_F_R80_FULL_LIFECYCLE_v1',protocol=ref(spec_path),profile_completion=ref(profile_path),source_pins=s['source_pins'],control_arm='Euclidean_F',training_seed=1729,new_training_allowed=False)
        execution_path=code/'Euclidean_F_R80_execution.json';write(execution_path,execution)
        phase='ALL2000_NORMAL_EEG_IMAGES_BEFORE_ANY_NEW_EFFECT_READ'
        for lo in range(0,2000,200):
            name=f'HIVE_EuclidF_R80_sub{lo//200+1:02d}_20261030_v1'
            assert call(name,[sys.executable,'-B',str(code/'generate_HIVE_Euclidean_F_R80_GPU_20261030_v1.py'),'--execution',str(execution_path),'--lo',str(lo),'--hi',str(lo+200),'--job-name',name],True)==0
            part=read(root/'R80/generation_attempts'/f'{lo:04d}_{lo+200:04d}/completion.json');assert part['queries']==200 and not part['effect_scores_read']
        phase='ALL2000_ARTIFACT_IDENTITY_NOISE_AND_NUMPY_AUDIT'
        assert call('HIVE_EuclidF_R80_artifact',[sys.executable,'-B',str(code/'audit_HIVE_Euclidean_F_R80_CPU_20261030_v1.py'),'--execution',str(execution_path)],False)==0
        phase='ALL16_METRICS';out=root/'R80';manifest=out/'loss_control/manifest.jsonl';features=out/'evaluation/features'
        for index in range(3):
            name=f'HIVE_EuclidF_R80_features{index}_20261030_v1'
            rc=call(name,guarded(execution_path,R/'extract_image_features.py',['--manifest',str(manifest),'--output',str(features)],name),True)
            if rc==0:break
            job=read(R/'audit'/(name+'_job.json'));assert job.get('termination_reason')=='Finite registered job timeout; preserve resumable checkpoints'
            for p in features.rglob('*.tmp'):
                assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(features.resolve());p.rename(p.with_name(p.name+f'.segment{index}_incomplete_retained'))
        else:raise RuntimeError('Registered feature timeout prefix segments exhausted')
        assert read(features/'completion.json')['queries']==2000
        gallery=spec['gallery_manifest']['path'];gf=execution['gallery_features']
        for name,script,args in (
            ('HIVE_EuclidF_R80_common',R/'score_full_manifest.py',['--manifest',str(manifest),'--features',str(features),'--gallery-manifest',gallery,'--gallery-features',gf,'--output',str(out/'evaluation/scores')]),
            ('HIVE_EuclidF_R80_low',R/'score_low_level.py',['--manifest',str(manifest),'--gallery-manifest',gallery,'--output',str(out/'evaluation/low_level')])):
            assert call(name,guarded(execution_path,script,args,name),False)==0
        for index in range(3):
            name=f'HIVE_EuclidF_R80_author{index}_20261030_v1'
            rc=call(name,guarded(execution_path,code/'score_HIVE_Euclidean_F_R80_author_GPU_20261030_v1.py',['--execution',str(execution_path)],name),True)
            if rc==0:break
            job=read(R/'audit'/(name+'_job.json'));assert job.get('termination_reason')=='Finite registered job timeout; preserve resumable checkpoints'
        else:raise RuntimeError('Registered author timeout prefix segments exhausted')
        phase='ALL8000_IMAGE_JOINS_AND_THREE_FIXED_ALL16_COMPARISONS'
        assert call('HIVE_EuclidF_R80_comparison',[sys.executable,'-B',str(code/'compare_HIVE_Euclidean_F_R80_CPU_20261030_v1.py'),'--study',str(path)],False)==0
        for item in (s['immutable_ordinary'],s['bounded_F_component']):assert ref(item['path'])==item
        write(root/'completion.json',dict(status='COMPLETE_SINGLE1729_EUCLIDEAN_F_WHOLE_GEOMETRY_FULL_FIT_AND_R80_ALL16_THREE_COMPARISONS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),study=ref(path),events=events,scientific_updates=24807,training_query_exposures=396900,new_research_images=2000,reference_images_reused=6000,training_seed=1729,seconds=time.monotonic()-started,selected_strength=audit['selected_strength'],NEW_contacted=False,whole_geometry_only=True,SOTA_established=False))
    except BaseException as error:
        if owner is not None:stop_owned(owner,pinned)
        write(root/'failure.json',dict(status='EUCLIDEAN_F_FAILURE_PRESERVED_NO_SCIENCE_RETRY',phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-started));raise


if __name__=='__main__':
    def stop(*unused):raise InterruptedError('Own Euclidean F controller stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);run(p.parse_args().study)
