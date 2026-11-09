"""Finite profile -> matched full-T fit -> independent checkpoint CPU audit."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def write(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def run(path):
    s=json.loads(Path(path).read_bytes());code=Path(s['code']);host=s['host'];started=time.monotonic();root=Path(s['queue_root']);root.mkdir();events=[]
    try:
        for item in s['source_pins']:assert ref(Path(item['path']))==item
        wrapper='run_gpu_job_v2.py' if host=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'
        def stage(mode,spec,index):
            assert time.monotonic()-started<3200
            job=f'EEG_native_marginal_{host}_{mode}{index}_20261029_v2'
            cmd=['/root/miniconda3/bin/python','-B',str(Path(s['repository_root'])/wrapper),'--name',job,'--max-seconds','900','--','/root/miniconda3/bin/python','-B',str(code/'train_EEG_native_marginal_20261029_v2.py'),'--spec',str(spec),'--job-name',job,'--mode',mode]
            with (root/(job+'.stdout.log')).open('xb') as so,(root/(job+'.stderr.log')).open('xb') as se:
                p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,cwd=str(code))
                try:rc=p.wait(timeout=950)
                except subprocess.TimeoutExpired:
                    # The wrapper owns GPU cleanup. Do not start another stage
                    # while an exceptional wrapper is still present.
                    raise RuntimeError('Wrapper exceeded its own finite deadline; no next stage dispatched')
            job_result=json.loads((Path(s['repository_root'])/'audit'/(job+'_job.json')).read_bytes())
            record=dict(job=job,returncode=rc,wrapper_receipt=ref(Path(s['repository_root'])/'audit'/(job+'_job.json')));events.append(record)
            assert rc==0,record
            spec_value=json.loads(spec.read_bytes());done=Path(spec_value['output_root'])/(mode+'_'+job)/'completion.json';result=json.loads(done.read_bytes())
            assert result['spec']==ref(spec) and result['pretrained_VAR_VQ_unchanged']
            return result,done
        ps=Path(s['profile_spec']['path']);assert ref(ps)==s['profile_spec']
        result,done=stage('profile',ps,1)
        assert result['status']=='PASS_REAL_64_QUERY_NATIVE_MARGINAL_PROFILE_ALL_TEMPORARY_UPDATES_RESET'
        assert result['peak_GPU_allocated']<20<<30 and result['extra_CPU_peak']<10<<30
        assert result['native_full_vs_marginal_logits_max_error']<2e-5
        value=json.loads(ps.read_bytes());value['output_root']=s['fit_root'];value['profile_completion']=ref(done)
        fit_spec=code/f'fit_{host}_marginal_spec_v2.json';write(fit_spec,value)
        for segment in range(1,4):
            result,done=stage('fit',fit_spec,segment)
            if result['status']=='FULL_T_NATIVE_MARGINAL_FIXED_FINAL_FIT_COMPLETE':break
            assert result['status']=='WHOLE_EPOCH_SEGMENT_CLOSED_REQUIRES_NEXT_FINITE_SEGMENT'
        assert result['status']=='FULL_T_NATIVE_MARGINAL_FIXED_FINAL_FIT_COMPLETE','Finite three-segment budget exhausted, retained partial result'
        # Independent CPU state checks do not reuse the producer's assertions.
        import torch
        state=torch.load(result['checkpoint']['path'],weights_only=False,map_location='cpu')
        assert ref(Path(result['checkpoint']['path']))==result['checkpoint']
        assert state['epoch']==10 and state['updates']==2070 and len(state['coverage'])==10 and len(state['events'])==2070
        assert all(item['queries']==13230 for item in state['coverage'])
        assert bool((state['posterior']['count']==40).all())
        assert all(torch.isfinite(value).all() for value in state['EEG_model'].values())
        assert all(torch.isfinite(value).all() for value in state['posterior'].values())
        opt=state['EEG_optimizer'];steps=[int(v['step']) for v in opt['state'].values()]
        assert steps and set(steps)=={2070}
        assert all(torch.isfinite(t).all() for v in opt['state'].values() for t in v.values() if isinstance(t,torch.Tensor))
        assert ref(Path(result['final_model']['path']))==result['final_model']
        final=torch.load(result['final_model']['path'],weights_only=True,map_location='cpu')
        assert set(final['model'])==set(state['EEG_model']) and all(torch.equal(final['model'][k],v) for k,v in state['EEG_model'].items())
        assert state['CPU_RNG'].dtype==torch.uint8 and state['latent_RNG'].dtype==torch.uint8 and len(state['CUDA_RNG'])==1
        write(root/'independent_CPU_audit.json',dict(status='PASS_FULL_T_FIXED_FINAL_EEG_POSTERIOR_OPTIMIZER_COVERAGE_AND_RNG_CPU',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),epoch=10,EEG_updates=2070,query_exposures=132300,posterior_updates_per_query=40,optimizer_states=len(steps),checkpoint=result['checkpoint'],final_model=result['final_model'],new_research_images=0))
        write(root/'completion.json',dict(status='NEW_ROUTE_FULL_T_FIT_AND_INDEPENDENT_CPU_AUDIT_CLOSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-started,host=host,arm=s['arm'],events=events,fit_completion=ref(done),CPU_audit=ref(root/'independent_CPU_audit.json'),new_research_images=0,benefit_or_SOTA_established=False))
    except BaseException as error:
        write(root/'failure.json',dict(status='FINITE_NEW_ROUTE_QUEUE_STOPPED_FAILURE_PRESERVED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-started,error=repr(error),traceback=traceback.format_exc(),events=events));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();run(a.spec)
