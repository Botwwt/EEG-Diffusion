"""One engineering gate, four fixed full chunks; never fit or select a winner."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_overlap_gradient_contract_CPU_20261103_v1 import setup,resources

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);cli=parser.parse_args();started=time.monotonic()
    spec,base,G=setup(cli.protocol);root=Path(spec['output_root']);code=Path(cli.protocol).parent;out=root/'queue';assert not out.exists();out.mkdir();events=[]
    try:
        G.write(out/'started.json',dict(UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),PID=G.os.getpid(),protocol=G.ref(cli.protocol)))
        for part in ('engineering','0','1','2','3'):
            resources(spec,base,G,start=True)
            seconds=600 if part=='engineering' else 900;name='HIVE_EEG_overlap_gradient_20261103_v1_'+part
            command=[sys.executable,'-B',str(Path(base['repository_root'])/'run_gpu_job_v2.py'),'--name',name,'--max-seconds',str(seconds),'--',
                sys.executable,'-B',str(code/'probe_HIVE_overlap_gradients_GPU_20261103_v1.py'),'--protocol',cli.protocol,'--part',part,'--job-name',name]
            with (out/(part+'.stdout.log')).open('xb') as stdout,(out/(part+'.stderr.log')).open('xb') as stderr:
                then=time.monotonic();worker=subprocess.Popen(command,cwd=code,stdout=stdout,stderr=stderr,start_new_session=True)
                G.write(out/(part+'_process.json'),dict(PID=worker.pid,command=command,UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat()))
                returncode=worker.wait(timeout=seconds+90)
            job_path=Path(base['repository_root'])/'audit'/(name+'_job.json');job=G.read(job_path)
            assert returncode==0 and job['status']=='COMPLETED' and job['exit_code']==0 and job['wrapper_pid']==worker.pid
            complete=G.read(root/('GPU_'+part)/'completion.json')
            assert complete['status']=='COMPLETE_FROZEN_NATIVE_VALUE_FP32_EEG_OVERLAP_PARAMETER_GRADIENT_PROFILE'
            assert complete['new_parameter_updates']==0 and not complete['optimizer_created'] and not complete['new_images']
            if part=='engineering':
                assert len(complete['finite_differences'])==2 and len(complete['actual_gradient_identities'])==24
                # Forecast uses the engineering mean including four VJPs/cell, more than the full two-VJP loop.
                forecast=complete['seconds']+240*complete['cell_seconds_mean']
                assert forecast<840,('Full20query-chunk conservative forecast exceeds fixed original900s job',forecast)
            event=dict(part=part,seconds=time.monotonic()-then,returncode=returncode,GPU_job=G.ref(job_path),completion=G.ref(root/('GPU_'+part)/'completion.json'))
            events.append(event);G.write(out/(part+'_complete.json'),event)
        assert [i for part in ('0','1','2','3') for i in G.read(root/('GPU_'+part)/'completion.json')['indices']]==list(range(80))
        G.write(out/'completion.json',dict(status='COMPLETE_FROZEN_EEG_OVERLAP_GRADIENT_ENGINEERING_AND_ALL_T80_CHUNKS',UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),
            protocol=G.ref(cli.protocol),events=events,seconds=time.monotonic()-started,independent_CPU_audit_pending=True,
            new_parameter_updates=0,new_models=0,new_images=0,NEW_contacted=False))
    except BaseException as error:
        G.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-started,automatic_retry=False));raise
if __name__=='__main__':main()
