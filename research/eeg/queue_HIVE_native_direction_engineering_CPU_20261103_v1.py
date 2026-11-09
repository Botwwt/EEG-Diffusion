"""One finite native-encoder engineering job, never starts a scientific fit."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_native_direction_contract_20261103_v1 import setup,resources
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);cli=parser.parse_args();started=time.monotonic();s,old,base,G,H=setup(cli.protocol)
 root=Path(s['output_root']);code=Path(cli.protocol).parent;out=root/'queue';assert not out.exists();out.mkdir()
 try:
  resources(s,old,base,G,H,True)
  CPU_command=[sys.executable,'-B',str(code/'prepare_HIVE_native_direction_EEG_CPU_20261103_v1.py'),'--protocol',cli.protocol]
  with (out/'CPU.stdout.log').open('xb') as stdout,(out/'CPU.stderr.log').open('xb') as stderr:
   cpu=subprocess.Popen(CPU_command,cwd=code,stdout=stdout,stderr=stderr,start_new_session=True);G.write(out/'CPU_process.json',dict(PID=cpu.pid,command=CPU_command));cpu_return=cpu.wait(timeout=240)
  assert cpu_return==0 and G.read(root/'inputs/completion.json')['status']=='COMPLETE_REAL_T80_RAW_R4_TEN_FULL_PARENT_CPU_INPUTS_NO_CUDA'
  name='HIVE_native_EEG_direction_engineering_20261103_v1';command=[sys.executable,'-B',str(Path(base['repository_root'])/'run_gpu_job_v2.py'),'--name',name,'--max-seconds','900','--',sys.executable,'-B',str(code/'profile_HIVE_native_direction_EEG_GPU_20261103_v1.py'),'--protocol',cli.protocol,'--job-name',name]
  with (out/'GPU.stdout.log').open('xb') as stdout,(out/'GPU.stderr.log').open('xb') as stderr:
   worker=subprocess.Popen(command,cwd=code,stdout=stdout,stderr=stderr,start_new_session=True);G.write(out/'GPU_process.json',dict(PID=worker.pid,command=command,UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat()));returncode=worker.wait(timeout=990)
  job_path=Path(base['repository_root'])/'audit'/(name+'_job.json');job=G.read(job_path)
  assert returncode==0 and job['status']=='COMPLETED' and job['exit_code']==0 and job['wrapper_pid']==worker.pid
  complete=G.read(root/'GPU/completion.json');assert complete['status']=='COMPLETE_NATIVE_EEG_DIRECTIONAL_ENGINEERING_ONLY_NOT_FORMAL_TRAINING'
  assert len(complete['FP32_gradient_identities'])==20 and len(complete['finite_differences'])==2 and len(complete['temporary_batch16_BF16_updates'])==3
  assert time.monotonic()-started<1500
  G.write(out/'completion.json',dict(status='COMPLETE_NATIVE_EEG_DIRECTIONAL_ENGINEERING_QUEUE_NOT_FIT_ADMISSION',protocol=G.ref(cli.protocol),inputs=G.ref(root/'inputs/completion.json'),profile=G.ref(root/'GPU/completion.json'),job=G.ref(job_path),
   seconds=time.monotonic()-started,independent_CPU_audit_pending=True,new_formal_updates=0,new_models=0,new_images=0,NEW_contacted=False))
 except BaseException as error:G.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,automatic_retry=False));raise
if __name__=='__main__':main()
