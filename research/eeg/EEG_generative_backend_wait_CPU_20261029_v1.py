"""Registered finite wait, then exactly one original-cap old-GPU profile."""
import argparse,datetime,hashlib,json,subprocess,time,traceback
from pathlib import Path
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(root):
    start=time.monotonic();spec_path=root/'spec.json';spec=json.loads(spec_path.read_bytes());M=root.parent;R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    try:
        # Use already verified VQ weights for the necessary full-T token bank
        # while the larger immutable VAR checkpoint is still downloading.
        previous=json.loads((R/'audit/EEG_memory_old_export_20261029_v1_job.json').read_bytes())
        assert previous['status']=='COMPLETED' and previous['exit_code']==0
        for key in ['VQ_checkpoint','memory_completion']:
            assert sha(Path(spec[key]['path']))==spec[key]['sha256']
        import numpy as np
        token_root=Path(spec['token_output']);token_root.mkdir();segments=[]
        for index in range(7):
            assert time.monotonic()-start<27000
            command=['/root/miniconda3/bin/python','-B',str(R/'run_gpu_job_v2.py'),'--name',f'EEG_VQ_T_tokens_s{index:02d}_20261029_v1','--max-seconds','900','--','/root/miniconda3/bin/python','-B',str(root/'code/EEG_generative_backend_VQ_tokens_GPU_20261029_v1.py'),'--spec',str(spec_path),'--segment',str(index)]
            (root/'queue_progress.json').write_text(json.dumps(dict(status='COMPLETE_T_VQ_TOKEN_BANK_GPU_STAGE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),segment=index,command=command),indent=2)+'\n',encoding='utf8')
            with (root/f'VQ_segment_{index:02d}.stdout.log').open('xb') as out,(root/f'VQ_segment_{index:02d}.stderr.log').open('xb') as err:result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=950)
            assert result.returncode==0,(index,result.returncode)
            path=token_root/f'segment_{index:02d}/completion.json';record=json.loads(path.read_bytes());assert record['status']=='FROZEN_STRICT_VQ_COMPLETE_T_SEGMENT_CLOSED';segments.append(record)
        joined=[];bank=np.lib.format.open_memmap(token_root/'T_VQ_indices.npy',mode='w+',dtype=np.int16,shape=(13230,680));position=0
        for record in segments:
            assert record['query_start']==position
            index_path=Path(record['indices']['path']);assert sha(index_path)==record['indices']['sha256'];data=np.load(index_path,allow_pickle=False)
            assert data.dtype==np.int16 and data.shape==(record['queries'],680) and data.min()>=0 and data.max()<4096
            bank[position:record['query_end']]=data;position=record['query_end'];rows_path=Path(record['identity_rows']['path']);assert sha(rows_path)==record['identity_rows']['sha256'];joined.extend(json.loads(rows_path.read_bytes()))
        assert position==13230 and [q['query_index'] for q in joined]==list(range(13230)) and len({q['stimulus_id'] for q in joined})==13230 and len({q['concept_id'] for q in joined})==1323
        bank.flush();del bank;(token_root/'T_VQ_identity_rows.json').write_text(json.dumps(joined,indent=2)+'\n',encoding='utf8')
        def ref(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
        (token_root/'completion.json').write_text(json.dumps(dict(status='COMPLETE_T13230_FROZEN_STRICT_VQ_TOKEN_BANK_INDEPENDENT_CPU_JOIN_AUDITED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),queries=13230,concepts=1323,segments=segments,indices=ref(token_root/'T_VQ_indices.npy'),identities=ref(token_root/'T_VQ_identity_rows.json'),token_width=680,indices_dtype='int16',source_images_T_only=True,VQ_weights_frozen=True,random_augmentation=False,trained_models=0,new_research_images=0),indent=2)+'\n',encoding='utf8')
        while True:
            assert time.monotonic()-start<27000
            for pin in spec['source_pins']:assert sha(Path(pin['path']))==pin['sha256'],pin['path']
            ready=all(Path(spec[key]['path']).is_file() for key in ['VQ_checkpoint','VAR_checkpoint','memory_completion'])
            job=R/'audit/EEG_memory_old_export_20261029_v1_job.json';previous=json.loads(job.read_bytes()) if job.exists() else None
            old_complete=previous is not None and previous['status']=='COMPLETED' and previous['exit_code']==0
            idle=not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            progress=dict(status='FINITE_OLD_GPU_BACKEND_WAITING_PINNED_PUBLIC_BYTES_AND_PRIOR_EXPORT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),public_bytes_and_memory_exist=ready,previous_memory_GPU_terminal=old_complete,GPU_idle=idle,new_GPU_jobs=0)
            (root/'queue_progress.json').write_text(json.dumps(progress,indent=2)+'\n',encoding='utf8')
            if ready and old_complete and idle:break
            # Public transport failure is retained as evidence; a transport-only
            # repair may still deliver the same immutable bytes before deadline.
            time.sleep(15)
        for key in ['VQ_checkpoint','VAR_checkpoint','memory_completion']:
            ref=spec[key];p=Path(ref['path']);assert sha(p)==ref['sha256'] and ('bytes' not in ref or p.stat().st_size==ref['bytes'])
        command=['/root/miniconda3/bin/python','-B',str(R/'run_gpu_job_v2.py'),'--name','EEG_generative_backend_engineering_20261029_v1','--max-seconds','900','--','/root/miniconda3/bin/python','-B',str(root/'code/EEG_generative_backend_profile_GPU_20261029_v1.py'),'--spec',str(spec_path)]
        (root/'GPU_dispatch.json').write_text(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command),indent=2)+'\n',encoding='utf8')
        with (root/'GPU_stdout.log').open('xb') as out,(root/'GPU_stderr.log').open('xb') as err:result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=950)
        assert result.returncode==0,result.returncode
        completion=json.loads((root/'profile_v1/completion.json').read_bytes());assert completion['status']=='PASS_REAL_T_GENUINE_OOF_FULL_PRETRAINED_VQ_VAR_EEG_ENGINEERING'
        (root/'queue_completion.json').write_text(json.dumps(dict(status='OLD_GPU_REAL_PRETRAINED_GENERATIVE_BACKEND_PROFILE_ACTUALLY_CLOSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-start,profile_completion=str(root/'profile_v1/completion.json'),profile_sha256=sha(root/'profile_v1/completion.json'),formal_models=0,new_research_images=0),indent=2)+'\n',encoding='utf8')
    except BaseException as error:
        (root/'queue_failure.json').write_text(json.dumps(dict(status='GENERATIVE_BACKEND_QUEUE_FAILURE_PRESERVED_NO_AUTOMATIC_FIT_OR_RETRY',error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start),indent=2)+'\n',encoding='utf8');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
