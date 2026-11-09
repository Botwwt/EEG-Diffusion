"""One frozen prior-bypass factorial block; preserve every prefix."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,datetime,fcntl,gc,hashlib,io,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--arm',required=True);ap.add_argument('--stage',type=int,required=True);ap.add_argument('--start',type=int,required=True);ap.add_argument('--part',type=int,required=True);args=ap.parse_args()
q=json.loads(Path(args.policy).read_bytes());out=Path(q['output_root']);sys.path.insert(0,str(out))
import atm_error_geometry_contract_20261104_v1 as K
assert K.sha(__file__)==q['sources']['generate']['sha256'] and args.arm in K.ARMS and args.stage in (80,400)
assert 0<=args.part<2 and args.start%80==0 and args.start+80<=args.stage
assert not (args.stage==80 and args.arm=='parent')
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
parent,train,e,C=K.setup(q);before=K.resources(q,parent,train,e,C,starting=True)
prepared=K.read(K.checked(K.ref(out/'prepared.json')));assert prepared['status']=='COMPLETE_FIXED_FINAL_NATIVE_ATM_FOUR_ARM_CONDITIONS_CPU'
assert prepared['policy']['sha256']==K.sha(args.policy)
values=np.load(K.checked(prepared['conditions']),mmap_mode='r');queries=K.read(K.checked(prepared['queries']))
repo=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
name=q['job_prefix']+f'_gen{args.stage}_{args.arm}_{args.start}_{args.part}'
job=K.read(repo/'audit'/(name+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==parent['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(parent['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
from atm_native_training_runtime_20261103_v1 import load_host,tensor_sha
from brainhive_native_direction_contract_20261103_v1 import tensor_digest
target_meta=K.read(K.checked(prepared['target_metadata_ref']))['Dcheck']
baseline=Path(q['baseline80_root']);old_rows=[json.loads(s) for s in (baseline/'images.jsonl').read_text().splitlines()]
directory=out/'images'/args.arm;directory.mkdir(parents=True,exist_ok=True)
journal=directory/'images.jsonl';rows=[json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
base_index=80 if args.arm=='parent' else 0
assert [r['query_index'] for r in rows]==list(range(base_index,base_index+len(rows)))
begin=base_index+len(rows);assert args.start<=begin<=args.start+80
for row in rows[max(0,args.start-base_index):]:K.checked(row['image'])
loader_path=out/('loader_'+name);loader_path.mkdir()
phase='LOAD_NATIVE_HOST'
def fast_guard():
    assert time.monotonic()-started<850
    used=sum(p.stat().st_blocks*512 for p in Path(q['project_root']).rglob('*') if p.is_file())
    free=os.statvfs(q['project_root']);available=free.f_bavail*free.f_frsize
    assert used<parent['maximum_output_bytes'] and available>=before['required_remaining_bytes']+parent['maximum_output_bytes']-used
    assert torch.cuda.max_memory_allocated()<parent['maximum_GPU_bytes']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<parent['maximum_extra_CPU_bytes']
try:
    pipe,projection,prompt,pooled,timeids,diffusion,loader=load_host(q,loader_path)
    frozen=tensor_digest(pipe.unet,half=True);frozen_vae=tensor_digest(pipe.vae,half=True)
    import diffusers.schedulers.scheduling_euler_ancestral_discrete as EA
    old_noise=EA.randn_tensor;old_prepare=pipe.prepare_latents;capture={}
    def capture_noise(*a,**kw):
        result=old_noise(*a,**kw);capture['ancestral'].append(tensor_sha(result));return result
    def capture_initial(*a,**kw):
        result=old_prepare(*a,**kw);capture['initial'].append(tensor_sha(result));return result
    EA.randn_tensor=capture_noise;pipe.prepare_latents=capture_initial
    from PIL import Image
    def generate(index,arm):
        value=torch.tensor(values[index,K.ARMS.index(arm)][None].copy(),device='cuda')
        with torch.inference_mode():
            tokens=projection(value.half())
            image=pipe(prompt_embeds=prompt,pooled_prompt_embeds=pooled,height=512,width=512,ip_adapter_image_embeds=[value[:,None].half()],num_inference_steps=4,guidance_scale=0.,generator=torch.Generator(device='cuda').manual_seed(queries[index]['image_seed'])).images[0]
        return value,tokens,image
    try:
        if args.stage==80 and args.arm=='native_v' and args.start==0 and args.part==0:
            capture={'initial':[],'ancestral':[]};value,tokens,replay=generate(0,'parent')
            actual=np.asarray(replay.convert('RGB')).copy()
            assert hashlib.sha256(actual.tobytes()).hexdigest()==old_rows[0]['pixels_sha256'] and capture==old_rows[0]['noise']
            replay.save(out/'native_q000_engineering_replay.webp',format='WEBP',lossless=True,quality=100)
        phase='FROZEN_NATIVE_GENERATION'
        for index in range(begin,args.start+80):
            if time.monotonic()-started>780:break
            fast_guard()
            if (index-begin)%16==0:K.resources(q,parent,train,e,C)
            capture={'initial':[],'ancestral':[]};value,tokens,image=generate(index,args.arm)
            pixels=np.asarray(image.convert('RGB')).copy();assert pixels.shape==(512,512,3)
            assert len(capture['initial'])==1 and len(capture['ancestral'])==4
            if index<80:assert capture==old_rows[index]['noise']
            buf=io.BytesIO();image.save(buf,format='WEBP',lossless=True,quality=100);raw=buf.getvalue()
            path=directory/f'q{index:03d}.webp';assert not path.exists()
            used=sum(p.stat().st_blocks*512 for p in Path(q['project_root']).rglob('*') if p.is_file())
            assert used+len(raw)+65536<parent['maximum_output_bytes']
            path.write_bytes(raw)
            record={**queries[index],'query_index':index,'arm':args.arm,'image':K.ref(path),'prior_sample_seed':None if args.arm=='T_mean' else 1729+queries[index]['array_row'],'prior_bypassed':args.arm=='T_mean','new_prior_steps':0 if args.arm=='T_mean' else 50,'new_prior_CFG':1,'new_prior_clip_sample':False,
                    'pixels_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),'target_image_sha256':target_meta[index]['image_sha256'],
                    'condition_FP32_sha256':tensor_sha(value),'condition_FP16_sha256':tensor_sha(value.half()),
                    'tokens_FP16_sha256':tensor_sha(tokens),'noise':capture,'prepared':K.ref(out/'prepared.json')}
            with journal.open('a') as f:f.write(json.dumps(record)+'\n')
            rows.append(record);K.write(out/'generation_progress.json',{'arm':args.arm,'stage':args.stage,'completed_queries_for_arm':index+1})
    finally:EA.randn_tensor=old_noise;pipe.prepare_latents=old_prepare
    assert tensor_digest(pipe.unet,half=True)==frozen and tensor_digest(pipe.vae,half=True)==frozen_vae
    end=base_index+len(rows);assert begin<=end<=args.start+80
    fast_guard();after=K.resources(q,parent,train,e,C)
    result={'status':'COMPLETE_FIXED_BLOCK' if end==args.start+80 else 'PARTIAL_COMMITTED_PREFIX',
            'arm':args.arm,'stage':args.stage,'start':args.start,'part':args.part,'initial':begin,'completed':end,
            'journal':K.ref(journal),'all_host_weights_unchanged':True,'loader':loader,'seconds':time.monotonic()-started,
            'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'extra_CPU_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,
            'resources_before':before,'resources_after':after}
    K.write(out/(name+'_completion.json'),result);print(json.dumps(result),flush=True)
except BaseException as exc:
    K.write(out/(name+'_failure.json'),{'status':'FAILED_PRESERVED_STOP_CONTROLLER','phase':phase,'error':repr(exc),'traceback':traceback.format_exc()});raise
