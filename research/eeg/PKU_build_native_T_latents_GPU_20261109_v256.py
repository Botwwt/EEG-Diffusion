"""All13230 allowed T, native FP32 VAE mode, finite resumable asset builder."""
import csv,hashlib,io,json,os,shutil,socket,time,zipfile
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'data/native_T_latents_v256';out.mkdir(exist_ok=True);start=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(n,v):
    b=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();p=out/n;t=p.with_name(p.name+'.pending')
    with t.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
assert not (out/'completion.json').exists(),'Never overwrite a complete bank'
import numpy as np
import torch,torchvision
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from safetensors import safe_open
torch.set_num_threads(4);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torchvision.__version__=='0.18.0+cu121'
m=json.loads((R/'manifests/training_images_audit_job_v254.json').read_text());audit=R/('runs/training_images_audit_'+m['job_id']+'/completion.json');proof=json.loads(audit.read_bytes())
assert proof['status']=='PASS_PINNED_OSF_ZIP_AND_ALL16540_JPEG_MEMBER_IDENTITIES'
split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());ids=split['image_rows']['T'];assert ids==proof['allowed_T_latent_rows'] and len(ids)==13230
assert set(ids).isdisjoint(split['image_rows']['D']+split['image_rows']['V'])
with (R/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:rows={int(r['array_row']):r for r in csv.DictReader(f) if r['dataset_split']=='train'}
cfgpath=R/'assets/native_generator_v243/turbo/vae/config.json';assert sha(cfgpath)=='85a181424a5fd0cc292e44e467be66a5263976eee1baf36d208133f9d49d4269';cfg=json.loads(cfgpath.read_text())
checkpoint=R/'assets/native_generator_v243/turbo/sd_xl_turbo_1.0_fp16.safetensors'
with safe_open(str(checkpoint),framework='pt',device='cpu') as f:original={k:f.get_tensor(k) for k in f.keys() if k.startswith('first_stage_model.')}
assert original;converted=convert_ldm_vae_checkpoint(original,cfg);vae=AutoencoderKL.from_config(cfg);vae.load_state_dict(converted,strict=True)
assert sum(p.numel() for p in vae.parameters())==83653863
def state_sha(values):
    h=hashlib.sha256()
    for name,v in sorted(values.items()):h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
source_state=state_sha(converted);loaded_state=state_sha({k:v.to(torch.float16) for k,v in vae.state_dict().items()});assert source_state==loaded_state
del converted,original;vae=vae.cuda().float().eval().requires_grad_(False);vae.enable_slicing()
preprocess=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(512),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
registration=dict(array_rows=ids,shape=[13230,4,64,64],dtype='float32',batch=4,chunk=256,source_sha256=sha(__file__),image_audit_sha256=sha(audit),
    VAE_config_sha256=sha(cfgpath),original_FP16_VAE_state_sha256=source_state,VAE_parameters=83653863,VAE_scaling_factor=float(vae.config.scaling_factor),
    preprocessing='Native Resize512 BILINEAR, CenterCrop512, RGB ToTensor Normalize0.5',torch=torch.__version__,torchvision=torchvision.__version__,
    VAE_compute='FP32 widened from exact FP16 source, TF32 disabled, mode not posterior sample',old_latent_bank_bitwise_identity_claimed=False,
    permissions='Only T images receive VAE forward; no D/V/official-test forward, no parameter fit')
if (out/'registration.json').exists():assert json.loads((out/'registration.json').read_bytes())==registration
else:write('registration.json',registration)
state=json.loads((out/'progress.json').read_bytes()) if (out/'progress.json').exists() else dict(completed=0,chunks=[],cumulative_seconds=0.,attempts=[])
old_seconds=state['cumulative_seconds'];state['attempts'].append(dict(job_id=os.environ['SLURM_JOB_ID'],start_row=state['completed']))
path=out/'native_VAE_T_modes.npy'
if path.exists():
    array=np.lib.format.open_memmap(path,mode='r+');assert array.shape==(13230,4,64,64) and array.dtype==np.float32
    for c in state['chunks']:assert hashlib.sha256(array[c['start']:c['stop']].tobytes()).hexdigest()==c['sha256']
else:
    assert state['completed']==0 and shutil.disk_usage(out).free>13230*4*64*64*4+(1<<30)
    array=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(13230,4,64,64))
archive=R/'assets/things_images_v254/training_images.zip';torch.cuda.reset_peak_memory_stats()
with zipfile.ZipFile(archive) as z,torch.no_grad():
    for lo in range(state['completed'],13230,256):
        hi=min(lo+256,13230)
        for pos in range(lo,hi,4):
            selected=ids[pos:min(pos+4,hi)];pixels=[]
            for row_id in selected:
                r=rows[row_id];raw=z.read(r['image_member']);assert hashlib.sha256(raw).hexdigest()==r['image_sha256']
                with Image.open(io.BytesIO(raw)) as im:pixels.append(preprocess(im.convert('RGB')))
            value=vae.encode(torch.stack(pixels).cuda()).latent_dist.mode().float()*vae.config.scaling_factor
            assert torch.isfinite(value).all();array[pos:pos+len(selected)]=value.cpu().numpy()
        array.flush();state['chunks'].append(dict(start=lo,stop=hi,sha256=hashlib.sha256(array[lo:hi].tobytes()).hexdigest()))
        state.update(completed=hi,expected=13230,cumulative_seconds=old_seconds+time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),last_job_id=os.environ['SLURM_JOB_ID'])
        assert state['peak_GPU_bytes']<20<<30;write('progress.json',state)
        print(json.dumps({k:v for k,v in state.items() if k not in ['chunks','attempts']}),flush=True)
        if time.monotonic()-start>1000:break
array.flush();del array
with path.open('rb') as f:os.fsync(f.fileno())
if state['completed']==13230:
    assert len(state['chunks'])==52
    result=dict(status='COMPLETE_ALL13230_NATIVE_T_VAE_MODE_LATENTS',job_id=os.environ['SLURM_JOB_ID'],path=str(path),sha256=sha(path),bytes=path.stat().st_size,
        shape=[13230,4,64,64],dtype='float32',array_rows=ids,registration_sha256=sha(out/'registration.json'),chunks=state['chunks'],attempts=state['attempts'],
        cumulative_seconds=state['cumulative_seconds'],seconds_this_attempt=time.monotonic()-start,peak_GPU_bytes=state['peak_GPU_bytes'],old_bank_bitwise_identity_claimed=False,
        models_fitted=0,source_image_supervision_scope='T only',new_images_generated=0)
    write('completion.json',result);os.chmod(path,0o400);print(json.dumps({k:v for k,v in result.items() if k not in ['array_rows','chunks']}),flush=True)
else:print(json.dumps(dict(status='FINITE_TIME_STOP_RESUMABLE_CACHE_PREFIX',completed=state['completed'],expected=13230,job_id=os.environ['SLURM_JOB_ID'])),flush=True)
