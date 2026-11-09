"""Independent whole-bank coverage/finiteness and two raw-image CPU VAE replays."""
import csv,hashlib,io,json,os,socket,time,traceback,zipfile
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/('runs/T_latent_independent_audit_'+os.environ['SLURM_JOB_ID']);out.mkdir();start=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(name,v):
    b=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();p=out/name;t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
try:
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    from safetensors import safe_open
    torch.set_num_threads(4);assert not torch.cuda.is_available()
    bank=R/'data/native_T_latents_v256';done=json.loads((bank/'completion.json').read_text());reg=json.loads((bank/'registration.json').read_text());state=json.loads((bank/'progress.json').read_text())
    assert done['status']=='COMPLETE_ALL13230_NATIVE_T_VAE_MODE_LATENTS'
    path=bank/'native_VAE_T_modes.npy';assert path.stat().st_size==done['bytes'] and sha(path)==done['sha256']
    assert sha(bank/'registration.json')==done['registration_sha256']
    split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());ids=split['image_rows']['T']
    assert done['array_rows']==reg['array_rows']==ids and len(set(ids))==13230
    assert set(ids).isdisjoint(split['image_rows']['D']+split['image_rows']['V'])
    assert reg['shape']==done['shape']==[13230,4,64,64] and done['dtype']==reg['dtype']=='float32'
    a=np.load(path,mmap_mode='r');assert a.shape==(13230,4,64,64) and a.dtype==np.float32 and a.flags.c_contiguous
    assert state['completed']==13230 and done['chunks']==state['chunks'] and len(done['chunks'])==52
    cursor=0;minimum=float('inf');maximum=-float('inf')
    for c in done['chunks']:
        assert c['start']==cursor and c['stop']==min(cursor+256,13230)
        x=np.asarray(a[c['start']:c['stop']]);assert np.isfinite(x).all() and hashlib.sha256(x.tobytes()).hexdigest()==c['sha256']
        minimum=min(minimum,float(x.min()));maximum=max(maximum,float(x.max()));cursor=c['stop']
    assert cursor==13230
    positions=[0,13229]  # Fixed endpoints before observing replay errors or images.
    with (R/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:rows={int(r['array_row']):r for r in csv.DictReader(f) if r['dataset_split']=='train'}
    cfgpath=R/'assets/native_generator_v243/turbo/vae/config.json';cfg=json.loads(cfgpath.read_text());assert sha(cfgpath)==reg['VAE_config_sha256']
    checkpoint=R/'assets/native_generator_v243/turbo/sd_xl_turbo_1.0_fp16.safetensors'
    with safe_open(str(checkpoint),framework='pt',device='cpu') as f:raw_state={k:f.get_tensor(k) for k in f.keys() if k.startswith('first_stage_model.')}
    converted=convert_ldm_vae_checkpoint(raw_state,cfg);vae=AutoencoderKL.from_config(cfg);vae.load_state_dict(converted,strict=True)
    assert sum(p.numel() for p in vae.parameters())==83653863
    del raw_state,converted;vae=vae.eval().requires_grad_(False)
    preprocess=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(512),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    replay=[]
    with zipfile.ZipFile(R/'assets/things_images_v254/training_images.zip') as z,torch.no_grad():
        for pos in positions:
            row=rows[ids[pos]];raw=z.read(row['image_member']);assert hashlib.sha256(raw).hexdigest()==row['image_sha256']
            with Image.open(io.BytesIO(raw)) as im:pixels=preprocess(im.convert('RGB')).unsqueeze(0)
            value=vae.encode(pixels).latent_dist.mode()*vae.config.scaling_factor;ref=torch.from_numpy(np.array(a[pos:pos+1]))
            delta=(value-ref).abs();assert torch.allclose(value,ref,atol=5e-4,rtol=5e-4),(pos,float(delta.max()),float(delta.mean()))
            replay.append(dict(T_position=pos,array_row=ids[pos],image_sha256=row['image_sha256'],max_abs_CPU_GPU=float(delta.max()),mean_abs_CPU_GPU=float(delta.mean()),atol=5e-4,rtol=5e-4))
    result=dict(status='PASS_ALL13230_NATIVE_LATENTS_SHA_COVERAGE_AND_TWO_CPU_REPLAYS',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),seconds=time.monotonic()-start,
        bank_sha256=done['sha256'],bank_bytes=done['bytes'],shape=list(a.shape),dtype=str(a.dtype),all_rows=13230,all_chunks=52,finite_min=minimum,finite_max=maximum,
        registration_sha256=done['registration_sha256'],source_completion_sha256=sha(bank/'completion.json'),replay=replay,new_optimizer_updates=0,new_images_generated=0,
        image_scope='Only T, no D/V/official test image forward',old_latent_bank_bitwise_identity_claimed=False,science_or_SOTA_claim=False)
    write('completion.json',result);print(json.dumps(result),flush=True)
except BaseException as e:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
