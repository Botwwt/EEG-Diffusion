"""Verify OSF container and all16540 JPEG identities, without model fitting."""
import csv,hashlib,io,json,os,socket,time,zipfile
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/('runs/training_images_audit_'+os.environ['SLURM_JOB_ID']);out.mkdir();start=time.monotonic()
path=R/'assets/things_images_v254/training_images.zip';h=hashlib.sha256()
with path.open('rb') as f:
    for b in iter(lambda:f.read(4<<20),b''):h.update(b)
assert path.stat().st_size==655039265 and h.hexdigest()=='dc9cc627141ba230641f729f84b80eb01f0a963d288c971eefcf76b4a7581f3a'
with (R/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['dataset_split']=='train']
assert len(rows)==16540 and len({r['image_member'] for r in rows})==16540
split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());T=split['image_rows']['T'];assert len(T)==13230
from PIL import Image
dimensions={};total=0
with zipfile.ZipFile(path) as z,(out/'all16540_JPEG_SHA.jsonl').open('x') as log:
    names=[r.filename for r in z.infolist() if not r.is_dir()];assert len(names)==len(set(names))==16540 and set(names)=={r['image_member'] for r in rows}
    for r in rows:
        b=z.read(r['image_member']);digest=hashlib.sha256(b).hexdigest();assert len(b)==int(r['image_bytes']) and digest==r['image_sha256']
        with Image.open(io.BytesIO(b)) as im:
            assert im.format=='JPEG' and min(im.size)>0;dims=str(im.size);dimensions[dims]=dimensions.get(dims,0)+1
        total+=len(b);log.write(json.dumps(dict(array_row=int(r['array_row']),stimulus_id=r['stimulus_id'],sha256=digest,bytes=len(b)))+'\n')
    log.flush();os.fsync(log.fileno())
result=dict(status='PASS_PINNED_OSF_ZIP_AND_ALL16540_JPEG_MEMBER_IDENTITIES',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),seconds=time.monotonic()-start,
    archive=str(path),archive_bytes=path.stat().st_size,archive_sha256=h.hexdigest(),JPEG_count=len(rows),JPEG_bytes=total,dimensions=dimensions,
    allowed_T_latent_rows=T,allowed_T_count=13230,models_fitted=0,latent_images_encoded=0,official_test_images_read=0,D_V_used_only_for_byte_identity_not_training=True)
b=(json.dumps(result,indent=2)+'\n').encode();tmp=out/'completion.pending'
with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
assert tmp.read_bytes()==b;tmp.replace(out/'completion.json');print(json.dumps({k:v for k,v in result.items() if k!='allowed_T_latent_rows'}),flush=True)
