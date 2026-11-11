"""Audit all3320 image identities/noise/pixels, restore830 exact targets, verify ranking algebra."""
import csv,hashlib,json,os,socket,time,traceback,zipfile
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_visitation_scores_v320';out.mkdir(exist_ok=True);start=time.monotonic()
def write(p,r):
    b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
try:
    import numpy as np
    import torch
    from PIL import Image
    from PKU_EEG_evaluation_core_20261109_v279 import sha,ranked,paper_source_scope
    torch.set_num_threads(4);torch.manual_seed(1729)
    assert not (out/'readiness.json').exists();image=R/'runs/EEG_visitation_images_v320';done=json.loads((image/'completion.json').read_text());assert done['status']=='COMPLETE_ALL830_QUERIES_4_VISITATION_ARMS_IMAGES_SCORES_PENDING' and done['images']==3320
    env=json.loads((R/'runs/evaluation_environment_v275/completion.json').read_text());assert env['status']=='PASS_OFFLINE_EVALUATION_ENV_ALL6_MODELS_AND_SSIM'
    bank=torch.load(image/'condition_bank.pt',map_location='cpu',weights_only=True);assert bank['condition'].shape==(4,830,1024)
    split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());ids=set(split['d_check_concepts'])
    with (R/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:identity={int(v['array_row']):v for v in csv.DictReader(f) if v['dataset_split']=='train' and v['concept_id'] in ids}
    assert len(identity)==830;targets=out/'targets';targets.mkdir();gallery=[]
    archive=R/'assets/things_images_v254/training_images.zip';assert sha(archive)=='dc9cc627141ba230641f729f84b80eb01f0a963d288c971eefcf76b4a7581f3a'
    with zipfile.ZipFile(archive) as z:
        for qi,row in enumerate(bank['array_rows']):
            r=identity[row];assert r['stimulus_id']==bank['stimulus_ids'][qi];b=z.read(r['image_member']);assert len(b)==int(r['image_bytes']) and hashlib.sha256(b).hexdigest()==r['image_sha256']
            p=targets/('%05d.jpg'%qi)
            with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
            with Image.open(p) as im:im.load();assert im.size==(500,500)
            gallery.append(dict(query_index=qi,array_row=row,stimulus_id=r['stimulus_id'],concept_id=r['concept_id'],path=str(p),image_sha256=r['image_sha256']))
    records=[];pixel_duplicates={};constant=0
    assert len(list((image/'records').glob('*.json')))==3320
    for i in range(3320):
        rec=json.loads((image/'records'/('%05d.json'%i)).read_text());ai,qi=divmod(i,830);assert rec['index']==i and rec['query_index']==qi and rec['arm']==bank['arms'][ai] and rec['array_row']==bank['array_rows'][qi] and rec['stimulus_id']==bank['stimulus_ids'][qi]
        assert rec['UNet_NFE']==4 and rec['IP_calls']==280 and rec['VAE_calls']==1 and not rec['normal_inference_target_access'] and not rec['selected_best_seed']
        assert rec['condition_sha256']==hashlib.sha256(bank['condition'][ai,qi].contiguous().numpy().tobytes()).hexdigest()
        p=Path(rec['image_path']);assert p.parent==image/'images' and p.stat().st_size==rec['bytes'] and sha(p)==rec['file_sha256']
        with Image.open(p) as im:
            assert im.size==(512,512);im=im.convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==rec['pixel_sha256'];a=np.asarray(im);constant+=int(not np.any(a!=a[0,0]))
        if ai:
            for key in ['image_seed','initial_noise_sha256','generator_after_initial_sha256','generator_final_sha256']:assert rec[key]==records[qi][key]
        pixel_duplicates.setdefault(rec['pixel_sha256'],[]).append(i);records.append(rec)
    records=[records[ai*830+qi] for qi in range(830) for ai in range(4)]
    scope=paper_source_scope();eye=torch.eye(10);images=eye.reshape(10,1,2,5);identity_fn=lambda x:x.flatten(1);concepts=['one_concept']*10
    expected=scope['two_way_identification'](images,images,identity_fn,lambda x:x,return_avg=False)
    measured=ranked(eye,eye,range(10),concepts,'paper');assert np.array_equal(measured['two_afc'].numpy(),np.asarray(expected[0])/9)
    wrong=ranked(torch.roll(eye,1,0),eye,range(10),concepts,'paper');expected_wrong,wrong_den=scope['two_way_identification'](torch.roll(images,1,0),images,identity_fn,lambda x:x,return_avg=False);assert np.allclose(wrong['two_afc'].numpy(),np.asarray(expected_wrong)/wrong_den,atol=1e-7,rtol=0)
    mixed=.25*eye+.75*torch.roll(eye,1,0);expected_mixed=scope['two_way_identification'](mixed.reshape(10,1,2,5),images,identity_fn,lambda x:x)
    actual_mixed=float(ranked(mixed,eye,range(10),concepts,'paper')['two_afc'].double().mean());assert abs(actual_mixed-expected_mixed)<1e-7
    asym=eye.clone();asym[9]=eye[0];expected_asym=scope['two_way_identification'](asym.reshape(10,1,2,5),images,identity_fn,lambda x:x,return_avg=False)
    actual_asym=ranked(asym,eye,range(10),concepts,'paper');assert np.allclose(actual_asym['two_afc'].numpy(),np.asarray(expected_asym[0])/9,atol=1e-7,rtol=0)
    write(out/'gallery.json',dict(queries=830,rows=gallery));write(out/'all_image_records.json',dict(arms=bank['arms'],records=records))
    write(out/'readiness.json',dict(status='PASS_ALL3320_IMAGES_830_TARGETS_AND_ORIGINAL_RANK_FIXTURES',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,images=3320,targets=830,
        gallery_sha256=sha(out/'gallery.json'),all_records_sha256=sha(out/'all_image_records.json'),image_completion_sha256=sha(image/'completion.json'),
        constant_spatial_images=constant,pixel_duplicate_groups={k:v for k,v in pixel_duplicates.items() if len(v)>1},all_failures_retained_no_silent_exclusion=True,
        rank_identity=float(measured['two_afc'].mean()),rank_deranged=float(wrong['two_afc'].mean()),rank_mixed=actual_mixed,asymmetric_fixed_gallery_verified=True,
        image_effects_read=False,source_sha256=sha(__file__)))
    print('PASS_ALL_IMAGE_TARGET_IDENTITIES_AND_RANK_ALGEBRA',flush=True)
except BaseException as e:
    write(out/('preparation_failure_'+os.environ['SLURM_JOB_ID']+'.json'),dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
