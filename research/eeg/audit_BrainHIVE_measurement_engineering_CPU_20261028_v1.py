"""Decode actual parity images and validate all shared temporary-update records.

No independent UNet/gradient replay is claimed by this CPU engineering audit.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,hashlib,json,math,time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,'/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages')
from PIL import Image

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(completion,output):
    start=time.monotonic();c=read(completion);assert c['status']=='D39_REAL_NATIVE_EEG_CORE_GRADIENT_AND_PARITY_ENGINEERING_PASSED'
    assert c['temporary_updates']==8 and c['temporary_states_discarded'] and c['research_images']==0
    assert all(math.isfinite(x) and x>0 for x in c['epsilon_only_gradients_into_all4_original_core_tensors'].values())
    pictures=[]
    for row in c['parity_images']:
        p=Path(row['generated_path']);assert sha(p)==row['generated_sha256']
        with Image.open(p) as im:pixels=np.asarray(im.convert('RGB')).copy()
        assert pixels.shape==(512,512,3) and hashlib.sha256(pixels.tobytes()).hexdigest()==row['generated_pixel_sha256']
        assert row['UNet_NFE']==4 and row['native_IP_K_calls']==row['native_IP_V_calls']==280
        assert len(row['ancestral_noise_observations'])==4 and row['scheduler_step_signature_unmodified']
        pictures.append(pixels)
    assert len(pictures)==2 and np.array_equal(*pictures)
    a,b=c['parity_images'];assert a['initial_noise_latents_sha256']==b['initial_noise_latents_sha256'] and a['ancestral_noise_observations']==b['ancestral_noise_observations']
    records=[json.loads(x) for x in (completion.parent/'temporary_updates.jsonl').read_text().splitlines()];assert len(records)==8
    arms=['unanchored','isotropic','measurement','permuted_measurement'];first=[]
    for arm in arms:
        rows=[x for x in records if x['arm']==arm];assert [x['step'] for x in rows]==[1,2]
        assert rows[0]['anchor_loss']==0
        assert all(math.isfinite(x[key]) for x in rows for key in ['total_loss','epsilon_loss','token_loss','anchor_loss'])
        first.append(rows[0]['encoder_state_sha256'])
        for r in rows:
            for other in [x for x in records if x['step']==r['step']]:
                assert r['rows']==other['rows'] and r['t']==other['t'] and r['noise_sha256']==other['noise_sha256']
            assert all(v>0 and math.isfinite(v) for v in r['all4_core_gradient_norms'].values())
            assert math.isclose(r['total_loss'],r['epsilon_loss']+.1*r['token_loss']+r['anchor_loss'],rel_tol=1e-6,abs_tol=1e-7)
            if arm=='unanchored':assert r['anchor_loss']==0
            if arm=='isotropic':assert math.isclose(r['anchor_loss'],.5*r['current_unit_shift_RMS']**2/.25**2,rel_tol=2e-6,abs_tol=1e-10)
    assert len(set(first))==1 and c['frozen_ordinary_strength']==1
    v=dict(status='D39_PARITY_PIXELS_NOISE_AND_ALL8_TEMPORARY_UPDATE_RECORDS_INDEPENDENT_CPU_PASSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      source_sha256=sha(__file__),completion_sha256=sha(completion),parity_images=2,parity_pixel_max_difference=0,ancestral_noise_records=8,initial_noise_records=2,
      all8_update_records_checked=True,first_update_identical_across_all4=True,ordinary_and_host_freeze_scope='Recorded GPU assertions and source hash, not an independent CPU replay of UNet gradients',
      no_independent_full_gradient_or_measurement_quadratic_recalculation=True,new_GPU_jobs=0,new_training_updates=0,seconds=time.monotonic()-start)
    with output.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
    print(json.dumps(v))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--completion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.completion,a.output)
