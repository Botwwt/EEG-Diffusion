"""Fixed native R80 reference and EEG interventions, then locked full scoring.

No checkpoint or sampling selection, no intermediate-score-driven changes.
Protected-T/R4 component development remains a separate later comparison.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')
PYTHON='/root/miniconda3/bin/python'
STATE=WORK/'audit/native_benchmark_queue_v2.json'


def write(value):
    temp=STATE.with_name(STATE.name+'.pending')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(STATE)


def gpu(name,command,seconds=10800):
    argv=[PYTHON,str(ROOT/'run_gpu_job_v2.py'),'--name',name,'--max-seconds',str(seconds),'--']+command
    write(dict(status='RUNNING_FIXED_NATIVE_BENCHMARK',job=name,argv=argv))
    receipt=ROOT/'audit'/(name+'_job.json')
    waited=time.monotonic()
    while receipt.exists():
        record=json.loads(receipt.read_text())
        assert record['command']==command,'Existing job command differs from registered arm'
        if record['status']=='COMPLETED':
            assert record['exit_code']==0
            return
        assert record['status']=='RUNNING','Preserve failed job for explicit diagnosis'
        assert time.monotonic()-waited<seconds+120,'Finite adoption wait expired'
        time.sleep(15)
    with (WORK/'logs'/(name+'.log')).open('x') as log:
        code=subprocess.call(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert code==0,'Preserve failed job and resume explicitly: '+name


def cpu(command,seconds=600):
    subprocess.run([PYTHON]+command,cwd=ROOT,timeout=seconds,check=True)


def main():
    assert not STATE.exists(),'Preserve existing queue'
    ready=json.loads((WORK/'audit/oracle_readiness_queue_v1.json').read_text())
    assert ready['status']=='ORACLE32_ACTUAL_IMAGES_AND_PAIRING_COMPLETE_AWAITING_INSPECTION'
    inspection=json.loads((WORK/'audit/oracle_Tdiag32_inspection_v1.json').read_text())
    assert inspection['all32_reviewed'] and inspection['native_condition_path_operational']
    specification=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        subjects=10,queries_per_arm=2000,repeat_protocol='R80',train_seed=1729,images_per_query=1,
        arms=['correct','permuted','missing'],fixed_final_parent='published_source_dropout_v2',
        source_entry='generate_BrainHIVE_native_images_v2.py',
        source_sha256=hashlib.sha256((ROOT/'generate_BrainHIVE_native_images_v2.py').read_bytes()).hexdigest(),
        no_score_peeking_to_tune=True,no_component_in_this_queue=True,
        generator='pinned SDXL Turbo 512/4/CFG0/IP1',protected_R4_component_track_separate=True)
    specification['scheduler_revision']='v2: separate semantic and low-level output directories; adopt already-running identical image arm'
    registration=WORK/'audit/native_benchmark_registration_v2.json'
    assert not registration.exists();registration.write_text(json.dumps(specification,indent=2)+'\n')
    for intervention in specification['arms']:
        run='native_R80_'+intervention+'_v1'
        gpu('BrainHIVE_'+run,[PYTHON,str(ROOT/'generate_BrainHIVE_native_images_v2.py'),
            '--role','published','--split','benchmark','--intervention',intervention,'--run-name',run])
        if not (WORK/'images'/run/'pairing_audit.json').exists():
            cpu([str(ROOT/'audit_BrainHIVE_image_arm_CPU_v1.py'),'--run-name',run],seconds=300)
    # Verify matched query noise and target identities across all intervention arms.
    rows={arm:[json.loads(x) for x in (WORK/'images'/('native_R80_'+arm+'_v1')/'manifest.jsonl').read_text().splitlines()]
          for arm in specification['arms']}
    base={r['query_id']:r for r in rows['correct']}
    identity={}
    for arm in ['permuted','missing']:
        assert {r['query_id'] for r in rows[arm]}==set(base)
        same=0
        for row in rows[arm]:
            ref=base[row['query_id']]
            for key in ['subject','stimulus_id','array_row','image_seed','target_image_sha256','UNet_NFE','IP_conditioned_attention_calls']:
                assert row[key]==ref[key],(arm,key)
            assert row['generated_path']!=ref['generated_path']
            same+=row['generated_sha256']==ref['generated_sha256']
        identity[arm]=dict(queries=2000,identical_pixels_container_hash_count=same,matched_query_seeds_and_targets=True)
    (WORK/'audit/native_R80_intervention_identity_v1.json').write_text(json.dumps(identity,indent=2)+'\n')
    # All image arms are complete before full metrics are first computed.
    gallery=WORK/'gallery/benchmark/manifest.jsonl'
    gallery_features=WORK/'evaluation/benchmark_gallery_features_v1'
    gpu('BrainHIVE_R80_gallery_features_v1',[PYTHON,str(ROOT/'extract_image_features.py'),
        '--manifest',str(gallery),'--output',str(gallery_features),'--real-gallery'])
    for arm in specification['arms']:
        run='native_R80_'+arm+'_v1';manifest=WORK/'images'/run/'manifest.jsonl'
        feature=WORK/'evaluation'/run/'features';score=WORK/'evaluation'/run/'scores'
        gpu('BrainHIVE_'+run+'_features_v1',[PYTHON,str(ROOT/'extract_image_features.py'),
            '--manifest',str(manifest),'--output',str(feature)])
        cpu([str(ROOT/'score_full_manifest.py'),'--manifest',str(manifest),'--features',str(feature),
             '--gallery-manifest',str(gallery),'--gallery-features',str(gallery_features),'--output',str(score)],seconds=1800)
        cpu([str(ROOT/'score_low_level.py'),'--manifest',str(manifest),'--gallery-manifest',str(gallery),
             '--output',str(WORK/'evaluation'/run/'low_level')],seconds=1800)
    write(dict(status='ALL3_NATIVE_R80_IMAGE_ARMS_AND_LOCKED_PER_QUERY_METRICS_COMPLETE',
        queries_per_arm=2000,total_saved_images=6000,component_gain_established=False,
        independent_final_H=False,protocol='Exploratory native R80 reproduction; separate from protected R4 component track'))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(dict(status='FAILED_PRESERVED_NEEDS_EXPLICIT_DIAGNOSIS',error_type=type(error).__name__,error=str(error)[:1200]))
        raise
