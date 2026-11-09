"""Finish the already evaluated historical reference gallery using CPU only."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
PY='/root/miniconda3/bin/python'


def load(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    os.nice(10);started=time.monotonic()
    images=DATA/'images/S1_full_native_batchsafe_v2/V/REF_REG_correct_R4'
    evaluation=DATA/'evaluation/V/REF_REG_correct_R4'
    output=DATA/'galleries/V_existing_REG_patience20_complete_v1'
    state_path=ROOT/'audit/existing_REG_gallery_CPU_state_v1.json'
    assert not state_path.exists() and not output.exists(),'Preserve previous attempts'
    assert load(images/'completion.json')['queries']==16600
    assert sha(images/'manifest.jsonl')==load(images/'completion.json')['image_manifest_sha256']
    assert load(evaluation/'metrics/summary.json')['queries']==16600
    identity=load(evaluation/'identity_with_actual_scores_v1/summary.json')
    assert identity['verified_queries']==identity['score_rows_checked']==16600 and identity['failed_queries']==0
    command=[PY,str(ROOT/'build_offline_gallery.py'),'--arm','Historical_REF_REG_patience20',
        str(images/'manifest.jsonl'),str(evaluation/'metrics/per_query.csv'),'--output',str(output)]
    log=ROOT/'logs/existing_REG_complete_gallery_CPU_v1.log';assert not log.exists()
    state=dict(status='RUNNING_EXISTING_IMAGES_CPU_ONLY',controller_pid=os.getpid(),GPU_used=False,
        output=str(output),not_current_patience50_idea_images=True,command=command,
        builder_sha256=sha(ROOT/'build_offline_gallery.py'))
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    code=None
    try:
        env=os.environ|dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with log.open('xb') as file:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=file,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;state_path.write_text(json.dumps(state,indent=2)+'\n')
            try:code=child.wait(timeout=900)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                code=-9
        assert code==0,('Inspect preserved gallery failure',str(log),code)
        receipt=load(output/'completion.json')
        assert receipt['status']=='COMPLETED_FULL_OFFLINE_GALLERY' and receipt['queries']==receipt['verified_generated_images']==16600
        assert receipt['subjects']==10 and receipt['thumbnail_pages']==415
        def gallery_data(folder):
            text=(folder/'data.js').read_text();assert text.startswith('window.GALLERY=') and text.endswith(';\n')
            return json.loads(text[len('window.GALLERY='):-2])
        regular=gallery_data(output);blinded=gallery_data(output/'blinded')
        assert len(regular['rows'])==len(blinded['rows'])==16600 and blinded['blinded'] and not regular['blinded']
        assert [x['id'] for x in regular['rows']]==sorted(x['id'] for x in regular['rows'])
        assert [x['id'] for x in regular['rows']]==[x['id'] for x in blinded['rows']]
        blind_text=(output/'blinded/data.js').read_text()
        for forbidden in ['Historical_REF_REG','generated_path','DINO2AFC','projection_checkpoint','"dino"','"object"']:
            assert forbidden not in blind_text,forbidden
        assert all(set(x['cells'][0])=={'label','full','thumb'} and x['cells'][0]['label']=='A' for x in blinded['rows'])
        assert len(list(csv.DictReader((output/'blinded/human_ratings_blank.csv').open())))==0
        with (output/'unblinding_and_pairing.csv').open() as file:pairing=list(csv.DictReader(file))
        assert len(pairing)==16600 and {x['query_id'] for x in pairing}=={x['id'] for x in regular['rows']}
        unique_assets={}
        for row in regular['rows']:
            unique_assets[row['targetFull']]=None
            for cell in row['cells']:unique_assets[cell['full']]=None
        for relative in unique_assets:assert (output/relative).is_file()
        for part in ['full','thumbs']:assert len(list((output/'blinded'/part).glob('*.jpg')))==receipt['assets']
        assert len(list((output/'pages').glob('page_*.html')))==415
        proof=dict(status='COMPLETE16600_OFFLINE_AND_BLINDED_MATERIALS_MECHANICALLY_VERIFIED',queries=16600,
            subjects=10,pages=415,assets=receipt['assets'],pairing_rows=len(pairing),human_ratings_performed=False,
            source_gallery_completion_sha256=sha(output/'completion.json'),blinded_data_sha256=sha(output/'blinded/data.js'),
            historical_patience20_reference_only=True,new_idea_image_gain_established=False,
            browser_interaction_visual_QA_performed=False,limits='Complete counts, files, IDs, labels and pairing verified; no human quality ratings or new-encoder result.')
        (output/'independent_materials_check_v1.json').write_text(json.dumps(proof,indent=2)+'\n')
        state.update(status='COMPLETED_EXISTING_REFERENCE_GALLERY_CPU_ONLY',gallery_receipt=receipt,materials_check=proof)
    except Exception as error:
        state.update(status='FAILED_PRESERVE_OUTPUTS_NO_AUTOMATIC_RETRY',error=repr(error));raise
    finally:
        state.update(seconds=time.monotonic()-started,exit_code=code,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
            file.write(json.dumps(dict(stage='existing_reference_full_offline_gallery',seconds=state['seconds'],exit_code=code,
                GPU_used=False,actual_status=state['status'],receipt=str(state_path)))+'\n')
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
