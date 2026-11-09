"""Independent saved-image differences; no GPU, targets, scores, or model edits."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')
ALLOCATION=ROOT/'audit/BrainHIVE_image_effects_CPU_allocation_v26.json'
RESERVE=WORK/'audit/BrainHIVE_image_effects_CPU_v26.reserve'
STATE=WORK/'audit/bounded_image_effects_watch_v26.json'
OUTPUT=WORK/'audit/bounded_image_effects_v26'
ARMS=['parent','ordinary','risk']
PAIRS=[('ordinary','parent'),('risk','parent'),('risk','ordinary')]


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.pending')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def process_state(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split()[2]
    except FileNotFoundError:return None


def image(path,expected):
    raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,path
    with Image.open(io.BytesIO(raw)) as im:
        assert im.size==(512,512) and im.mode=='RGB',(path,im.size,im.mode)
        return np.asarray(im).copy()


def pixels(a,b):
    delta=a.astype(np.float32)-b.astype(np.float32)
    return dict(equal_RGB_pixels=bool(np.array_equal(a,b)),
        mean_abs_RGB_01=float(np.abs(delta).mean(dtype=np.float64)/255),
        RMS_RGB_01=float(np.sqrt(np.square(delta).mean(dtype=np.float64))/255),
        max_abs_RGB_8bit=int(np.abs(delta).max()))


def preflight():
    source=WORK/'engineering/bounded_component_profile_v2_batch16/completion.json'
    data=read(source);assert data['status']=='ENGINEERING_IMAGE_GRADIENT_AND_FIXED_T_IMAGE_PATH_COMPLETE'
    groups={arm:{} for arm in ARMS}
    for row in data['generated_images']:
        assert row['engineering_T_training_image']
        key=(row['array_row'],row['stimulus_id'],row['image_seed'])
        assert key not in groups[row['arm']];groups[row['arm']][key]=row
    assert all(set(groups[a])==set(groups['parent']) and len(groups[a])==4 for a in ARMS)
    result=[]
    for key in sorted(groups['parent']):
        rows={a:groups[a][key] for a in ARMS}
        assert len({Path(r['path']).resolve() for r in rows.values()})==3
        decoded={a:image(r['path'],r['sha256']) for a,r in rows.items()}
        result.append(dict(array_row=key[0],stimulus_id=key[1],image_seed=key[2],pairs={
            a+'_vs_'+b:pixels(decoded[a],decoded[b]) for a,b in PAIRS}))
    output=ROOT/'audit/BrainHIVE_image_effects_T_preflight_v26.json';assert not output.exists()
    record=dict(UTC=utc(),status='ACTUAL12_EXISTING_T_IMAGES_AND12_PAIR_OPERATIONS_VERIFIED',
        engineering_receipt_sha256=sha(source),source_sha256=sha(__file__),queries=4,
        actual_saved_images_read=12,pair_operations=12,rows=result,
        formal_or_V_or_test_results=False,formal_query_scope_reduced=False,
        new_images_generated=0,new_GPU_jobs=0,component_gain_claimed=False)
    write(output,record)
    print(json.dumps(dict(status=record['status'],queries=4,images_read=12,
        pair_mean_absolute_RGB_differences={a+'_vs_'+b:statistics.mean(r['pairs'][a+'_vs_'+b]['mean_abs_RGB_01'] for r in result) for a,b in PAIRS})),flush=True)


def ready(recipe):
    return all(all((WORK/'images'/recipe['run_names'][a]/name).is_file()
        for name in ['completion.json','manifest.jsonl','pairing_audit.json','model_receipt.json']) for a in ARMS)


def release_owned_reservation(spec):
    if RESERVE.exists():
        assert not RESERVE.is_symlink() and RESERVE.resolve().parent==(WORK/'audit').resolve()
        assert RESERVE.stat().st_ino==spec['reservation_inode']
        RESERVE.unlink()


def stats(values):
    return dict(count=len(values),minimum=min(values),maximum=max(values),mean=statistics.mean(values),
        population_std=statistics.pstdev(values),exactly_constant=all(v==values[0] for v in values)) if values else None


def formal(recipe_path,spec):
    recipe=read(recipe_path);assert sha(recipe_path)==spec['recipe_sha256'] and ready(recipe)
    assert not OUTPUT.exists();OUTPUT.mkdir()
    rows={};receipts={};proofs={}
    for arm in ARMS:
        folder=WORK/'images'/recipe['run_names'][arm]
        done=read(folder/'completion.json');pairing=read(folder/'pairing_audit.json');receipt=read(folder/'model_receipt.json')
        assert done['status']=='ALL_ACTUAL_LOSSLESS_IMAGES_SAVED_INFERENCE_ONLY' and done['queries']==16600
        assert done['role']=='full' and done['split']=='V' and done['intervention']=='correct'
        assert sha(folder/'inference_manifest.jsonl')==done['inference_manifest_sha256']
        assert pairing['status']=='ALL_SAVED_IMAGES_TARGETS_AND_STIMULUS_ROWS_JOIN_VERIFIED' and pairing['queries']==16600
        assert sha(folder/'manifest.jsonl')==pairing['manifest_sha256']
        assert receipt['recipe_sha256']==sha(recipe_path) and receipt['arm']==arm
        assert (receipt['resolution'],receipt['steps'],receipt['CFG'],receipt['images_per_query'])==(512,4,0,1)
        if arm!='parent':
            completion=read(WORK/'components/bounded_v1'/arm/'completion.json')
            assert sha(completion['checkpoint_path'])==completion['checkpoint_sha256']==receipt['component_checkpoint_sha256']
            calibration=WORK/'components/bounded_v1'/arm/'calibration.json'
            assert sha(calibration)==completion['calibration_sha256'] and read(calibration)==receipt['calibration']
        else:assert receipt['component_checkpoint_sha256'] is None
        items=[json.loads(x) for x in (folder/'manifest.jsonl').read_text().splitlines()]
        rows[arm]={r['query_id']:r for r in items};assert len(items)==len(rows[arm])==16600
        assert len({r['subject'] for r in items})==10
        receipts[arm]=receipt
        proofs[arm]={n:sha(folder/n) for n in ['model_receipt.json','manifest.jsonl','pairing_audit.json']}
    assert all(set(rows[a])==set(rows['parent']) for a in ARMS)
    assert all(receipts[a]['generator']==receipts['parent']['generator'] for a in ARMS)
    assert RESERVE.stat().st_ino==spec['reservation_inode'] and RESERVE.stat().st_size==spec['reservation_bytes']
    aggregates={a+'_vs_'+b:dict(equal_files=0,equal_pixels=0,equal_producer_logged_A_inputs=0,MAE=[],RMS=[],maxima=[]) for a,b in PAIRS}
    arm_values={a:dict(raw_correction_RMS=[],gate=[],predicted_risk=[]) for a in ARMS}
    started=time.monotonic();used=0;completed=0;fieldnames=None
    # Write into the already allocated file. Its unused extent remains reserved
    # until the complete CSV is closed, so a concurrent evaluator cannot use it.
    fp=RESERVE.open('r+b',buffering=0)
    try:
        for q in sorted(rows['parent']):
            assert time.monotonic()-started<spec['maximum_CPU_work_seconds']
            current={a:rows[a][q] for a in ARMS};base=current['parent']
            assert len({Path(r['generated_path']).resolve() for r in current.values()})==3
            for arm,row in current.items():
                for key in ['subject','stimulus_id','array_row','global_query_index','image_seed','condition_sha256','EEG_source_stimulus_id','target_image_sha256','score_target_path']:
                    assert row[key]==base[key],(q,arm,key)
                assert row['EEG_source_stimulus_id']==row['stimulus_id'] and row['repeat_protocol']=='R4'
                assert not row['target_or_visual_features_read_for_condition'] and row['target_mapping_attached_after_inference']
                assert row['UNet_NFE']==4 and row['IP_conditioned_attention_calls']==280 and row['VAE_decoder_calls']==1
                assert row['lossless_pixel_roundtrip'] and len(row['actual_A_input_fp16_sha256'])==64
                value=float(row['component_correction_RMS']);assert math.isfinite(value) and value>=0
                arm_values[arm]['raw_correction_RMS'].append(value)
                for key,label,maximum in [('component_gate','gate',1),('predicted_risk','predicted_risk',2)]:
                    if row[key] is not None:
                        value=float(row[key]);assert math.isfinite(value) and 0<=value<=maximum
                        arm_values[arm][label].append(value)
            assert base['component_correction_RMS']==0
            decoded={a:image(r['generated_path'],r['generated_sha256']) for a,r in current.items()}
            record=dict(query_id=q,subject=base['subject'],stimulus_id=base['stimulus_id'],image_seed=base['image_seed'])
            for a,b in PAIRS:
                name=a+'_vs_'+b;effect=pixels(decoded[a],decoded[b]);acc=aggregates[name]
                equal_file=current[a]['generated_sha256']==current[b]['generated_sha256']
                equal_input=current[a]['actual_A_input_fp16_sha256']==current[b]['actual_A_input_fp16_sha256']
                acc['equal_files']+=equal_file;acc['equal_pixels']+=effect['equal_RGB_pixels'];acc['equal_producer_logged_A_inputs']+=equal_input
                acc['MAE'].append(effect['mean_abs_RGB_01']);acc['RMS'].append(effect['RMS_RGB_01']);acc['maxima'].append(effect['max_abs_RGB_8bit'])
                record.update({name+'_'+k:v for k,v in dict(equal_file_SHA=equal_file,equal_logged_A_input_SHA=equal_input,**effect).items()})
            for arm in ARMS:
                for key in ['component_correction_RMS','component_gate','predicted_risk']:record[arm+'_'+key]=current[arm][key]
            buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(record),lineterminator='\n')
            if fieldnames is None:writer.writeheader();fieldnames=list(record)
            else:assert fieldnames==list(record)
            writer.writerow(record);encoded=buffer.getvalue().encode('utf-8')
            assert used+len(encoded)<=spec['reservation_bytes']-65536,'Preserve complete prefix within allocated output budget'
            fp.write(encoded);used+=len(encoded);completed+=1
            if completed%1000==0:
                fp.flush();print(json.dumps(dict(stage='CPU_FULL_SAVED_IMAGE_EFFECTS',completed=completed,expected=16600)),flush=True)
        assert completed==16600
        fp.truncate(used);fp.flush();os.fsync(fp.fileno());fp.close();RESERVE.rename(OUTPUT/'per_query.csv')
    except Exception as error:
        if not fp.closed:
            fp.truncate(used);fp.flush();fp.close();RESERVE.rename(OUTPUT/'per_query_partial.csv')
        write(OUTPUT/'failure.json',dict(UTC=utc(),status='PARTIAL_EFFECT_AUDIT_PRESERVED',completed=completed,error=repr(error)))
        raise
    summary={name:dict(queries=16600,identical_file_SHA_count=v['equal_files'],identical_RGB_count=v['equal_pixels'],
        identical_producer_logged_A_input_count=v['equal_producer_logged_A_inputs'],
        identical_RGB_fraction=v['equal_pixels']/16600,mean_abs_RGB_01=stats(v['MAE']),RMS_RGB_01=stats(v['RMS']),
        max_abs_RGB_8bit=stats(v['maxima'])) for name,v in aggregates.items()}
    result=dict(UTC=utc(),status='ALL49800_SAVED_IMAGES_AND_ALL16600_THREE_ARM_PIXEL_PAIRS_CHECKED_CPU',
        queries=16600,images_freshly_verified_and_decoded=49800,pair_operations=49800,
        pairs=summary,arm_logged_condition_statistics={a:{k:stats(v) for k,v in values.items()} for a,values in arm_values.items()},
        producer_receipts=receipts,proof_sha256=proofs,source_sha256=sha(__file__),recipe_sha256=sha(recipe_path),
        per_query_csv_sha256=sha(OUTPUT/'per_query.csv'),per_query_csv_bytes=used,seconds=time.monotonic()-started,
        risk_calibration_slope_is_zero=receipts['risk']['calibration']['slope']==0,
        normal_EEG_and_noise_identity_verified_by_metadata=True,target_images_or_features_or_scores_read=False,
        native_A_input_digests_are_producer_logged_not_independently_recomputed=True,
        original_images_modified=False,images_generated=0,GPU_used=False,
        fixed_or_shuffled_gate_counterfactuals_performed=False,
        pixel_difference_is_not_image_gain_or_quality_or_EEG_mechanism_evidence=True,
        component_gain_established=False,scientific_goal_complete=False)
    write(OUTPUT/'summary.json',result)
    return result


def watch(recipe_path):
    spec=read(ALLOCATION);assert spec['source_sha256']==sha(__file__) and Path(spec['reservation_path'])==RESERVE
    assert not STATE.exists();recipe=read(recipe_path);start=time.monotonic()
    while time.monotonic()-start<spec['maximum_wait_seconds']:
        if ready(recipe):
            write(STATE,dict(UTC=utc(),status='RUNNING_CPU_SAVED_IMAGE_EFFECTS',GPU_used=False))
            result=formal(recipe_path,spec);write(STATE,dict(UTC=utc(),status=result['status'],summary=str(OUTPUT/'summary.json')));return
        state=process_state(spec['main_controller_pid'])
        main=read(WORK/'audit/bounded_full_lifecycle_queue_v1.json')
        if state in [None,'Z','X'] or main['status'] in ['FINITE_MAIN_COMPARISON_PARTIAL_WITH_EXPLICIT_FAILURES','FAILED_PRESERVED_NEEDS_EXPLICIT_DIAGNOSIS']:
            release_owned_reservation(spec)
            write(STATE,dict(UTC=utc(),status='NOT_RUN_REQUIRED_FULL_IMAGE_ARMS_MISSING',main_process_state=state,main_status=main['status'],negative_method_result_inferred=False));return
        write(STATE,dict(UTC=utc(),status='WAITING_FOR_ALL_THREE_ACTUAL_FULL_PAIRED_IMAGE_ARMS',main_controller_pid=spec['main_controller_pid'],main_process_state=state,new_GPU_jobs=0))
        time.sleep(30)
    release_owned_reservation(spec)
    write(STATE,dict(UTC=utc(),status='FINITE_CPU_WATCH_TIMEOUT_NO_COMPUTE_RESTART'))


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--engineering-preflight',action='store_true');mode.add_argument('--watch',action='store_true')
    parser.add_argument('--recipe');args=parser.parse_args()
    if args.engineering_preflight:preflight()
    else:
        assert args.recipe
        try:watch(Path(args.recipe))
        except Exception as error:
            if ALLOCATION.exists():release_owned_reservation(read(ALLOCATION))
            write(STATE,dict(UTC=utc(),status='FAILED_CPU_IMAGE_EFFECTS_AUDIT_PRESERVED',error=repr(error),new_GPU_jobs=0));raise


if __name__=='__main__':main()
