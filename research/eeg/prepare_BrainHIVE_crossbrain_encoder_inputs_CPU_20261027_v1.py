"""Cache exact full-D17 EEG inputs once; avoid rehashing ten huge raw files per fit."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
R=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();torch.set_num_threads(2)
    assert not a.output.exists() and not torch.cuda.is_initialized();a.output.mkdir(parents=True)
    Q=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/rapid_iteration_20261027_v1/candidates')
    old=U.read(Q/'D15_cross_encoder/inputs_v2/completion.json');neural=U.read(R/'audit/raw_neural_recoverability_20261027_v1/actual_input_completion.json')
    contract=U.read('/root/brainhive_reproduction_20261020_v1/data/component_input_contract_v1.json');d15=U.read(Q/'D15_cross_encoder/probe_native_v2/completion.json')
    queries=U.read(U.checked(d15['queries_and_controls']))['queries'];d09=U.read(Q/'D09_gain_calibration/stage400_recovered_v3/completion.json');di=U.read(U.checked(d09['inputs']));panel=U.read(U.checked(di['panel']))
    source_rows=[];dvalues=[];check=None
    for si,subject in enumerate(contract['subjects']):
        assert time.monotonic()-start<100;name=subject['subject'];source=old['subjects'][name]['input'];raw=torch.load(U.checked(source),weights_only=True,map_location='cpu',mmap=True)
        assert raw.shape==(16540,63,250)
        T=neural['subjects'][name];array=np.load(U.checked(dict(path=T['path'],sha256=T['sha256'])),allow_pickle=False,mmap_mode='r')
        assert array.shape==(13230,17,250)
        for lo in range(0,13230,1024):
            ids=old['T_array_rows'][lo:lo+1024];actual=raw[ids][:,46:63].numpy();assert np.array_equal(actual,array[lo:lo+len(ids)])
        dqs=queries[si*820:(si+1)*820];dvalues.append(raw[[q['array_row'] for q in dqs]][:,46:63].clone().numpy())
        if si==0:check=raw[[q['array_row'] for q in panel['Dcheck']]][:,46:63].clone().numpy()
        source_rows.append(dict(subject=name,source=source,all13230_T_rows_bitwise_D04_match=True,Dfit_array_rows=[q['array_row'] for q in dqs]));del raw,array
    D=np.stack(dvalues);assert D.shape==(10,820,17,250) and check.shape==(400,17,250) and np.isfinite(D).all() and np.isfinite(check).all()
    U.save_array(a.output/'all10_Dfit17_R4.npy',D);U.save_array(a.output/'sub01_Dcheck17_R4.npy',check)
    U.atomic(a.output/'mapping.json',dict(T_array_rows=old['T_array_rows'],Dfit_queries=queries,Dcheck_queries=panel['Dcheck'],subjects=list(old['subjects'])))
    result=dict(status='D17_ALL10_COMPLETE_T17_RAW_REPLAY_AND_REUSABLE_DFIT_DCHECK_EEG_CACHE_VERIFIED',source=U.ref(__file__),sources=source_rows,
        Dfit=U.ref(a.output/'all10_Dfit17_R4.npy'),Dcheck=U.ref(a.output/'sub01_Dcheck17_R4.npy'),mapping=U.ref(a.output/'mapping.json'),
        all132300_T_input_values_bitwise_original_R4=True,Dfit_queries=8200,Dcheck_queries=400,Dfit_Dcheck_target_F_or_images_loaded=False,
        new_GPU_jobs=0,new_parameter_updates=0,CUDA_initialized=False,original_D_resumed=False,seconds=time.monotonic()-start)
    assert not torch.cuda.is_initialized();U.atomic(a.output/'completion.json',result);print(json.dumps({k:result[k] for k in ('status','seconds','Dfit_queries','Dcheck_queries')}))

if __name__=='__main__':main()
