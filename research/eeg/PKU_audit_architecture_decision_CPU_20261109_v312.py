"""Independent stored-prediction replay for both full-T architecture diagnostics."""
import csv
import gzip
import hashlib
import json
import os
import socket
import time
import traceback
from pathlib import Path

assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
R = Path('/dfs/share/maleiLab/wwt/eeg')
OUT = R / 'runs/EEG_architecture_independent_replay_v312'
OUT.mkdir(exist_ok=False)
start = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, value):
    p = OUT / name
    t = p.with_name(p.name + '.pending')
    t.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    t.replace(p)


try:
    import numpy as np
    import torch
    torch.set_num_threads(4)
    parent = R / 'runs/HIVE_native_sub03_5096542'
    bank = torch.load(parent / 'T_true_OOF_conditions.pt', weights_only=True, map_location='cpu')
    rows = bank['array_rows'].tolist()
    roles = bank['folds'].numpy()
    with (R / 'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:
        lookup = {int(v['array_row']): v for v in csv.DictReader(f) if v['dataset_split'] == 'train'}
    concepts = np.asarray([lookup[r]['concept_id'] for r in rows])
    positions = {r:i for i,r in enumerate(rows)}
    groups = {k:np.flatnonzero(concepts == k) for k in sorted(set(concepts))}
    candidates = np.stack([groups[k] for k in concepts])
    all_inputs = torch.load(R/'data/native_training_v239/train_visual_embeddings.pt', weights_only=True, map_location='cpu')
    views = {k:v[rows].double().numpy() for k,v in sorted(all_inputs.items())}
    del all_inputs, bank
    views['native_fused_F'] = torch.load(parent/'deterministic_F_all16540.pt', weights_only=True, map_location='cpu')[rows].double().numpy()
    receipts = {}
    for experiment in ['compression', 'fusion']:
        folder = R / ('runs/EEG_architecture_' + experiment + '_v312')
        completion = json.loads((folder/'completion.json').read_text())
        assert completion['status'].startswith('COMPLETE_ALL13230_SIX_SOURCE_ROLE_CONCEPT_ISOLATED_FITS_')
        assert sha(folder/'all_T_predictions.npy') == completion['predictions_SHA256']
        assert sha(folder/'all_T_metrics.csv.gz') == completion['table_SHA256']
        pred = np.load(folder/'all_T_predictions.npy', mmap_mode='r', allow_pickle=False)
        cases = completion['cases']
        names = list(completion['view_dimensions'])
        boundaries = np.cumsum([0]+[completion['view_dimensions'][k] for k in names])
        assert pred.shape == (len(cases), 13230, int(boundaries[-1]))
        cells = np.zeros((len(cases), len(names), 13230, 6), dtype=np.float64)
        covered = np.zeros(13230, dtype=bool)
        for role in [0, 1]:
            for fold in range(3):
                rec = json.loads((folder/('role'+str(role)+'_fold'+str(fold)+'.json')).read_text())
                fit = np.asarray([positions[r] for r in rec['outer_fit_rows']])
                test = np.asarray([positions[r] for r in rec['outer_test_rows']])
                assert not covered[test].any() and np.all(roles[test] == role)
                assert set(concepts[fit]).isdisjoint(concepts[test])
                covered[test] = True
                for ci,case in enumerate(cases):
                    for vi,name in enumerate(names):
                        y = views[name]
                        p = np.asarray(pred[ci,test,boundaries[vi]:boundaries[vi+1]], dtype=np.float64)
                        mse = np.mean((p-y[test])**2, axis=1)
                        variance = max(float(np.mean((y[fit]-y[fit].mean(0))**2)), 1e-12)
                        unit = p/np.maximum(np.sqrt(np.sum(p*p,axis=1,keepdims=True)),1e-12)
                        target = y/np.maximum(np.sqrt(np.sum(y*y,axis=1,keepdims=True)),1e-12)
                        cosine = np.sum(unit*target[test],axis=1)
                        score = np.sum(unit[:,None,:]*target[candidates[test]],axis=2)
                        mask = candidates[test] != test[:,None]
                        wrong = score[mask].reshape(len(test),9)
                        margin = cosine-wrong.max(1)
                        afc = np.sum((cosine[:,None]>wrong)+.5*(cosine[:,None]==wrong),axis=1)/9
                        cells[ci,vi,test] = np.stack([mse,mse/variance,cosine,margin,(margin>0).astype(float),afc],axis=1)
        assert covered.all()
        fields = ['MSE','variance_normalized_MSE','cosine','same_concept_margin','same_concept_strict_Top1','same_concept_2AFC']
        maxerror = 0.
        strict_mismatch = 0
        count = 0
        observed = set()
        with gzip.open(folder/'all_T_metrics.csv.gz','rt') as f:
            for row in csv.DictReader(f):
                ci=cases.index(row['case']);vi=names.index(row['view']);i=positions[int(row['array_row'])]
                key=(ci,vi,i)
                assert key not in observed and row['concept_id']==concepts[i]
                assert int(row['source_encoder_role'])==int(roles[i])
                observed.add(key);count+=1
                values=np.asarray([float(row[k]) for k in fields])
                maxerror=max(maxerror,float(np.max(np.abs(cells[ci,vi,i]-values))))
                strict_mismatch+=int(cells[ci,vi,i,4]!=values[4])
        assert count==len(cases)*len(names)*13230 and maxerror<1e-9 and strict_mismatch==0
        means_error=max(abs(float(cells[ci,vi,:,mi].mean())-completion['means'][case][name][field]) for ci,case in enumerate(cases) for vi,name in enumerate(names) for mi,field in enumerate(fields))
        assert means_error<1e-10
        receipts[experiment]=dict(status='PASS_ALL_STORED_PREDICTIONS_AND_ALL_METRIC_ROWS', rows=count, metric_cells=count*6, maximum_absolute_cell_error=maxerror, strict_Top1_mismatches=strict_mismatch, means_maximum_error=means_error, source_completion_SHA256=sha(folder/'completion.json'), privileged_F_input=(experiment=='fusion'))
        save(experiment+'_audit.json',receipts[experiment])
        del pred,cells
        assert time.monotonic()-start<1100
    decision=dict(status='COMPLETE_BOTH_FULL_T_DIAGNOSTICS_WITH_INDEPENDENT_ALL_ROW_REPLAY', CPU_job=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-start, audits=receipts,
        interpretation='Evaluate native compression using raw17 versus c and their matched within-concept controls. Evaluate fusion using original-view recovery from privileged nonlinear F versus F+EEG and matched EEG controls. Read all geometry and identity metrics together; kernel-family limitations remain. These experiments can justify the location of an architectural change, not establish a new method or a SOTA claim.', new_models_submitted_automatically=0, new_GPU_jobs=0, new_images=0, novelty_established=False)
    save('completion.json',decision)
    print(json.dumps(decision),flush=True)
except BaseException as e:
    save('failure.json',dict(status='FAILED_PRESERVED_NO_AUTO_RETRY',error=repr(e),traceback=traceback.format_exc()))
    raise
