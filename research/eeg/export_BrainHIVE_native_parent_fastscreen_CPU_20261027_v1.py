"""Export a reusable, identity-verified native OOF baseline without fitting."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')
INPUT=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/trialset_fastscreen_20261027_v1/inputs')
OUT=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/continuous_iteration_20261027_v1/shared/native_parent')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True,exist_ok=True);assert not any(OUT.iterdir())
    index=[json.loads(x) for x in (INPUT/'index.jsonl').read_text().splitlines()]
    assert len(index)==13230 and [r['row'] for r in index]==list(range(13230))
    pos={r['source_row']:r['row'] for r in index}
    contract_path=WORK/'data/component_input_contract_v1.json';contract=json.loads(contract_path.read_text())
    subject=next(s for s in contract['subjects'] if s['subject']=='sub-01')
    prediction=torch.empty(13230,1024);coverage=torch.zeros(13230,dtype=torch.int16);sources=[]
    for s in subject['sources']:
        if not s['role'].startswith('oof'):continue
        assert sha(s['condition_cache_path'])==s['condition_cache_sha256']
        assert sha(s['checkpoint_path'])==s['checkpoint_sha256']
        data=torch.load(s['condition_cache_path'],weights_only=True,mmap=True,map_location='cpu')
        assert data['model_sha256']==s['checkpoint_sha256']
        group=s['groups']['component_train'];dst=torch.tensor([pos[r['array_row']] for r in group]);src=torch.tensor([r['cache_row'] for r in group])
        assert torch.equal(data['array_rows'][src],torch.tensor([r['array_row'] for r in group]))
        assert all(index[int(i)]['fold']==int(s['role'][-1]) for i in dst)
        prediction[dst]=data['condition'][src];coverage[dst]+=1
        sources.append({k:v for k,v in s.items() if k!='groups'})
    assert (coverage==1).all() and torch.isfinite(prediction).all()
    path=OUT/'predictions.npy';np.save(path,prediction.numpy(),allow_pickle=False)
    binding=dict(status='COMPLETE_NATIVE_PARENT_OOF_PREDICTIONS',subject_id='sub-01',
                 prediction_index=str(INPUT/'index.jsonl'),prediction_index_sha256=sha(INPUT/'index.jsonl'),
                 predictions_path=str(path),predictions_sha256=sha(path),predictions_shape=[13230,1024],
                 oof_coverage_min=1,oof_coverage_max=1,
                 prediction_coordinates='raw native BrainMLP OOF output; original affine retained; no query target access or scaling',
                 source_contract_sha256=sha(contract_path),sources=sources,no_new_fit=True,CUDA_initialized=False)
    (OUT/'binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    target=np.load(INPUT/'targets.npy',allow_pickle=False,mmap_mode='r')
    stats={}
    for held in (0,1):
        tr=np.array([i for i,r in enumerate(index) if r['fold']!=held]);te=np.array([i for i,r in enumerate(index) if r['fold']==held])
        mu=np.asarray(target[tr]).mean(0);test=np.asarray(target[te]);norm=np.linalg.norm(test,axis=1)
        stats[str(held)]=dict(fit_rows=len(tr),held_rows=len(te),constant_F_train_mean_held_MSE=float(((test-mu)**2).mean()),
                             constant_F_train_mean_held_cosine=float(((test@mu)/(norm*np.linalg.norm(mu))).mean()),
                             F_fit_mean_energy_fraction=float((mu@mu)/(np.asarray(target[tr])**2).sum(1).mean()),
                             F_held_norm_mean=float(norm.mean()),native_condition_held_norm_mean=float(prediction[te].norm(dim=1).mean()))
    (OUT/'F_constant_baseline_diagnostic.json').write_text(json.dumps(dict(status='TRAIN_FOLD_MEAN_ON_HELD_F_DIAGNOSTIC_ONLY',folds=stats,normal_inference_not_using_targets=True),indent=2)+'\n')
    print(json.dumps(dict(status=binding['status'],binding=str(OUT/'binding.json'),binding_sha256=sha(OUT/'binding.json'),F_constant_baseline=stats)),flush=True)


if __name__=='__main__':main()
