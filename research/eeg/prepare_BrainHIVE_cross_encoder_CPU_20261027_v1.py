"""Audit original ATMS sources, all30 checkpoints, full R4 inputs and OOF roles."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U

R=Path(__file__).resolve().parent
SOURCE=Path('/root/atm_official_20261016')
LEGACY=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runs/idea_first_ATMS_patience50_v1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic()
    assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    config=U.read(R/'configs/idea_first_ATMS_training_v1.json'); split=U.read(R/'configs/split_manifest.json')
    assert U.sha(R/'configs/split_manifest.json')==config['split_sha256']
    assert U.sha(SOURCE/'models/atms.py')==config['official_atms_source_sha256']
    addons=[Path('/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages'),Path('/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages')]
    sys.path[:0]=[str(SOURCE)]+[str(p) for p in addons]
    from models.atms import ATMS
    metadata={int(x['array_row']):x for x in csv.DictReader((R/'audit/image_manifest.csv').open()) if x['dataset_split']=='train'}
    source_rows=list(csv.DictReader((R/'audit/data_manifest.csv').open()))
    bindings=U.read('/root/brainhive_reproduction_20261020_v1/data/actual_input_bindings.json')
    T=split['image_rows']['T']; D=split['image_rows']['D'];assert len(T)==13230 and len(D)==1650
    folds=np.array([split['oof_fold_by_concept'][metadata[i]['concept_id']] for i in T]);assert set(folds)=={0,1}
    locks={}
    for fold in (0,1):
        path=LEGACY/f'oof{fold}'/'OOF_data_roles.json';lock=U.read(path)
        held={i for i,f in zip(T,folds) if f==fold};train=set(T)-held
        assert set(lock['training_array_rows'])==train and set(lock['heldout_T_array_rows'])==held
        assert not set(lock['training_concepts'])&set(lock['heldout_concepts'])
        assert train.isdisjoint(D+split['image_rows']['V']);locks[f'oof{fold}']=U.ref(path)
    sources={x['path']:x for x in U.read(R/'configs/evidence_guided_prior_20261027_v1/candidate_spec.json')['source_pins']}
    def pin(path):
        item=U.ref(path);sources[item['path']]=item;return item
    for path in (Path(__file__),R/'export_BrainHIVE_cross_encoder_GPU_20261027_v1.py',R/'configs/idea_first_ATMS_training_v1.json',
                 R/'train_full_reference.py',R/'train_idea_first_atms_v1.py',R/'configs/split_manifest.json',R/'audit/data_manifest.csv',R/'audit/image_manifest.csv',
                 Path('/root/brainhive_reproduction_20261020_v1/data/actual_input_bindings.json')):pin(path)
    for path in sorted((SOURCE/'models').rglob('*.py')):pin(path)
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).suffix=='.py' and any(Path(path).is_relative_to(p) for p in addons):pin(path)
    for item in locks.values():sources[item['path']]=item
    reference={};subjects={}
    for si in range(1,11):
        name=f'sub-{si:02d}';old=next(x for x in source_rows if x['subject']==name and x['dataset_split']=='train' and x['repeat_protocol']=='R4')
        actual=bindings['subjects'][name]
        assert actual['train_sha256']==old['sha256'],'New input must be the exact originally fitted R4 EEG cache'
        input_ref=dict(path=actual['train_path'],sha256=actual['train_sha256']);U.checked(input_ref);sources[input_ref['path']]=input_ref
        eeg=torch.load(input_ref['path'],weights_only=True,map_location='cpu',mmap=True);assert eeg.shape==(16540,63,250)
        records={}
        for role in ('oof0','oof1','full'):
            completion=U.read(LEGACY/role/name/'completion.json');assert completion['status']=='COMPLETED' and completion['role']==role and completion['subject']==name
            assert completion['split_sha256']==config['split_sha256'] and completion['prior'] is None
            checkpoint=dict(path=completion['encoder']['checkpoint'],sha256=completion['encoder']['checkpoint_sha256']);U.checked(checkpoint);sources[checkpoint['path']]=checkpoint
            model=ATMS().eval().requires_grad_(False);state=torch.load(checkpoint['path'],weights_only=True,map_location='cpu',mmap=True);model.load_state_dict(state,strict=True)
            assert sum(q.numel() for q in model.parameters())==3200413
            rows=([i for i,f in zip(T,folds) if f==int(role[-1])] if role!='full' else D)
            selected=[rows[0],rows[-1]];x=eeg[selected].clone();assert torch.isfinite(x).all()
            with torch.inference_mode():value=model(x,torch.full((2,),si,dtype=torch.long)).float()
            assert value.shape==(2,1024) and torch.isfinite(value).all()
            norm=model.proj_eeg[2];assert isinstance(norm,torch.nn.LayerNorm) and norm.weight.abs().min()>1e-6
            reference[name+'/'+role]=dict(array_rows=selected,prediction=value,weight=norm.weight.detach().clone(),bias=norm.bias.detach().clone(),eps=norm.eps)
            records[role]=dict(completion=pin(LEGACY/role/name/'completion.json'),checkpoint=checkpoint,export_array_rows=rows,
                input_rows_per_query=[63,250],held_concepts_excluded=(role!='full'),CPU_parity_array_rows=selected)
            del model,state
        subjects[name]=dict(input=input_ref,models=records);del eeg
    parity=a.output/'CPU_reference.pt';torch.save(reference,parity)
    for item in sources.values():U.checked(item)
    result=dict(status='D15_ALL30_STRICT_MODELS_ORIGINAL_R4_INPUTS_AND_OOF_ROLES_CPU_VERIFIED',source=U.ref(__file__),source_pins=list(sources.values()),
        legacy_training_config=U.ref(R/'configs/idea_first_ATMS_training_v1.json'),split=U.ref(R/'configs/split_manifest.json'),
        subjects=subjects,T_array_rows=T,D_array_rows=D,T_heldout_folds=folds.tolist(),CPU_reference=U.ref(parity),
        expected_T_OOF_predictions=132300,expected_D_full_predictions=16500,parameters_per_existing_model=3200413,
        strict_CPU_model_loads=30,CPU_forward_queries=60,normalization='Original raw pooled CLIP_H output; learned output affine recorded, no coordinate substitution',
        original_V_model_selection_history=True,original_train_only_MVNN_independent_replay=False,
        original_per_role_prediction_cache_available=False,old_prediction_bytes_equality_claim=False,
        new_training_updates=0,new_images=0,original_D_resumed=False,CUDA_initialized=False,seconds=time.monotonic()-start)
    assert time.monotonic()-start<60 and not torch.cuda.is_initialized();U.atomic(a.output/'completion.json',result)
    print(json.dumps({k:result[k] for k in ('status','strict_CPU_model_loads','CPU_forward_queries','seconds')}))


if __name__=='__main__':main()
