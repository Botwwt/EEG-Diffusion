"""Actual complete OOF inputs and separate calibration, no invented interface."""
import json
from brainhive_native_models_v1 import *


def load_training(device='cuda'):
    path=WORK/'data/component_input_contract_v1.json';contract=json.loads(path.read_text())
    assert contract['status']=='ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT'
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text());ids=protocol['image_rows']['T']
    assert len(ids)==13230 and len(contract['subjects'])==10
    position={i:j for j,i in enumerate(ids)}
    all_c=[];all_affine_ids=[];affine_w=[];affine_b=[]
    for subject in contract['subjects']:
        c=torch.empty(13230,1024);coverage=torch.zeros(13230,dtype=torch.int16);aids=torch.full((13230,),-1,dtype=torch.long)
        for source in subject['sources']:
            if not source['role'].startswith('oof'):continue
            assert sha(source['condition_cache_path'])==source['condition_cache_sha256']
            assert sha(source['checkpoint_path'])==source['checkpoint_sha256']
            data=torch.load(source['condition_cache_path'],weights_only=True,map_location='cpu',mmap=True)
            model=torch.load(source['checkpoint_path'],weights_only=True,map_location='cpu')['model']
            assert data['model_sha256']==source['checkpoint_sha256']
            rows=source['groups']['component_train'];dst=torch.tensor([position[r['array_row']] for r in rows]);src=torch.tensor([r['cache_row'] for r in rows])
            assert torch.equal(data['array_rows'][src],torch.tensor([r['array_row'] for r in rows]))
            c[dst]=data['condition'][src];coverage[dst]+=1;aids[dst]=len(affine_w)
            affine_w.append(model['block.layers.0.norm.weight']);affine_b.append(model['block.layers.0.norm.bias'])
        assert torch.all(coverage==1) and torch.all(aids>=0) and torch.isfinite(c).all()
        all_c.append(c);all_affine_ids.append(aids)
    assert sha(contract['target_cache_path'])==contract['target_cache_sha256']
    target=torch.load(contract['target_cache_path'],weights_only=True,map_location='cpu',mmap=True)[ids].clone()
    assert target.shape==(13230,1024) and torch.isfinite(target).all()
    return dict(condition=torch.stack(all_c).reshape(132300,1024).to(device),
        affine_ids=torch.cat(all_affine_ids).to(device),weight=torch.stack(affine_w).to(device),bias=torch.stack(affine_b).to(device),
        target=target.to(device),subjects=[s['subject'] for s in contract['subjects']],array_rows=ids,
        input_contract_sha256=sha(path))


def load_development(split,device='cuda'):
    assert split in ['Dfit','Dcheck']
    contract=json.loads((WORK/'data/component_input_contract_v1.json').read_text());conditions=[];weight=[];bias=[];risk=[];queries=[];targets=[]
    target=torch.load(contract['target_cache_path'],weights_only=True,map_location='cpu',mmap=True)
    group='calibration_only_Dfit' if split=='Dfit' else 'calibration_check_only_Dcheck'
    for subject in contract['subjects']:
        source=next(r for r in subject['sources'] if r['role']=='full');assert sha(source['condition_cache_path'])==source['condition_cache_sha256']
        assert sha(source['checkpoint_path'])==source['checkpoint_sha256']
        data=torch.load(source['condition_cache_path'],weights_only=True,map_location='cpu',mmap=True)
        model=torch.load(source['checkpoint_path'],weights_only=True,map_location='cpu')['model'];rows=source['groups'][group]
        ids=[r['array_row'] for r in rows];src=[r['cache_row'] for r in rows]
        geometry=torch.load(WORK/'data/condition_geometry_v1'/(subject['subject']+'_'+split+'.pt'),weights_only=True,map_location='cpu')
        assert geometry['array_rows'].tolist()==ids
        conditions.append(data['condition'][src]);risk.append(1-geometry['values']['token_cosine'])
        weight.append(model['block.layers.0.norm.weight'].expand(len(ids),-1));bias.append(model['block.layers.0.norm.bias'].expand(len(ids),-1))
        targets.append(target[ids]);queries.extend(dict(subject=subject['subject'],array_row=i) for i in ids)
    assert len(queries)==(8200 if split=='Dfit' else 8300)
    return dict(condition=torch.cat(conditions).to(device),weight=torch.cat(weight).to(device),bias=torch.cat(bias).to(device),
        target=torch.cat(targets).to(device),risk=torch.cat(risk).to(device),queries=queries)
