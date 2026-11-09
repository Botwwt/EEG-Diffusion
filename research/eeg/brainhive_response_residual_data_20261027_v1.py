"""D16 source-bound OOF response inputs and native-c/F coordinates."""
import csv
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U

R=Path(__file__).resolve().parent


def tensor_array(reference,device):
    return torch.from_numpy(np.array(np.load(U.checked(reference),allow_pickle=False,mmap_mode='r'),copy=True)).to(device)


def load_T(spec,device='cuda'):
    from brainhive_component_data_v1 import load_training
    training=load_training(device);exported=U.read(U.checked(spec['response_export']));old=U.read(U.checked(spec['D15_probe']))
    original=U.read(U.checked(exported['D15_export']));lineage=U.read(U.checked(exported['inputs']))
    assert training['subjects']==list(lineage['subjects']) and training['array_rows']==lineage['T_array_rows']
    prior=torch.load(U.checked(old['model']),weights_only=True,map_location='cpu',mmap=True)
    state={key:prior[key].to(device) for key in ('c_mean','c_scale','ATMS_mean','ATMS_scale','F_mean','F_scale','F_weight','F_bias')}
    state['Bc']=prior['models']['c_to_y'].to(device)
    u=(training['condition']-training['bias'][training['affine_ids']])/training['weight'][training['affine_ids']]
    subject=torch.arange(10,device=device).repeat_interleave(13230)
    atms=tensor_array(original['T_ATMS_OOF'],device)
    atms.sub_(state['ATMS_mean'][subject]).div_(state['ATMS_scale'][subject])
    response=tensor_array(exported['T_responses'],device).reshape(10,13230,4,1024)
    rm=response.mean(1,keepdim=True);rs=(response-rm).square().mean((1,3),keepdim=True).sqrt();assert rs.min()>1e-6
    response.sub_(rm).div_(rs);state['response_mean']=rm.squeeze(1);state['response_scale']=rs.squeeze(1)
    response=response.reshape(132300,4,1024)
    target=((training['target']-state['F_bias'])/state['F_weight']-state['F_mean'])/state['F_scale']
    metadata={int(x['array_row']):x for x in csv.DictReader((R/'audit/image_manifest.csv').open()) if x['dataset_split']=='train'}
    concepts=sorted({metadata[row]['concept_id'] for row in training['array_rows']});assert len(concepts)==1323
    local=[[j for j,row in enumerate(training['array_rows']) if metadata[row]['concept_id']==name] for name in concepts];assert all(len(x)==10 for x in local)
    groups=np.concatenate([np.array(local)+si*13230 for si in range(10)])
    permutation=np.empty(132300,np.int64);gp=U.derangement(1323)
    for si in range(10):
        for j,group in enumerate(local):permutation[si*13230+np.array(group)]=si*13230+np.array(local[gp[j]])
    assert sorted(groups.reshape(-1).tolist())==list(range(132300))
    return dict(u=u,subject=subject,atms=atms,responses=response,target_unique=target,
        groups=torch.as_tensor(groups,device=device),permutation=torch.as_tensor(permutation,device=device),state=state,
        input_contract_sha256=training['input_contract_sha256'],T_array_rows=training['array_rows'])


def load_Dfit(spec,state,device='cuda'):
    exported=U.read(U.checked(spec['response_export']));original=U.read(U.checked(exported['D15_export']));lineage=U.read(U.checked(exported['inputs']))
    old=U.read(U.checked(spec['D15_probe']));mapping=U.read(U.checked(old['queries_and_controls']))
    contract=U.read('/root/brainhive_reproduction_20261020_v1/data/component_input_contract_v1.json')
    rawF=torch.load(U.checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
    ad=np.load(U.checked(original['D_ATMS_full']),allow_pickle=False,mmap_mode='r');rd=np.load(U.checked(exported['D_responses']),allow_pickle=False,mmap_mode='r')
    dl={row:j for j,row in enumerate(lineage['D_array_rows'])};us=[];aa=[];rr=[];ys=[];cs=[];weights=[];biases=[];queries=mapping['queries']
    for si,subject in enumerate(contract['subjects']):
        assert subject['subject']==f'sub-{si+1:02d}'
        full=next(x for x in subject['sources'] if x['role']=='full')
        cache=torch.load(U.checked(dict(path=full['condition_cache_path'],sha256=full['condition_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
        model=torch.load(U.checked(dict(path=full['checkpoint_path'],sha256=full['checkpoint_sha256'])),weights_only=True,map_location='cpu')['model']
        rows=full['groups']['calibration_only_Dfit'];ids=[q['array_row'] for q in rows];ix=[q['cache_row'] for q in rows]
        assert cache['array_rows'][ix].tolist()==ids==[q['array_row'] for q in queries[si*820:(si+1)*820]]
        c=cache['condition'][ix].to(device);w=model['block.layers.0.norm.weight'].to(device);b=model['block.layers.0.norm.bias'].to(device)
        us.append((c-b)/w);cs.append(c);weights.append(w.expand(820,-1));biases.append(b.expand(820,-1))
        indices=[si*1650+dl[row] for row in ids]
        a=torch.from_numpy(np.array(ad[indices],copy=True)).to(device);r=torch.from_numpy(np.array(rd[indices],copy=True)).to(device)
        aa.append((a-state['ATMS_mean'][si])/state['ATMS_scale'][si]);rr.append((r-state['response_mean'][si])/state['response_scale'][si])
        ys.append(((rawF[ids].to(device)-state['F_bias'])/state['F_weight']-state['F_mean'])/state['F_scale'])
    return dict(u=torch.cat(us),condition=torch.cat(cs),weight=torch.cat(weights),bias=torch.cat(biases),atms=torch.cat(aa),responses=torch.cat(rr),target=torch.cat(ys),
        subject=torch.arange(10,device=device).repeat_interleave(820),queries=queries,groups=mapping['groups'],
        permutation=torch.as_tensor(mapping['shuffled_innovation_source_indices'],device=device),
        linear_predictions=tensor_array(old['Dfit_predictions'],device),linear_source=old['Dfit_predictions'])
