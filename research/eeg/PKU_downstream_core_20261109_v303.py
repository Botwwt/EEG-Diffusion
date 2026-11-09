"""Existing P or native IP weights; fixed EEG/ordinary and clean-visual replay."""
import copy,hashlib,json,os
from pathlib import Path
import torch
R=Path('/dfs/share/maleiLab/wwt/eeg');ROOT=R/'runs/EEG_downstream_v303'
CASES=['P_positive','P_replay','P_pair','P_replay_pair','P_replay_reverse','IP_positive','IP_replay','IP_replay_pair','IP_replay_reverse']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,r):
    p=Path(p);b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode();tmp=p.with_name(p.name+'.pending')
    with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert tmp.read_bytes()==b;tmp.replace(p)
def save(p,r):
    p=Path(p);tmp=p.with_name(p.name+'.pending')
    with tmp.open('xb') as f:torch.save(r,f);f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def flags(case):
    assert case in CASES
    return dict(replay=float('replay' in case),pair=float('pair' in case or 'reverse' in case),sign=-1. if 'reverse' in case else 1.)
def target_modules(pipe,P,case):
    if case.startswith('P_'):return {'P':P}
    d={k:v for k,v in pipe.unet.attn_processors.items() if k.endswith('attn2.processor')};assert len(d)==70
    assert sum(p.numel() for m in d.values() for p in m.parameters())==340787200
    return d
def states(modules):return {k:{n:v.detach().cpu().clone() for n,v in m.state_dict().items()} for k,m in modules.items()}
def load_states(modules,s):
    assert set(modules)==set(s)
    for k,m in modules.items():m.load_state_dict(s[k],strict=True)
def setup_target(pipe,P,case):
    modules=target_modules(pipe,P,case)
    for m in modules.values():m.float().requires_grad_(True)
    refs={k:copy.deepcopy(m).requires_grad_(False).eval() for k,m in modules.items()}
    if case.startswith('P_'):refP=refs['P'];ref_processors=None
    else:refP=P;ref_processors=dict(pipe.unet.attn_processors);ref_processors.update(refs)
    pars=[p for m in modules.values() for p in m.parameters()]
    assert sum(p.numel() for p in pars)==(8400896 if case.startswith('P_') else 340787200)
    return modules,refs,refP,ref_processors,pars
def fixed_inputs(device):
    from PKU_ordinary_component_source_v253 import BoundedConditionAdapter
    parent=R/'runs/HIVE_native_sub03_5096542';split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());ids=split['image_rows']['T']
    bank=torch.load(parent/'T_true_OOF_conditions.pt',map_location='cpu',weights_only=True);assert bank['array_rows'].tolist()==ids and bank['subject']=='sub-03'
    hs=R/'runs/HIVE_matching_ordinary_calibration_v264/component_final.pt';assert sha(hs)=='f6998db9a49d566547bfbca045d47d471c771d4d0c6880322fbd9067d7024f99'
    head_state=torch.load(hs,map_location='cpu',weights_only=True);head=BoundedConditionAdapter(predict_risk=False).to(device).eval().requires_grad_(False);head.load_state_dict(head_state['model'],strict=True)
    assert head_state['calibration']['selected_strength']==1.
    affine=[]
    for role in ['oof0','oof1']:
        m=torch.load(parent/role/'brain_final.pt',map_location='cpu',weights_only=True)['model'];affine.append((m['block.layers.0.norm.weight'],m['block.layers.0.norm.bias']))
    gamma=torch.stack([v[0] for v in affine]).to(device);beta=torch.stack([v[1] for v in affine]).to(device);fold=bank['folds'].long().to(device)
    with torch.no_grad():c=head(bank['condition'].to(device),gamma[fold],beta[fold],gate_mode='fixed')[0].detach()
    full=torch.load(parent/'deterministic_F_all16540.pt',map_location='cpu',weights_only=True);F=full[ids].clone().to(device)
    return c,F,dict(OOF_sha256=sha(parent/'T_true_OOF_conditions.pt'),ordinary_sha256=sha(hs),F_sha256=sha(parent/'deterministic_F_all16540.pt'),head_inference_strength=1.,upstream_OOF=True,whole_head_not_OOF=True)
