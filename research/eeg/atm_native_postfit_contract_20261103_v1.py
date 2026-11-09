"""Fixed four-arm native ATM comparison; no novel-method or SOTA inference."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
ARMS=('parent','ordinary','F_only','T_mean')
CONTRASTS=(('ordinary','parent'),('ordinary','F_only'),('ordinary','T_mean'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def checked(pin):
    p=Path(pin['path']);assert p.is_file() and sha(p)==pin['sha256']
    if 'bytes' in pin:assert p.stat().st_size==pin['bytes']
    return p
def ref(path):return {'path':str(path),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def write(path,value):
    p=Path(path);temporary=p.with_name(p.name+'.pending');assert not temporary.exists()
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(p)
def save_array(path,value):
    p=Path(path);temporary=p.with_name(p.name+'.pending');assert not temporary.exists()
    with temporary.open('xb') as stream:np.save(stream,value,allow_pickle=False)
    temporary.replace(p)
def process(pid,ticks=None):
    p=Path('/proc',str(pid))
    try:
        st=(p/'stat').read_text().rsplit(')',1)[1].split()
        if ticks is not None:assert int(st[19])==ticks
        return {'PID':pid,'start_ticks':int(st[19]),'state':st[0],'live':st[0] not in ('Z','X')}
    except FileNotFoundError:return {'PID':pid,'state':'MISSING','live':False}
def setup(q):
    parent=read(checked(q['parent_policy']))
    import sys
    sys.path.insert(0,str(Path(parent['image_protocol']['path']).parent))
    import brainhive_native_direction_image_contract_20261103_v1 as I
    _,train,e,old,base,G,H,C=I.setup(parent['image_protocol']['path'])
    return parent,train,e,C
def resources(q,parent,train,e,C,*,starting=False):
    root=Path(q['project_root']);used=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file())
    result=C.resources(train,e,starting=starting)
    assert used<parent['maximum_output_bytes']
    assert result['free_bytes']>=result['required_remaining_bytes']+parent['maximum_output_bytes']-used
    return {**result,'native_project_used_bytes':used,'native_project_cap_bytes':parent['maximum_output_bytes']}
def paired(arrays,queries):
    concepts=sorted({x['concept_id'] for x in queries})
    groups=[[i for i,x in enumerate(queries) if x['concept_id']==c] for c in concepts]
    assert all(len(g)==10 for g in groups)
    rng=np.random.default_rng(1729);draws=rng.integers(0,len(groups),size=(1000,len(groups)))
    result={}
    for a,b in CONTRASTS:
        values={}
        for key in arrays[a]:
            difference=np.asarray(arrays[a][key])-np.asarray(arrays[b][key])
            means=np.array([difference[g].mean() for g in groups])
            low,high=np.quantile(means[draws].mean(1),[.025,.975])
            values[key]={'difference':float(difference.mean()),'descriptive_concept_bootstrap95':[float(low),float(high)],
                         'positive_concepts':int((means>0).sum()),'concepts':len(groups)}
        result[a+'-'+b]=values
    return result
def interpret(comparisons,stage):
    assert stage in (80,400)
    native=comparisons['ordinary-parent'];semantic=native['DINO2AFC'];detail=native['within_concept10_two_afc']
    joint_harm=(semantic['difference']<=-.01 and semantic['descriptive_concept_bootstrap95'][1]<0
                and detail['difference']<0 and detail['descriptive_concept_bootstrap95'][1]<0)
    efficacy=(semantic['difference']>=.01 and semantic['descriptive_concept_bootstrap95'][0]>0)
    if stage==80:
        action='STOP_CURRENT_INSTANCE_JOINT_CLEAR_HARM' if joint_harm else 'EXPAND_ALL_FOUR_ARMS_TO400'
    else:action='COMPLETE_DEVELOPMENT_COMPARISON'
    support={}
    for key in ('ordinary-F_only','ordinary-T_mean'):
        value=comparisons[key]['DINO2AFC']
        support[key]='DESCRIPTIVE_POSITIVE' if value['descriptive_concept_bootstrap95'][0]>0 else 'ADDITIONAL_SEMANTIC_GAIN_NOT_ESTABLISHED'
    return {'action':action,'native_semantic_efficacy_supported_on_this_panel':bool(efficacy),
            'joint_clear_harm':bool(joint_harm),'mechanism_comparisons':support,
            'stimulus_direction':detail,'all_secondary_metrics_retained':True,
            'single_metric_failure_does_not_reject_method_family':True,
            'development_only':True,'nominal_optional_stopping_error_control':False,
            'second_independent_method_established':False,'SOTA_established':False,
            'publication_readiness':'NOT_ESTABLISHED_BY_THIS_SCREEN'}

def constant_direction_head(model,direction):
    """Reuse the exact bounded geometry with a fixed T-only raw direction."""
    import copy,torch
    from torch import nn
    class ConstantDirection(nn.Module):
        def __init__(self,value):super().__init__();self.register_buffer('value',value.detach().clone())
        def forward(self,x):return self.value[None].expand(len(x),-1)
    answer=copy.deepcopy(model);answer.direction=ConstantDirection(direction)
    return answer.eval().requires_grad_(False)
