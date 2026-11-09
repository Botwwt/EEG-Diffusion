"""Derive an eight-method conditional evaluator without changing either live queue."""
import ast,json,hashlib,base64
from pathlib import Path
L=Path(__file__).resolve().parent;OUT=L/'audit/HIVE_noordinary_factorial_queue_20261031_v1';CODE=OUT/'secondary_code';CODE.mkdir(exist_ok=False)
contract='brainhive_noordinary_factorial_study_contract_CPU_20261031_v1'
names=['run_HIVE_paired_semantic_SEED_followon_20261030_v1','queue_HIVE_paired_semantic_SEED_followon_CPU_20261030_v1','audit_HIVE_paired_semantic_SEED_CPU_20261030_v1','brainhive_paired_semantic_SEED_statistics_CPU_20261030_v1']
mapping={n:n.replace('paired_semantic','noordinary_factorial').replace('20261030','20261031') for n in names}
for name in names:
    s=(L/(name+'.py')).read_text(encoding='utf8')
    for a,b in mapping.items():s=s.replace(a,b)
    s=s.replace('brainhive_paired_semantic_study_contract_CPU_20261030_v1',contract)
    s=s.replace('import ARMS,IMAGE_ARMS,','import CANDIDATES as ARMS,IMAGE_ARMS,')
    s=s.replace('ALL7','ALL8').replace('all7','all8').replace('2800','3200').replace('30_COMPARISON','42_COMPARISON')
    s=s.replace('(8*192,','(9*192,')
    if name.startswith('brainhive_'):
        s=s[:s.index('def next_action(')]+'''def next_action(upstream_decision,contrasts):
    assert upstream_decision['action']=='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION'
    modes={}
    for label,a,b in [('standalone','joint_correct','parent'),('composition','joint_correct_plus_ordinary','ordinary')]:
        primary=contrasts[a][b];seed=primary['SEED'];f1=primary['ObjectF1']
        core=upstream_decision['modes'][label]['supported']
        support=core and seed['difference']>0 and seed['descriptive_concept_bootstrap95'][0]>0 and f1['difference']>=0
        modes[label]=dict(original_DINO_and_control_gate_retained=core,SEED_and_ObjectF1_support=support)
    support=any(v['SEED_and_ObjectF1_support'] for v in modes.values())
    return dict(action='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION' if support else 'HOLD_FOR_ANALYSIS_NO_AUTOMATIC_MULTISUBJECT_FIT',modes=modes,development_SEED_and_ObjectF1_support=support,does_not_rescue_either_failed_primary_mode=True,CLIP_training_reward_NOT_independent_confirmation=True,official_R80=False,independent_H=False,SOTA=False)
'''
    ast.parse(s);(CODE/(mapping[name]+'.py')).write_bytes(s.encode())
(CODE/(contract+'.py')).write_bytes((OUT/'code'/(contract+'.py')).read_bytes())
print(json.dumps(dict(output=str(CODE),files=len(list(CODE.glob('*.py'))))))
