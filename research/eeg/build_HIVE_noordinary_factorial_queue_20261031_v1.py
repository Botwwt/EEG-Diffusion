"""Build a separate, finite ordinary-independence experiment; never edit live code."""
import ast, hashlib, json
from pathlib import Path

L=Path(__file__).resolve().parent
OUT=L/'audit/HIVE_noordinary_factorial_queue_20261031_v1'
CODE=OUT/'code'
CODE.mkdir(parents=True,exist_ok=False)
OLD=("joint_correct","EEG_only_correct","P_only_correct","joint_swapped_targets","joint_absolute_only")
NEW=("joint_correct","joint_swapped_targets","joint_absolute_only")
names=[
 'brainhive_paired_semantic_study_contract_CPU_20261030_v1',
 'brainhive_paired_semantic_runtime_20261030_v2',
 'brainhive_paired_semantic_effect_20261030_v1',
 'brainhive_paired_semantic_images_streamed_20261030_v3',
 'train_HIVE_paired_semantic_EEG_P_GPU_20261030_v1',
 'run_HIVE_paired_semantic_images_20261030_v1',
 'audit_HIVE_paired_semantic_study_CPU_20261030_v1',
 'queue_HIVE_paired_semantic_study_CPU_20261030_v1']
mapping={n:n.replace('paired_semantic','noordinary_factorial').replace('20261030','20261031') for n in names}
manifest=[]
def change(s,a,b):
    assert a in s,repr(a)
    return s.replace(a,b)

for n in names:
    source=(L/(n+'.py')).read_text(encoding='utf8');s=source
    for a,b in mapping.items():s=s.replace(a,b)
    s=s.replace("ARMS=('joint_correct','EEG_only_correct','P_only_correct','joint_swapped_targets','joint_absolute_only')", "ARMS=('joint_correct','joint_swapped_targets','joint_absolute_only')")
    s=s.replace('ADMITTED_FULL_T_EEG_P_PAIRED_SEMANTIC_STUDY_V1','ADMITTED_NOORDINARY_FACTORIAL_FULL_T_V1')
    s=s.replace('REGISTERED_EEG_P_PAIRED_SEMANTIC_NATIVE_IMAGE_EXECUTION_V1','REGISTERED_NOORDINARY_FACTORIAL_NATIVE_IMAGES_V1')
    s=s.replace('all5_fixed_finals_verified','all3_fixed_finals_verified').replace('ALL5_FULL_T','ALL3_FULL_T')
    s=s.replace('all7_inference_paths_passed','all8_inference_paths_passed').replace('ALL7','ALL8').replace('30_COMPARISON','42_COMPARISON')
    if n.startswith('brainhive_paired_semantic_study_contract'):
        s=change(s,"IMAGE_ARMS=('parent','ordinary')+ARMS", "CANDIDATES=ARMS+tuple(a+'_plus_ordinary' for a in ARMS)\nIMAGE_ARMS=('parent','ordinary')+CANDIDATES\ndef model_arm(arm):return arm.removesuffix('_plus_ordinary')\ndef use_ordinary(arm):return arm=='ordinary' or arm.endswith('_plus_ordinary')")
        s=s[:s.index('def image_decision(')]+'''def image_decision(contrasts,stage):
    """Two fixed hypotheses; do not pick a control or use interaction to rescue."""
    modes={}
    for label,a,base,suffix in [('standalone','joint_correct','parent',''),('composition','joint_correct_plus_ordinary','ordinary','_plus_ordinary')]:
        c=contrasts[a];b=c[base];d=b['DINO2AFC']
        harm=d['difference']<=-.02 and d['descriptive_concept_bootstrap95'][1]<0
        efficacy=d['difference']>=.01 and d['descriptive_concept_bootstrap95'][0]>0 and b['within_concept10_top1']['difference']>=0
        controls={v:c[v]['DINO2AFC']['difference']>0 and c[v]['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 for v in ['joint_swapped_targets'+suffix,'joint_absolute_only'+suffix]}
        modes[label]=dict(clear_harm=harm,efficacy_support=efficacy,control_support=controls,supported=efficacy and all(controls.values()))
    assert stage in (80,400)
    if stage==80:action='STOP_INSTANCE' if all(v['clear_harm'] for v in modes.values()) else 'EXPAND_ALL_TO400'
    else:action='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION' if any(v['supported'] for v in modes.values()) else 'STOP_INSTANCE'
    return dict(action=action,modes=modes,two_predeclared_hypotheses=True,descriptive_intervals_not_multiplicity_corrected=True,not_SOTA=True)
'''
    if n.startswith('brainhive_paired_semantic_runtime'):
        s=change(s,"def load_context(s,out):", "def load_context(s,out):\n    assert s['ordinary_training_enabled'] is False")
        s=change(s,"corrected=context['ordinary'](c,w,b)[0]", "# The no-O fit is a distinct matched training policy, not inference-only bypass.\n    corrected=c")
    if n.startswith('train_HIVE'):
        s=change(s,"assert s['arms']==list(ARMS) and s['training_seed']==1729", "assert s['arms']==list(ARMS) and s['training_seed']==1729 and s['ordinary_training_enabled'] is False")
        s=change(s,"payload=dict(EEG=", "payload=dict(ordinary_training_enabled=False,EEG=")
        # Fail at every actual training step if ordinary were accidentally invoked.
        s=change(s,"original_versions={", "ordinary_calls=[]\n        ordinary_hook=context['ordinary'].register_forward_pre_hook(lambda *args:ordinary_calls.append(1))\n        original_versions={")
        s=change(s,"assert original_versions==", "ordinary_hook.remove();assert not ordinary_calls\n        assert original_versions==")
    if n.startswith('run_HIVE') or n.startswith('audit_HIVE'):
        s=change(s,"import ARMS,IMAGE_ARMS,", "import ARMS,IMAGE_ARMS,CANDIDATES,model_arm,use_ordinary,")
        s=s.replace('if arm in ARMS:', 'if model_arm(arm) in ARMS:')
        s=s.replace("/arm/'completion.json'", "/model_arm(arm)/'completion.json'")
        s=s.replace("certificate['arms'][arm]['model']", "certificate['arms'][model_arm(arm)]['model']")
        s=s.replace("c if arm=='parent' else ordinary(c,w,b)[0]", "ordinary(c,w,b)[0] if use_ordinary(arm) else c")
        s=s.replace("c if arm=='parent' else manual_ordinary(c,O,gamma,beta,torch,F)", "manual_ordinary(c,O,gamma,beta,torch,F) if use_ordinary(arm) else c")
        s=s.replace('for a in ARMS:', 'for a in CANDIDATES:')
        s=s.replace('for a in ARMS}', 'for a in CANDIDATES}')
        s=s.replace('for a in ARMS for b in IMAGE_ARMS', 'for a in CANDIDATES for b in IMAGE_ARMS')
        s=s.replace('images=7*stage', 'images=len(IMAGE_ARMS)*stage').replace("images=(args.hi-args.lo)*7", "images=(args.hi-args.lo)*len(IMAGE_ARMS)")
        # Only image/token loops use model_arm; fit remains ARMS and actual folders.
        if n.startswith('audit_HIVE'):
            s=change(s,"assert final['registration']==ref(path)", "assert final['ordinary_training_enabled'] is False\n        assert final['registration']==ref(path)")
    if n.startswith('queue_HIVE'):
        s=change(s,"for arm in ARMS:\n", "run('noordinary_GPU_profile',py+[str(code/'profile_HIVE_noordinary_factorial_GPU_20261031_v1.py'),'--protocol',str(path)],900,True)\n        for arm in ARMS:\n")
        s=change(s,"for slot in range(3):", "for slot in range(4):")
    ast.parse(s)
    p=CODE/(mapping[n]+'.py');p.write_text(s,encoding='utf8',newline='\n')
    manifest.append(dict(old_source=n+'.py',old_sha256=hashlib.sha256(source.encode()).hexdigest(),new_source=p.name,new_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(OUT/'source_derivation.json').write_text(json.dumps(dict(status='SEPARATE_SOURCE_COPIES_NOT_LIVE_EDITS',files=manifest,training_arms=NEW,old_sources_unchanged=True),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(code=str(CODE),source_files=len(manifest))))
