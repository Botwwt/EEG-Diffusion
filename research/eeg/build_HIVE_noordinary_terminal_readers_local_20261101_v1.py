"""Prepare three-fit/eight-path evidence readers without modifying sealed jobs."""
import ast,json,hashlib
from pathlib import Path
L=Path(__file__).resolve().parent;A=L/'audit/crossdisciplinary_20261028_v1';O=L/'audit/HIVE_noordinary_terminal_reader_preparation_20261101_v1';O.mkdir(exist_ok=False)
collector=(A/'fetch_HIVE_paired_semantic_terminal_OLD_CPU_20261030_v1.py').read_text(encoding='utf8')
auditor=(L/'audit_HIVE_paired_semantic_terminal_local_20261030_v1.py').read_text(encoding='utf8')
old="ARMS = ('joint_correct', 'EEG_only_correct', 'P_only_correct',\n        'joint_swapped_targets', 'joint_absolute_only')\nMETHODS = ('parent', 'ordinary') + ARMS"
new="ARMS = ('joint_correct', 'joint_swapped_targets', 'joint_absolute_only')\nCANDIDATES = ARMS + tuple(a + '_plus_ordinary' for a in ARMS)\nMETHODS = ('parent', 'ordinary') + CANDIDATES"
def change(s,a,b):
    assert a in s,repr(a);return s.replace(a,b)
collector=change(collector,old,new);auditor=change(auditor,old,new)
for key in ['all5_fixed_finals_verified','all7_inference_paths_passed']:
    replacement=key.replace('5','3').replace('7','8');collector=collector.replace(key,replacement);auditor=auditor.replace(key,replacement)
collector=collector.replace("final['images'] == 7 * stage","final['images'] == 8 * stage")
collector=change(collector,"s = read(launch['protocol']['path'])", "s = read(launch['protocol']['path'])\n    assert s['ordinary_training_enabled'] is False")
collector=change(collector,"for name in ('controller_completion.json', 'controller_progress.json', 'independent_fit_CPU_audit.json'):", "for name in ('controller_completion.json', 'controller_progress.json', 'independent_fit_CPU_audit.json', 'waiter_completion.json', 'actual_dependency_and_resource_admission.json', 'composition_interaction.json'):")
collector=change(collector,"for path in sorted((root / 'lifecycle').glob('*.json')):", "profile = read(root / 'engineering/completion.json')\n    assert profile['status']=='PASS_ACTUAL_NOORDINARY_THREE_BRANCH_PROFILE' and profile['ordinary_calls']==0 and profile['temporary_updates_discarded']==3\n    add('actual/engineering/completion.json', root / 'engineering/completion.json')\n    for path in sorted((root / 'lifecycle').glob('*.json')):")
start=auditor.index('def decision(');end=auditor.index('def verify_statistics(')
auditor=auditor[:start]+'''def decision(contrasts, stage):
    assert stage in (80,400)
    modes={}
    for label,primary,base,controls in [
      ('standalone','joint_correct','parent',('joint_swapped_targets','joint_absolute_only')),
      ('composition','joint_correct_plus_ordinary','ordinary',('joint_swapped_targets_plus_ordinary','joint_absolute_only_plus_ordinary'))]:
        values=contrasts[primary];d=values[base]['DINO2AFC']
        damage=d['difference']<=-.02 and d['descriptive_concept_bootstrap95'][1]<0
        practical=d['difference']>=.01 and d['descriptive_concept_bootstrap95'][0]>0 and values[base]['within_concept10_top1']['difference']>=0
        mechanism={name:values[name]['DINO2AFC']['difference']>0 and values[name]['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 for name in controls}
        modes[label]=dict(clear_harm=damage,efficacy_support=practical,control_support=mechanism,supported=practical and all(mechanism.values()))
    if stage==80:
        action='STOP_INSTANCE' if modes['standalone']['clear_harm'] and modes['composition']['clear_harm'] else 'EXPAND_ALL_TO400'
    else:
        action='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION' if modes['standalone']['supported'] or modes['composition']['supported'] else 'STOP_INSTANCE'
    return dict(action=action,modes=modes,two_predeclared_hypotheses=True,descriptive_intervals_not_multiplicity_corrected=True,not_SOTA=True)


'''+auditor[end:]
# Only the score contrast loop expands from three fits to six inference policies.
auditor=change(auditor,"for arm in ARMS:\n        contrasts[arm] = {}", "for arm in CANDIDATES:\n        contrasts[arm] = {}")
auditor=change(auditor,"assert done['registration'] == registration and e['training_protocol'] == registration", "assert done['registration'] == registration and e['training_protocol'] == registration\n    assert s['ordinary_training_enabled'] is False and s['arms']==list(ARMS)\n    profile=read(directory/'actual/engineering/completion.json')\n    assert profile['protocol']==registration and profile['ordinary_calls']==0 and profile['temporary_updates_discarded']==3")
auditor=change(auditor,"assert statistics[-1]['decision'] == done['final_audit']['decision']", "assert statistics[-1]['decision'] == done['final_audit']['decision']\n    composition=verify_composition(directory,stage,queries[:stage])")
auditor=auditor.replace('30_CONTRASTS','42_CONTRASTS').replace('local_pixel_images=42','local_pixel_images=48, composition_interaction=composition')
auditor=change(auditor,"labels = ('target', 'parent', 'ordinary', 'joint', 'EEG only', 'P only', 'swapped', 'absolute only')", "labels = ('target','parent','ordinary','no-O joint','no-O swapped','no-O absolute','joint + O','swapped + O','absolute + O')")
auditor=auditor.replace('(8 * width,','(9 * width,').replace('fixed6_all7_methods.png','fixed6_all8_methods.png')
insert=auditor.index('def audit(')
extra='''def verify_composition(directory,stage,queries):
    values=np.load(directory/f'actual/images/all_query_values{stage}.npz',allow_pickle=False)
    recorded=read(directory/'actual/composition_interaction.json');assert recorded['stage']==stage and recorded['does_not_rescue_image_gate']
    maximum=0.;result={}
    for a in ARMS:
        # Independent grouping/bootstrap implementation uses count weights.
        delta=np.column_stack([(values[a+'_plus_ordinary__'+k]-values['ordinary__'+k])-(values[a+'__'+k]-values['parent__'+k]) for k in KEYS])
        v=contrast(delta,queries);result[a]=v
        for k in KEYS:
            r=recorded['interaction'][a][k]
            maximum=max(maximum,abs(v[k]['difference']-r['difference_in_differences']),float(np.max(np.abs(np.array(v[k]['descriptive_concept_bootstrap95'])-r['descriptive_concept_bootstrap95']))))
    assert maximum<2e-12
    return dict(maximum_error=maximum,all3_policy_interactions=result,original_decisions_unchanged=True)


'''
auditor=auditor[:insert]+extra+auditor[insert:]
paths=[(A/'fetch_HIVE_noordinary_terminal_OLD_CPU_20261101_v1.py',collector),(L/'audit_HIVE_noordinary_terminal_local_20261101_v1.py',auditor)]
for p,s in paths:ast.parse(s);assert not p.exists();p.write_bytes(s.encode())
v=json.loads((L/'audit/HIVE_noordinary_factorial_queue_20261031_v1/actual_launch.json').read_bytes());v['controller_PID']=v['PID']
(A/'HIVE_noordinary_terminal_fetch_request_20261101_v1.json').write_bytes((json.dumps(dict(host='old',upstream_launch=v),indent=2)+'\n').encode())
(O/'source_pins.json').write_bytes((json.dumps([dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,_ in paths],indent=2)+'\n').encode())
print(json.dumps(dict(status='PREPARED_NOO_TERMINAL_READER_ONLY',new_GPU_jobs=0,files=[str(p) for p,_ in paths])))
