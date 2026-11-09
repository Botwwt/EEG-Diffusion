"""Meaningful decision-boundary tests; synthetic cases are not image results."""
import copy,hashlib,json
from pathlib import Path
from interpret_HIVE_norm_only_complete_CPU_20261030_v1 import interpret,PAIRS,MECHANISM,PERFORMANCE
L=Path(__file__).resolve().parent
study=json.loads((L/'audit/HIVE_norm_only_R80_registration_20261030_v1/protocol/study.json').read_bytes());assert study['mechanism_support']==MECHANISM and study['performance_progress_gate']==PERFORMANCE
def metric(value,lo,hi):return dict(status='COMPLETE',defined=2000,paired_left_minus_right=value,descriptive_subject_concept_bootstrap95=[lo,hi])
base={p:dict(DINO2AFC=metric(.02,.01,.03),SEED=metric(.01,.001,.02),ObjectF1=metric(0.,-.01,.01)) for p in PAIRS}
checks=[]
assert interpret(base)['performance_progress_gate_passed'] is True;checks.append('both strong baselines required; zero ObjectF1 mean allowed')
v=copy.deepcopy(base);v[PAIRS[2]]['DINO2AFC']=metric(.009875,.001,.02);r=interpret(v);assert r['mechanism_support'] is True and r['performance_progress_gate_passed'] is False;checks.append('real historical .009875 must not round to .01')
v=copy.deepcopy(base);v[PAIRS[1]]['DINO2AFC']=metric(.009999999999,.001,.02);assert interpret(v)['performance_progress_gate_passed'] is False;checks.append('arbitrarily close below threshold still fails')
v=copy.deepcopy(base);v[PAIRS[1]]['DINO2AFC']=metric(.01,0.,.02);assert interpret(v)['performance_progress_gate_passed'] is False;checks.append('CI lower zero is not positive')
v=copy.deepcopy(base);v[PAIRS[2]]['ObjectF1']=metric(-1e-8,-.01,.01);assert interpret(v)['performance_progress_gate_passed'] is False;checks.append('negative ObjectF1 mean is not rescued by interval crossing')
v=copy.deepcopy(base);v[PAIRS[2]]['SEED']=dict(status='UNKNOWN_UNDEFINED_FULL_DENOMINATOR_RETAINED',expected=2000);assert interpret(v)['performance_progress_gate_passed'] is None;checks.append('undefined primary preserved UNKNOWN')
v=copy.deepcopy(base);v[PAIRS[2]]['DINO2AFC']=metric(-.02,-.03,-.01);r=interpret(v);assert r['mechanism_support'] is True and r['performance_progress_gate_passed'] is False and r['clear_DINO_harm_versus_ordinary'];checks.append('repairing weak Euclidean control can coexist with clear harm versus ordinary')
v={p:{k:metric(0.,0.,0.) for k in ['DINO2AFC','SEED','ObjectF1']} for p in PAIRS};r=interpret(v);assert r['mechanism_support'] is False and r['performance_progress_gate_passed'] is False;checks.append('zero effect is not support or progress')
out=L/'audit/HIVE_norm_only_registered_interpretation_CPU_20261030_v1';assert not out.exists();out.mkdir()
result=dict(status='PASS_REGISTERED_UNCHANGED_GATE_BOUNDARIES_SYNTHETIC_ONLY',checks=checks,count=len(checks),registered_study_SHA256=hashlib.sha256((L/'audit/HIVE_norm_only_R80_registration_20261030_v1/protocol/study.json').read_bytes()).hexdigest(),interpreter_SHA256=hashlib.sha256((L/'interpret_HIVE_norm_only_complete_CPU_20261030_v1.py').read_bytes()).hexdigest(),new_norm_image_effect_values_read=False,thresholds_or_active_queue_sources_changed=False,SOTA_established=False)
(out/'completion.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result,indent=2))
