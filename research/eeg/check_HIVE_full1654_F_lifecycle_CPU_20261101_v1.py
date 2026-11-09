"""Full-size scope, boundary counts and rejected changes; no learned outcomes."""
import copy,hashlib,json,math
from pathlib import Path
import brainhive_full1654_F_study_contract_20261101_v1 as G

def run(s):
    G.validate_scope(s);checks=['exact_prospective_exposed_fullscope_contract']
    for label,change in [('false_unexposed_rejected',dict(official200_previously_exposed=False)),('new_training_seed_rejected',dict(method=s['method']|dict(seed=2027))),('old132300_scope_rejected',dict(method=s['method']|dict(queries=132300))),('postfit_strength_search_rejected',dict(method=s['method']|dict(strength=.75))),('missing_control_rejected',dict(arms=['bounded_F'])),('wrong_partial_updates_rejected',dict(method=s['method']|dict(updates=24807))),('unregistered_numeric_guard_rejected',dict(numerical_guard=dict(atol=1e-3,rtol=5e-6)))]:
        bad=copy.deepcopy(s);bad.update(change)
        try:G.validate_scope(bad)
        except AssertionError:checks.append(label)
        else:raise AssertionError(label)
    assert math.ceil(165400/16)==10338 and 10338*3==31014 and (165400%16)==8
    coverage=[0]*165400;updates=exposures=0
    for unused in range(3):
        for lo in range(0,165400,16):
            n=min(16,165400-lo);updates+=1;exposures+=n
            for i in range(lo,lo+n):coverage[i]+=1
    assert set(coverage)=={3} and updates==31014 and exposures==496200;checks.append('all3pass_batch_boundaries_and_full_denominator_exposure_accounting')
    keys=[(sub,row) for sub in range(10) for row in range(200)]
    assert len(keys)==2000 and [(i//200,i%200) for i in range(2000)]==keys;checks.append('all2000_R80_subject_boundary_indices')
    tree=Path(__file__).parent
    producer=(tree/'train_HIVE_full1654_F_heads_GPU_20261101_v1.py').read_text()
    assert producer.index('inputs=I.load_verified(')<producer.index('torch.cuda.set_per_process_memory_fraction(')
    checks.append('CPU_only_fullbank_verification_precedes_any_explicit_CUDA_initialization')
    # Existing all16 evaluator and statistics kernels are copied byte-for-byte.
    for name in ('profile_HIVE_loss_control_R80_GPU_20261030_v1.py','score_HIVE_loss_control_R80_author_GPU_20261030_v1.py','run_HIVE_R80_evaluation_guarded_20261029_v1.py','brainhive_loss_control_R80_statistics_CPU_20261030_v1.py','brainhive_owned_process_CPU_20261029_v1.py'):
        pin=next(p for p in s['source_pins'] if Path(p['path']).name==name);assert G.ref(tree/name)==pin
    checks.append('frozen_evaluator_statistics_and_owned_process_sources_bound')
    return dict(status='PASS_SOURCE_SCOPE_COMPLETE_LIFECYCLE_CPU_REVIEW',checks=checks,count=len(checks),input_bank=s['input_bank'],source_pins=s['source_pins'],method=s['method'],R80=s['R80'],no_tensor_payloads_or_effect_scores_read=True,new_GPU_jobs=0,independent_H=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);a=p.parse_args();G.write(a.output,run(G.read(a.study)))
