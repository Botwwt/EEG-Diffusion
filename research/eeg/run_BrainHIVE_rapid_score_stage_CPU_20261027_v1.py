"""Bridge one locked scorer action into status/bool guards for the finite runner."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path

import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as D

SCORER_SHA='bc9610b97501cdf078789f913b0520b023c0079801e24c37093cbac406189009'
MAPPING={'EXPAND400':('80','RAPID_STAGE80_EXPAND400'),
         'STOP_INSTANCE':(None,'RAPID_SCREEN_STOP_INSTANCE'),
         'PROMISING_FOR_FULL_EVALUATION':('400','RAPID_STAGE400_PROMISING')}


def make_guard(summary,arm,stage,output):
    assert D.sha(D.__file__)==SCORER_SHA
    assert summary['schema']==D.SCHEMA and summary['status']=='COMPLETE_CPU_FIXED400_GALLERY_PAIRED_SCREEN'
    assert stage in ('80','400') and summary['stage']==stage
    count=int(stage)
    assert summary['queries']==count and summary['concepts']==count//10 and summary['gallery_targets']==400
    assert summary['rows']==2*count and summary['hidden_exclusions']==0
    assert summary['full_registered_stage_denominator'] is True and summary['CPU_only'] is True and summary['new_GPU_calls']==0
    assert summary['independent_confirmation'] is False and summary['SOTA_established'] is False
    assert set(summary['candidates'])=={arm} and arm not in ('native_parent','target_gallery')
    assert summary['source']['sha256']==SCORER_SHA and D.checked_ref(summary['source']).resolve()==Path(D.__file__).resolve()
    decision=summary['candidates'][arm]['decision']
    assert decision==D.decision(summary['candidates'][arm]['paired_candidate_minus_native'],stage),'Action differs from the frozen scorer semantics'
    required_stage,status=MAPPING[decision['action']]
    assert required_stage is None or stage==required_stage
    panel=D.load_panel(D.checked_ref(summary['panel']))
    expected={query['query_id'] for query in panel['queries'][:count]}
    with D.checked_ref(summary['per_query']).open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==2*count and Counter(row['arm'] for row in rows)==Counter({'native_parent':count,arm:count})
    assert Counter((row['query_id'],row['arm']) for row in rows)==Counter((query,side) for query in expected for side in ('native_parent',arm))
    summary_path=Path(output)/'summary.json'
    assert D.digest(D.read(summary_path))==D.digest(summary),'Written scorer summary changed'
    return dict(schema='BrainHIVE_rapid_score_stage_guard_20261027_v1',status=status,arm=arm,stage=stage,
        decision=decision,source_summary=D.ref(summary_path),source_scorer=D.ref(D.__file__),source_bridge=D.ref(__file__),
        expand400=decision['action']=='EXPAND400',full_registered_stage_denominator=True,queries=count,gallery_targets=400,
        GPU_used=False,independent_confirmation=False,thresholds_or_decision_policy_changed=False)


def run(panel,target_cache,native_cache,candidate_caches,stage,output,stage1_summary=None):
    assert len(candidate_caches)==1,'One fixed candidate arm per CPU stage'
    assert D.sha(D.__file__)==SCORER_SHA
    arm=D.read(candidate_caches[0])['arm'];assert isinstance(arm,str) and arm and arm not in ('native_parent','target_gallery')
    summary=D.score(panel,target_cache,native_cache,candidate_caches,stage,output,stage1_summary)
    guard=make_guard(summary,arm,stage,output)
    path=Path(output)/'round_guard.json';assert not path.exists()
    D.atomic(path,guard)
    return guard


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('panel','target-cache','native-cache','output','stage1-summary'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--candidate-caches',nargs='+',type=Path);parser.add_argument('--stage',choices=('80','400'))
    args=parser.parse_args(argv)
    assert all((args.panel,args.target_cache,args.native_cache,args.output,args.candidate_caches,args.stage))
    result=run(args.panel,args.target_cache,args.native_cache,args.candidate_caches,args.stage,args.output,args.stage1_summary)
    print(json.dumps(result,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
