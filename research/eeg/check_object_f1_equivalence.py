"""Actual detector JSONs: streamed pair algebra must equal pinned SEED source."""
from image_evaluation import *
import json
from seed.metrics import eval_object_f1

ROOT_PROFILE=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runs/S0_semantic_evaluator')


def main():
    records={}
    for arm in ['self','oracle','shuffle']:
        generated=sorted((ROOT_PROFILE/arm/'preds').glob('*.json'))
        target=sorted((ROOT_PROFILE/'gt/preds').glob('*.json'))
        ours=np.array([object_f1_pair(json.loads(a.read_text()),json.loads(b.read_text())) for a,b in zip(generated,target)])
        official=eval_object_f1(str(ROOT_PROFILE/arm),str(ROOT_PROFILE/'gt'))
        assert np.array_equal(ours,official),(ours,official)
        records[arm]=dict(queries=len(ours),max_abs_error=float(abs(ours-official).max()))
    report=dict(status='EXACT_EQUAL_TO_PINNED_OFFICIAL_FUNCTION',cases=records,
        official_source_commit='5327a134b90b04fb1b37066d98b309c9cc533a02',
        official_function='seed/metrics.py:eval_object_f1',threshold_grid='np.arange(.05,.95,.05)',
        empty_detection_convention_preserved=True,metric_changed_for_scores=False)
    (ROOT/'audit/object_f1_streaming_equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
