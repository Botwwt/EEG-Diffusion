"""One prespecified ordinary-parent contrast after the complete main lifecycle.

No images, new inference, model selection, model imports or repeated score peeks.
Uses the same subject/concept bootstrap and fixed seed as the main comparison.
"""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
STATE = WORK / 'audit/ordinary_parent_compare_CPU_v1.json'
RESULT = WORK / 'comparison/ordinary_parent_supplement_v1.json'
PERIOD = 1200
MAXIMUM = 7 * 86400
RECIPE = ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json'
EXPECTED = '55d92772f3e76240799cf69df12f590229a408c7b7a48050b9a0c7ec01f31db3'
SUCCESS = 'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED'
FAILURES = {'FINITE_MAIN_STORAGE_CONTINUATION_PARTIAL_WITH_EXPLICIT_FAILURES',
    'STORAGE_FORECAST_STOPPED_PRESERVED_NO_CASCADING_JOBS',
    'FAILED_MAIN_STORAGE_CONTINUATION_PRESERVED_NEEDS_DIAGNOSIS'}


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(dict(value, UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def compare(recipe):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # No GPU worker remains in the main lifecycle when this function is admitted.
    assert int(Path('/sys/fs/cgroup/memory.max').read_text()) - int(Path('/sys/fs/cgroup/memory.current').read_text()) >= 512 << 20
    import numpy as np
    keys = ['DINO2AFC', 'SEED', 'ObjectF1', 'DINO_Top1']
    tables = {}; hashes = {}
    for arm in ['ordinary', 'parent']:
        directory = WORK / 'evaluation' / recipe['run_names'][arm] / 'scores'
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['status'] == 'COMPLETED' and summary['queries'] == 16600
        path = directory / 'per_query.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        tables[arm] = {row['query_id']: row for row in rows}
        assert len(rows) == len(tables[arm]) == 16600
        hashes[arm] = dict(path=str(path), sha256=sha(path))
    assert tables['ordinary'].keys() == tables['parent'].keys()
    groups = {}
    for query, row in tables['ordinary'].items():
        base = tables['parent'][query]
        for key in ['subject', 'stimulus_id', 'concept_id', 'image_seed', 'score_target_path', 'EEG_source_stimulus_id']:
            assert row[key] == base[key], (query, key)
        assert row['generated_path'] != base['generated_path']
        groups.setdefault((row['subject'], row['concept_id']), []).append([float(row[k])-float(base[k]) for k in keys])
    subjects = sorted({s for s, _ in groups}); concepts = sorted({c for _, c in groups})
    assert len(subjects) == 10 and len(concepts) == 166 and len(groups) == 1660
    matrix = np.array([[np.mean(groups[(s,c)], axis=0) for c in concepts] for s in subjects])
    assert matrix.shape == (10,166,4) and np.isfinite(matrix).all()
    rng = np.random.default_rng(1729); draws=[]
    for _ in range(1000):
        si=rng.integers(10,size=10);ci=rng.integers(166,size=166)
        draws.append(matrix[si[:,None],ci[None,:]].mean(axis=(0,1)))
    draws=np.array(draws)
    effects={k:dict(mean=float(matrix[:,:,i].mean()),
        CI95_subject_and_concept_bootstrap=np.quantile(draws[:,i],[.025,.975]).tolist(),
        training_seed_variability_covered=False) for i,k in enumerate(keys)}
    d=effects
    gain=(d['DINO2AFC']['mean']>=.01 and d['DINO2AFC']['CI95_subject_and_concept_bootstrap'][0]>0
          and d['SEED']['mean']>0 and d['SEED']['CI95_subject_and_concept_bootstrap'][0]>0
          and d['ObjectF1']['mean']>=0)
    return dict(status='COMPLETE_ORDINARY_MINUS_PARENT_FULL_QUERY_PAIRED_COMPARISON',
        paired_differences=effects, practical_numerical_gain_signal=bool(gain), queries=16600,
        subjects=10, concepts=166, repeat='R4', gallery=1660, fixed_training_seed=1729,
        source_scores=hashes, score_table_read_count_per_arm=1, image_generation_calls=0,
        fitted_parameters=0, checkpoint_selection_performed=False, mechanism_confirmed=False,
        independent_SOTA_established=False,
        interpretation='Risk-gate failure does not reject direction correction; ordinary needs its own paired contrast. V is exposed development evidence.')


def main():
    with (WORK/'audit/ordinary_parent_compare_CPU_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not STATE.exists() and not RESULT.exists()
        started=time.monotonic(); own=sha(__file__)
        while time.monotonic()-started<MAXIMUM:
            assert sha(RECIPE)==EXPECTED and sha(__file__)==own
            state=json.loads((WORK/'audit/main_storage_continuation_20261021_v1.json').read_text())
            if state['status']==SUCCESS:
                result=compare(json.loads(RECIPE.read_text()))
                RESULT.parent.mkdir(parents=True,exist_ok=True);save(RESULT,result)
                save(STATE,dict(status='COMPLETED_ORDINARY_PARENT_CONTRAST_SAVED',pid=os.getpid(),result=str(RESULT)))
                return
            if state['status'] in FAILURES:
                save(STATE,dict(status='NOT_RUN_MAIN_COMPARISON_INCOMPLETE',pid=os.getpid(),main_status=state['status']))
                return
            save(STATE,dict(status='WAITING_WITHOUT_READING_SCORES',pid=os.getpid(),interval_seconds=PERIOD,
                maximum_seconds=MAXIMUM,source_sha256=own,score_read_count=0))
            time.sleep(min(PERIOD,max(0,MAXIMUM-(time.monotonic()-started))))
        save(STATE,dict(status='STOPPED_FINITE_SEVEN_DAY_BOUND',pid=os.getpid(),score_read_count=0))


if __name__=='__main__':
    main()
