"""All seven author metrics against four paper configurations, descriptive only.

This reuses the immutable completed ordinary R80 readout. Published rounded
means cannot provide paired intervals or prove SOTA. An envelope is not a model.
"""
import csv, datetime, hashlib, json, math
from pathlib import Path

L = Path(__file__).resolve().parent
D = L / 'audit/HIVE_ordinary_R80_complete_20261029_v1'
KEYS = ('eval_pixcorr', 'eval_ssim', 'eval_alex2', 'eval_alex5',
        'eval_inception', 'eval_clip', 'eval_swav')
NAMES = ('PixCorr', 'SSIM', 'AlexNet(2)', 'AlexNet(5)', 'Inception', 'CLIP', 'SwAV')

def read(path):
    return json.loads(Path(path).read_bytes())

def ref(path):
    path = Path(path)
    return dict(path=path.relative_to(L).as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    receipt = read(D / 'receiving_completion.json')
    pins = {x['name']:x for x in receipt['entries']}
    required = ('candidate/paper.csv', 'baseline/paper.csv', 'metadata/paired_summary.json')
    for name in required:
        data = (D / name).read_bytes()
        assert len(data) == pins[name]['bytes'] and hashlib.sha256(data).hexdigest() == pins[name]['sha256']
    summary = read(D / 'metadata/paired_summary.json')
    assert summary['queries'] == 2000 and summary['subjects'] == 10
    means, identities = {}, {}
    for arm, name in (('ordinary', 'candidate/paper.csv'), ('parent', 'baseline/paper.csv')):
        with (D / name).open(newline='', encoding='utf8') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len({r['query_id'] for r in rows}) == 2000
        subjects = sorted({r['subject'] for r in rows})
        assert subjects == [f'sub-{i:02d}' for i in range(1, 11)]
        assert all(sum(r['subject']==sub for r in rows) == 200 for sub in subjects)
        identities[arm] = {r['query_id']:(r['subject'], r['stimulus_id'], r['concept_id'], r['image_seed'], r['score_target_path']) for r in rows}
        means[arm] = {}
        for key in KEYS:
            values = [float(r[key]) for r in rows]
            assert all(math.isfinite(v) for v in values)
            value = math.fsum(values) / 2000
            assert abs(value - summary['results'][key][arm+'_mean']) < 1e-12
            means[arm][key] = value
    assert identities['ordinary'] == identities['parent']
    # Independently re-read from primary arXiv2602.07495v1 Tables10--13.
    # Rows are retained separately: their metricwise envelope is not a baseline.
    published = {
        'H14_Table10': [.174, .327, .825, .871, .733, .773, .574],
        'H14_B32_Table11': [.187, .339, .836, .908, .783, .814, .547],
        'H14_VAE_Table12': [.173, .312, .789, .838, .672, .721, .611],
        'H14_B32_VAE_Table13_and_Table2_EEG_Ours': [.195, .336, .843, .905, .756, .808, .554],
    }
    original = read(L / 'audit/BrainHIVE_paper_reference_R80_descriptive_actual_20261021_v1.json')
    rows = original['comparison_rows']
    assert [r['source_csv_column'] for r in rows] == list(KEYS)
    assert [r['paper_Table2_EEG_Ours_and_Table13_Ave'] for r in rows] == published['H14_B32_VAE_Table13_and_Table2_EEG_Ours']
    comparisons = {}
    for configuration, numbers in published.items():
        comparison = []
        for key, label, paper in zip(KEYS, NAMES, numbers):
            own = means['ordinary'][key]
            sign = -1 if key == 'eval_swav' else 1
            delta = own - paper
            oriented = sign * delta
            comparison.append(dict(metric=label, column=key,
                direction='lower' if sign < 0 else 'higher',
                published_rounded_mean=paper, original_parent_mean=means['parent'][key],
                ordinary_mean=own, ordinary_minus_published=delta,
                favorable_direction_difference=oriented,
                absolute_gap_within_half_last_published_digit=abs(delta) <= .0005,
                rounding_sensitivity_if_nearest_rounding=[oriented-.0005, oriented+.0005],
                rounding_sensitivity_is_NOT_a_confidence_interval=True))
        comparisons[configuration] = comparison
    envelope = []
    for j, key in enumerate(KEYS):
        best = min(x[j] for x in published.values()) if key=='eval_swav' else max(x[j] for x in published.values())
        owners = [k for k, v in published.items() if v[j] == best]
        envelope.append(dict(metric=NAMES[j], best_published_configuration_mean=best,
            configurations=owners, ordinary_mean=means['ordinary'][key],
            favorable_direction_difference=(means['ordinary'][key]-best)*(-1 if key=='eval_swav' else 1),
            metricwise_envelope_is_NOT_a_real_model=True))
    result = dict(status='COMPLETE_SINGLE_SEED_ORDINARY_SEVEN_AUTHOR_METRIC_PAPER_CONFIGURATION_REFERENCE_DESCRIPTIVE_ONLY',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        primary_source='https://arxiv.org/html/2602.07495v1',
        primary_tables=['10 H14','11 H14+B32','12 H14+VAE','13 H14+B32+VAE','2 EEG reconstruction'],
        primary_tables_reread=True, sources=[ref(Path(__file__)), ref(L / 'audit/BrainHIVE_paper_reference_R80_descriptive_actual_20261021_v1.json')] + [ref(D / x) for x in required],
        queries=2000, subjects=10, training_seed=1729, independent_original_csv_and_summary_means_agree=True,
        comparisons=comparisons, four_configuration_metricwise_envelope=envelope,
        another_Table2_SSIM_reference=dict(method='CognitionCapturer_Depth', published_SSIM=.370,
            ordinary_SSIM=means['ordinary']['eval_ssim'], table_only_not_reproduced=True),
        protocol_limits=['Author averages10reconstruction sets; ours1image/query, notbestofN.',
            'Published rounded means lack paired query data; no cross-paper confidence or significance claim.',
            'The four configuration envelope is a descriptive target, not an implemented ensemble.',
            'Table2SSIM has another higher comparator; no blanket all-metric superiority.',
            'Prior pretraining/extra-data/software/protocol differences remain; this is not a current all-method SOTA audit.',
            'Original ordinary remains T1323-trained; full1654component fit/independentH/crosshost remain incomplete.'],
        do_not_change_current_paired_semantic_study_from_these_exposed_results=True,
        new_training_GPU_images_or_candidate_results=0,
        author_raw_query_data_available=False, independent_H=False, SOTA=False)
    out = L / 'audit/HIVE_ordinary_paper_configurations_reference_20261030_v1'
    out.mkdir(exist_ok=False)
    (out / 'actual_full2000_seven_metric_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(dict(status=result['status'], queries=2000, comparisons=28,
        ordinary_means=means['ordinary'], metricwise_four_configuration_envelope=envelope,
        new_GPU_jobs=0, SOTA=False)))

if __name__ == '__main__':
    main()
