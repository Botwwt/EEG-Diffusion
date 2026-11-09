"""Historical-feature and adversarial archive checks, never new candidate results."""
import base64, copy, hashlib, importlib.util, io, json, stat, tempfile, zipfile
from pathlib import Path
import numpy as np
import audit_HIVE_paired_semantic_terminal_local_20261030_v1 as V
import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S

L = Path(__file__).resolve().parent
A = L / 'audit/crossdisciplinary_20261028_v1'

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

def rejected(call):
    try:
        call()
    except (AssertionError, ValueError, KeyError):
        return
    raise AssertionError('Unsafe or invalid input unexpectedly accepted')

def archive_fixture(name='protocol/test.json', *, duplicate=False, symlink=False):
    data = b'{"fixture_only":true}\n'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        item = zipfile.ZipInfo(name)
        if symlink:
            item.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(item, data)
        if duplicate:
            archive.writestr(item, data)
    raw = buffer.getvalue()
    return dict(status='COMPLETE_CORE_TERMINAL_EVIDENCE_READY', exact_main_live=False,
        stage=80, manifest={name:dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())},
        archive_base64=base64.b64encode(raw).decode(), archive_SHA256=hashlib.sha256(raw).hexdigest())

def main():
    checks = []
    receiver = module('paired_receiver_check', A / 'receive_HIVE_paired_semantic_terminal_local_20261030_v1.py')
    fetcher = module('paired_fetch_check', A / 'fetch_HIVE_paired_semantic_terminal_OLD_CPU_20261030_v1.py')
    fixture = archive_fixture()
    assert receiver.validated_archive(fixture)
    checks.append('bounded SHA-valid ZIP accepted as fixture only, not full study closure')
    live = copy.deepcopy(fixture); live['exact_main_live'] = True
    rejected(lambda: receiver.validated_archive(live)); checks.append('live outcome archive rejected')
    for name in ('../escape.json', '/absolute.json', 'C:/escape.json', 'protocol\\escape.json', 'protocol/./escape.json'):
        rejected(lambda name=name: receiver.validated_archive(archive_fixture(name)))
    checks.append('parent, absolute, Windows drive, backslash and noncanonical archive paths rejected')
    rejected(lambda: receiver.validated_archive(archive_fixture(symlink=True)))
    checks.append('archive symlink rejected')
    rejected(lambda: receiver.validated_archive(archive_fixture(duplicate=True)))
    checks.append('duplicate ZIP members rejected')
    tampered = copy.deepcopy(fixture)
    tampered['manifest']['protocol/test.json']['sha256'] = '0' * 64
    rejected(lambda: receiver.validated_archive(tampered)); checks.append('manifest file SHA corruption rejected')
    fetcher.checked = lambda x: None
    fetcher.live = lambda pid, ticks: True
    def forbidden_read(path):
        raise AssertionError('A live collector attempted to read an outcome')
    fetcher.read = forbidden_read
    wait = fetcher.collect(dict(host='old', upstream_launch=dict(protocol={}, controller_PID=123, start_ticks=456)))
    assert wait['result_files_read'] == 0 and wait['exact_main_live']
    checks.append('live terminal gate returns before any result JSON read')

    q = [dict(stimulus_id=f'fixture_{i}', concept_id=f'concept_{i//10}') for i in range(400)]
    eye = np.eye(400, 768)
    one = V.metrics(eye, eye, q, q)
    assert np.array_equal(one[:, 1:], np.ones((400, 4)))
    checks.append('exact target one-hot self pairing has full scores')
    ties = V.metrics(np.ones((400, 768)), eye, q, q)
    assert np.array_equal(ties[:, (1, 4)], np.full((400, 2), .5))
    assert ties[:, 2].sum() == 1 and ties[:, 3].sum() == 40
    checks.append('all-tie fixture gives half 2AFC and first-index Top1')
    rejected(lambda: V.metrics(np.zeros((400, 768)), eye, q, q))
    checks.append('zero-norm feature bank rejected')
    shuffled = np.arange(400)[::-1]
    permuted = V.metrics(eye, eye[shuffled], q, [q[i] for i in shuffled])
    assert np.array_equal(permuted, one)
    checks.append('gallery identity lookup is independent of its row order')

    old = L / 'audit/HIVE_common_continuation_complete_20261030_v1'
    queries = V.read(old / 'reference/panel.json')['Dcheck']
    target = np.load(old / 'reference/D09_targets.npy', allow_pickle=False)
    prior = np.load(old / 'reference/D09_old3.npy', allow_pickle=False)
    mapping = dict(parent='parent', ordinary='ordinary', joint_correct='joint_same',
        EEG_only_correct='relative_only', P_only_correct='AP_only',
        joint_swapped_targets='joint_mismatched_teacher', joint_absolute_only='starting_F')
    features = dict(parent=prior[:, 0], ordinary=prior[:, 1],
                    joint_absolute_only=np.load(old / 'reference/starting_F.npy', allow_pickle=False))
    for arm in ('joint_correct', 'EEG_only_correct', 'P_only_correct', 'joint_swapped_targets'):
        features[arm] = np.load(old / 'actual/images' / f'features400_{mapping[arm]}.npy', allow_pickle=False)
    errors, cells, comparisons = [], 0, 0
    for n in (80, 400):
        raw = np.load(old / 'actual/images' / f'all_query_values{n}.npz', allow_pickle=False)
        score = V.read(old / 'actual/images' / f'score{n}.json')
        measured = {}
        for arm in V.METHODS:
            values = V.metrics(features[arm][:n], target, queries[:n], queries)
            measured[arm] = values
            for j, key in enumerate(V.KEYS):
                errors.append(float(np.max(np.abs(values[:, j] - raw[mapping[arm] + '__' + key]))))
                errors.append(abs(float(values[:, j].mean()) - score['means'][mapping[arm]][key])); cells += n
        for arm in V.ARMS:
            for baseline in V.METHODS:
                if arm == baseline:
                    continue
                delta = measured[arm] - measured[baseline]
                actual = V.contrast(delta, queries[:n])
                # The locked production indexed-bootstrap independently evaluates
                # all 30 aliases; the aliases are HISTORICAL methods, not new fits.
                left = {key:measured[arm][:, j] for j, key in enumerate(V.KEYS)}
                right = {key:measured[baseline][:, j] for j, key in enumerate(V.KEYS)}
                expected = S.compare(left, right, queries[:n])
                for j, key in enumerate(V.KEYS):
                    errors.append(abs(actual[key]['difference'] - expected[key]['difference']))
                    errors.extend(np.abs(np.array(actual[key]['descriptive_concept_bootstrap95']) - expected[key]['descriptive_concept_bootstrap95']).tolist())
                    if arm != 'joint_absolute_only':
                        production = score['contrasts'][mapping[arm]][mapping[baseline]][key]
                        errors.append(abs(actual[key]['difference'] - production['difference']))
                        errors.extend(np.abs(np.array(actual[key]['descriptive_concept_bootstrap95']) - production['descriptive_concept_bootstrap95']).tolist())
                        errors.append(float(np.max(np.abs(delta[:, j] - raw[mapping[arm] + '_minus_' + mapping[baseline] + '__' + key])))); cells += n
                comparisons += 1
    assert max(errors) < 2e-12
    checks.append('historical real80/400 ranks/raw cells and all30 weighted/indexed bootstrap aliases match')
    contract = module('paired_original_decision_check', L / 'audit/HIVE_paired_semantic_full_study_registration_20261030_v1/brainhive_paired_semantic_study_contract_CPU_20261030_v1.py')
    fixture_parent = L / 'audit/HIVE_paired_semantic_terminal_reader_preparation_20261030_v1'
    fixture_parent.mkdir(exist_ok=True)
    fixture_reports = []
    with tempfile.TemporaryDirectory(prefix='historical_schema_fixture_', dir=fixture_parent) as name:
        temporary = Path(name).resolve()
        assert temporary.is_relative_to(fixture_parent.resolve())
        folder = temporary / 'actual/images'; folder.mkdir(parents=True)
        for n in (80, 400):
            production = {arm:S.metric_arrays(features[arm][:n].astype(np.float64), target.astype(np.float64), queries[:n], queries) for arm in V.METHODS}
            contrasts = {arm:{base:S.compare(production[arm], production[base], queries[:n]) for base in V.METHODS if base != arm} for arm in V.ARMS}
            raw = {arm+'__'+key:value for arm, fields in production.items() for key, value in fields.items()}
            raw.update({arm+'_minus_'+base+'__'+key:production[arm][key]-production[base][key] for arm in V.ARMS for base in V.METHODS if arm != base for key in V.KEYS})
            np.savez_compressed(folder / f'all_query_values{n}.npz', **raw)
            for arm in V.METHODS:
                np.save(folder / f'features{n}_{arm}.npy', features[arm][:n], allow_pickle=False)
            score = dict(means={arm:{key:float(value.mean()) for key, value in fields.items()} for arm, fields in production.items()}, contrasts=contrasts, decision=contract.image_decision(contrasts, n))
            (folder / f'score{n}.json').write_text(json.dumps(score), encoding='utf8')
            fixture_reports.append(V.verify_statistics(temporary, n, queries, target))
        # TemporaryDirectory cleanup is confined to the checked absolute fixture path.
        assert temporary.is_relative_to(fixture_parent.resolve())
    checks.append('full seven-method80/400 file schema independently replayed using historical aliases only')
    # Strict unchanged core policy, tested at meaningful boundaries.
    def contrasts_fixture(difference, lower, upper, detail=0.):
        fields = {key:dict(difference=detail if key=='within_concept10_top1' else difference,
                          descriptive_concept_bootstrap95=[lower, upper]) for key in V.KEYS}
        return {'joint_correct':{b:copy.deepcopy(fields) for b in V.METHODS if b != 'joint_correct'}}
    assert V.decision(contrasts_fixture(-.02, -.03, -.001), 80)['action'] == 'STOP_INSTANCE'
    assert V.decision(contrasts_fixture(-.02, -.03, 0.), 80)['action'] == 'EXPAND_ALL_TO400'
    assert V.decision(contrasts_fixture(.01, .001, .02), 400)['action'] == 'SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION'
    assert V.decision(contrasts_fixture(.009999, .001, .02), 400)['action'] == 'STOP_INSTANCE'
    assert V.decision(contrasts_fixture(.02, .001, .03, detail=-.01), 400)['action'] == 'STOP_INSTANCE'
    checks.append('unchanged 80/400 practical, strict interval and detail gate boundaries')
    result = dict(status='PASS_CPU_TERMINAL_READER_AND_HISTORICAL_ALIAS_CHECKS', checks=checks,
        historical_actual_metric_and_contrast_cells=cells,
        historical_alias_stage_contrasts=comparisons, maximum_historical_error=max(errors),
        historical_full_schema_replay=[{k:v for k,v in r.items() if k not in ('means','decision')} for r in fixture_reports],
        new_candidate_results_read=False, actual_new_terminal_reception_NOT_executed=True,
        actual_full_new_terminal_audit_and_panel_NOT_executed=True, new_GPU_jobs=0)
    out = L / 'audit/HIVE_paired_semantic_terminal_reader_preparation_20261030_v1'
    out.mkdir(exist_ok=True)
    (out / 'actual_local_CPU_checks.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
