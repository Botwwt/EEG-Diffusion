"""Independent local ranks, raw cells, weighted bootstrap, provenance and fixed panels.

No local training, vision-model re-encoding, complete-state/gradient replay, or
all-image pixel verification is claimed. The latter full checks stay on OLD.
"""
import argparse, datetime, hashlib, json, os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
from pathlib import Path, PurePosixPath
import numpy as np

ARMS = ('joint_correct', 'joint_swapped_targets', 'joint_absolute_only')
CANDIDATES = ARMS + tuple(a + '_plus_ordinary' for a in ARMS)
METHODS = ('parent', 'ordinary') + CANDIDATES
KEYS = ('target_cosine', 'DINO2AFC', 'gallery_top1',
        'within_concept10_top1', 'within_concept10_two_afc')

def read(path):
    return json.loads(Path(path).read_bytes())

def metrics(features, targets, queries, gallery):
    f = np.asarray(features, dtype=np.float64)
    v = np.asarray(targets, dtype=np.float64)
    assert f.shape == (len(queries), 768) and v.shape == (400, 768)
    assert np.isfinite(f).all() and np.isfinite(v).all()
    assert (np.linalg.norm(f, axis=1) > 0).all() and (np.linalg.norm(v, axis=1) > 0).all()
    f = f / np.linalg.norm(f, axis=1)[:, None]
    v = v / np.linalg.norm(v, axis=1)[:, None]
    rows = {q['stimulus_id']: i for i, q in enumerate(gallery)}
    assert len(rows) == 400 and len(gallery) == 400
    similarity = f @ v.T
    out = np.empty((len(queries), 5))
    for i, q in enumerate(queries):
        j = rows[q['stimulus_id']]
        score = similarity[i]
        correct = score[j]
        ids = [k for k, g in enumerate(gallery) if g['concept_id'] == q['concept_id']]
        assert len(ids) == 10 and j in ids
        # Subtract the self tie explicitly; every other tie receives half credit.
        out[i] = (correct,
            (np.count_nonzero(score < correct) + .5 * (np.count_nonzero(score == correct) - 1)) / 399,
            int(np.argmax(score) == j),
            int(ids[int(np.argmax(score[ids]))] == j),
            (np.count_nonzero(score[ids] < correct) + .5 * (np.count_nonzero(score[ids] == correct) - 1)) / 9)
    return out

def contrast(delta, queries):
    groups = [[i for i, q in enumerate(queries) if q['concept_id'] == c]
              for c in sorted({q['concept_id'] for q in queries})]
    means = np.stack([np.sum(delta[g], axis=0) / len(g) for g in groups])
    draws = np.random.default_rng(1729).integers(0, len(groups), (1000, len(groups)))
    # Independent implementation: weighted counts rather than indexed means.
    counts = np.stack([np.bincount(row, minlength=len(groups)) for row in draws]) / len(groups)
    intervals = np.quantile(counts @ means, [.025, .975], axis=0)
    return {key: dict(difference=float(delta[:, j].mean()),
                     descriptive_concept_bootstrap95=intervals[:, j].tolist())
            for j, key in enumerate(KEYS)}

def decision(contrasts, stage):
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


def verify_statistics(directory, stage, queries, targets):
    folder = directory / 'actual/images'
    score = read(folder / f'score{stage}.json')
    raw = np.load(folder / f'all_query_values{stage}.npz', allow_pickle=False)
    measured = {arm: metrics(np.load(folder / f'features{stage}_{arm}.npy', allow_pickle=False),
                            targets, queries[:stage], queries) for arm in METHODS}
    errors, cells, contrasts = [], 0, {}
    expected_keys = set()
    for arm, values in measured.items():
        for j, key in enumerate(KEYS):
            name = arm + '__' + key
            expected_keys.add(name)
            assert raw[name].shape == (stage,) and np.isfinite(raw[name]).all()
            errors.extend([float(np.abs(values[:, j] - raw[name]).max()),
                           abs(float(values[:, j].mean()) - score['means'][arm][key])])
            cells += stage
    for arm in CANDIDATES:
        contrasts[arm] = {}
        for baseline in METHODS:
            if arm == baseline:
                continue
            delta = measured[arm] - measured[baseline]
            values = contrast(delta, queries[:stage])
            contrasts[arm][baseline] = values
            for j, key in enumerate(KEYS):
                name = arm + '_minus_' + baseline + '__' + key
                expected_keys.add(name)
                errors.append(float(np.abs(delta[:, j] - raw[name]).max()))
                a = values[key]; b = score['contrasts'][arm][baseline][key]
                errors.append(abs(a['difference'] - b['difference']))
                errors.extend(np.abs(np.asarray(a['descriptive_concept_bootstrap95']) - b['descriptive_concept_bootstrap95']).tolist())
                cells += stage
    assert set(raw.files) == expected_keys and max(errors) < 2e-12
    assert decision(contrasts, stage) == score['decision']
    return dict(stage=stage, raw_metric_and_contrast_cells=cells,
                statistic_checks=len(errors), maximum_error=max(errors),
                means=score['means'], decision=score['decision'])

def verify_composition(directory,stage,queries):
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


def audit(directory):
    from PIL import Image
    directory = Path(directory)
    manifest = read(directory / 'received_manifest.json')
    assert manifest['status'] == 'COMPLETE_CORE_TERMINAL_EVIDENCE_READY' and not manifest['exact_main_live']
    for name, item in manifest['manifest'].items():
        raw = (directory / name).read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    def original_ref(name):
        item = manifest['manifest'][name]
        return dict(path=item['remote_path'], bytes=item['bytes'], sha256=item['sha256'])
    s = read(directory / 'protocol/study.json')
    e = read(directory / 'protocol/image_execution.json')
    registration = original_ref('protocol/study.json')
    execution = original_ref('protocol/image_execution.json')
    done = read(directory / 'actual/controller_completion.json')
    assert done['registration'] == registration and e['training_protocol'] == registration
    assert s['ordinary_training_enabled'] is False and s['arms']==list(ARMS)
    profile=read(directory/'actual/engineering/completion.json')
    assert profile['protocol']==registration and profile['ordinary_calls']==0 and profile['temporary_updates_discarded']==3
    fit = read(directory / 'actual/independent_fit_CPU_audit.json')
    assert fit['registration'] == registration and fit['all3_fixed_finals_verified']
    order = np.load(directory / 'reference/training_order.npy', allow_pickle=False)
    assert order.shape == (6615,) and np.array_equal(np.sort(order), np.arange(6615))
    pairs = read(directory / 'reference/fixed_T_pairs.json')
    assert len(pairs) == 6615 and [p['pair_index'] for p in pairs] == list(range(6615))
    assert sorted(i for p in pairs for i in p['T_indices']) == list(range(13230))
    assert len({p['concept_id'] for p in pairs}) == 1323
    checkpoints = {}
    for arm in ARMS:
        previous, end, count = None, 0, 0
        for path in sorted((directory / 'actual' / arm / 'checkpoints').glob('step-*/commit.json')):
            if '.pending.' in path.parent.name:
                continue
            c = read(path)
            assert c['arm'] == arm and c['registration'] == registration
            assert c['start_update'] == end and c['end_update'] > end
            assert c['previous_commit'] == previous and c['primary_EEG_exposures'] == 2 * c['end_update']
            end = c['end_update']; count += 1
            previous = original_ref(path.relative_to(directory).as_posix())
        completed = read(directory / 'actual' / arm / 'completion.json')
        assert end == completed['updates'] == 6615 and completed['registration'] == registration
        assert completed['model'] == fit['arms'][arm]['model']
        checkpoints[arm] = dict(commits=count, final_update=end, exposed_EEG=2 * end)
    queries = read(directory / 'reference/panel.json')['Dcheck']
    assert len(queries) == 400 and len({q['query_id'] for q in queries}) == 400
    assert all(q['subject_id'] == 'sub-01' and q['partition'] == 'Dcheck' for q in queries)
    export = read(directory / 'actual/images/export_completion.json')
    target = np.load(directory / 'reference/target_DINO.npy', allow_pickle=False)
    stage = manifest['stage']; statistics = []
    for n in (80, 400):
        if n <= stage:
            statistics.append(verify_statistics(directory, n, queries, target))
    assert done['final_audit']['stage'] == stage
    assert statistics[-1]['decision'] == done['final_audit']['decision']
    composition=verify_composition(directory,stage,queries[:stage])
    paths = set(); record_count = 0
    for i, q in enumerate(queries[:stage]):
        base = read(directory / 'actual/images/records' / f'{i:03d}_parent.json')
        for arm in METHODS:
            row = read(directory / 'actual/images/records' / f'{i:03d}_{arm}.json')
            assert all(row[k] == v for k, v in q.items())
            assert row['query_index'] == i and row['arm'] == arm and row['execution'] == execution
            assert row['source_model'] == export['models'][arm]['source']
            assert not row['normal_inference_target_access'] and row['one_image_per_query']
            assert row['UNet_NFE'] == 4 and row['native_IP_K_calls'] == row['native_IP_V_calls'] == 280
            assert len(row['ancestral_noise_observations']) == 4
            for key in ('initial_noise_latents_sha256', 'initial_UNet_input_sha256', 'ancestral_noise_observations'):
                assert row[key] == base[key]
            path = PurePosixPath(row['generated_path'])
            assert path.parent == PurePosixPath(e['output_root']) / 'pixels' and str(path) not in paths
            paths.add(str(path)); record_count += 1
    metadata = read(directory / 'reference/target_metadata.json')['Dcheck']
    selected = e[f'fixed_qualitative{stage}']
    assert selected == manifest['fixed_qualitative_indices'] and len(selected) == 6
    for i in selected:
        target_bytes = (directory / 'qualitative' / f'target_{i:03d}.jpg').read_bytes()
        assert hashlib.sha256(target_bytes).hexdigest() == metadata[i]['image_sha256']
        for arm in METHODS:
            path = directory / 'qualitative' / f'{i:03d}_{arm}.webp'
            row = read(directory / 'actual/images/records' / f'{i:03d}_{arm}.json')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['generated_sha256']
            with Image.open(path) as image:
                rgb = np.asarray(image.convert('RGB'))
            assert rgb.shape == (512, 512, 3) and hashlib.sha256(rgb.tobytes()).hexdigest() == row['generated_pixel_sha256']
    result = dict(status='PASS_LOCAL_CLOSED_CORE_RANKS_42_CONTRASTS_RAW_CELLS_AND_PROVENANCE',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), registration=registration,
        execution=execution, stages=statistics, checkpoints=checkpoints,
        identity_and_noise_records=record_count, local_pixel_images=48, composition_interaction=composition,
        fixed_indices=selected, full_training_state_and_all_pixels_verified_on_server_only=True,
        no_local_vision_reencoding_or_gradient_replay=True, not_independent_H=True, not_SOTA=True)
    (directory / 'independent_local_statistics_and_provenance.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    return result

def render_fixed(directory):
    from PIL import Image, ImageDraw, ImageFont
    directory = Path(directory)
    verified = read(directory / 'independent_local_statistics_and_provenance.json')
    assert verified['status'].startswith('PASS_LOCAL_CLOSED_CORE')
    labels = ('target','parent','ordinary','no-O joint','no-O swapped','no-O absolute','joint + O','swapped + O','absolute + O')
    width, height, header, rowheight = 176, 176, 52, 208
    canvas = Image.new('RGB', (9 * width, header + 6 * rowheight), 'white')
    draw = ImageDraw.Draw(canvas)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    for column, label in enumerate(labels):
        draw.text((column * width + 5, 9), label, fill='black', font=font)
    queries = read(directory / 'reference/panel.json')['Dcheck']
    for row, i in enumerate(verified['fixed_indices']):
        for column, arm in enumerate(('target',) + METHODS):
            name = f'target_{i:03d}.jpg' if arm == 'target' else f'{i:03d}_{arm}.webp'
            with Image.open(directory / 'qualitative' / name) as source:
                image = source.convert('RGB'); image.thumbnail((width - 8, height - 8))
            canvas.paste(image, (column * width + (width - image.width) // 2,
                                header + row * rowheight))
        draw.text((5, header + row * rowheight + height),
                  f"q{i:03d}: {queries[i]['stimulus_id']}", fill='black', font=font)
    output = directory / 'fixed6_all8_methods.png'
    canvas.save(output)
    return str(output.resolve())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--render', action='store_true'); args = parser.parse_args()
    value = audit(args.directory)
    if args.render:
        value['fixed_panel'] = render_fixed(args.directory)
    print(json.dumps(value))
