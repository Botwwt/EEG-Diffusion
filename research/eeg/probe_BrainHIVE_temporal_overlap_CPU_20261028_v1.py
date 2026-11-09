"""D33: all-T structural measurement; never a reconstruction efficacy test."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import signal
import socket
import time
import traceback
import numpy as np
import brainhive_temporal_overlap_20261028_v1 as O

START = time.monotonic()


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    return dict(path=str(path), bytes=Path(path).stat().st_size, sha256=sha(path))


def checked(item):
    path = Path(item['path'])
    assert path.is_file() and sha(path) == item['sha256'], str(path)
    if 'bytes' in item:
        assert path.stat().st_size == item['bytes']
    return path


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def rss():
    return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:'))) * 1024


def run(spec_path):
    spec = read(spec_path)
    assert spec['status'] == 'D33_T_ONLY_TEMPORAL_OVERLAP_SPEC_LOCKED'
    assert spec['lags'] == [50, 49, 51, 73] and spec['fingerprint_dimensions'] == 32
    assert spec['selection_samples'] == 64 and spec['verification_samples'] == 64 and spec['guard'] == 20
    assert spec['seed'] == 1729 and spec['no_automatic_model_or_image_admission']
    output = Path(spec['output'])
    assert not output.exists()
    output.mkdir(parents=True)
    base_rss = rss()
    stage = 'VERIFY_INPUTS'
    results = []

    def budget():
        assert time.monotonic() - START < 850, 'Finite900second CPU job reserve'
        assert rss() - base_rss < 10 << 30, 'Original extraCPU cap'
        assert sum(p.stat().st_size for p in output.iterdir() if p.is_file()) < 128 << 20

    try:
        for item in spec['source_pins']:
            checked(item)
        trials_path = checked(spec['T_trials'])
        index_path = checked(spec['T_index'])
        split = read(checked(spec['split']))
        auth = read(checked(spec['authorization']))
        assert auth['formal_old_D_training_resumed'] is False and auth['existing_ordinary_baseline_bytes_immutable']
        rows = [json.loads(x) for x in index_path.read_text().splitlines()]
        assert len(rows) == 13230 and [r['array_row'] for r in rows] == split['image_rows']['T']
        assert len({r['stimulus_id'] for r in rows}) == 13230
        assert set(r['concept_id'] for r in rows) == set(split['train_concepts']['T'])
        assert set(Counter(r['concept_id'] for r in rows).values()) == {10}
        for i, row in enumerate(rows):
            assert row['row'] == i and row['partition'] == 'T' and row['subject_id'] == 'sub-01'
            assert row['fold'] == split['oof_fold_by_concept'][row['concept_id']]
        trials = np.load(trials_path, mmap_mode='r', allow_pickle=False)
        assert trials.shape == (13230, 4, 17, 250) and trials.dtype == np.float32
        flat = trials.reshape(52920, 17, 250)
        node_stimulus = np.repeat(np.arange(13230), 4)
        node_fold = np.repeat([r['fold'] for r in rows], 4)
        write(output / 'provenance.json', dict(
            status='T_ONLY_SIGNAL_OVERLAP_NO_IMAGE_LABEL_SEARCH', spec_sha256=sha(spec_path),
            hostname=socket.gethostname(), trial_nodes=52920, stimulus_rows=13230,
            node_order='4*stimulus_T_row+physical_repeat_index; no session identity assumed',
            fingerprints_selected_without_labels=True, verification_time_samples_disjoint=True,
            targets_F_or_images_read=False, ordinary_or_encoder_weights_read=False,
            source_recording_event_identity_independently_available=False,
            methods_are_classical_overlap_matching_not_claimed_novel_architecture=True))
        quantiles = [0, .01, .5, .99, 1]
        for lag in spec['lags']:
            stage = 'LAG_' + str(lag)
            budget()
            t0 = time.monotonic()
            left, right = O.fingerprints(flat, lag)
            lp, rp = output / f'lag{lag}_source_fingerprint.npy', output / f'lag{lag}_target_fingerprint.npy'
            np.save(lp, left, allow_pickle=False)
            np.save(rp, right, allow_pickle=False)
            target, scores, inverse, inverse_scores = O.search(left, right, check=budget)
            budget()
            verified = O.verify(flat, lag, target[:, 0])
            keep, reciprocal = O.accepted(target, scores, inverse, verified)
            structure = O.graph(len(flat), target[:, 0], keep)
            matched = np.flatnonzero(keep)
            same_stimulus = int((node_stimulus[matched] == node_stimulus[target[matched, 0]]).sum())
            cross_fold = int((node_fold[matched] != node_fold[target[matched, 0]]).sum())
            npz = output / f'lag{lag}_all52920_matches.npz'
            np.savez_compressed(npz, target=target, scores=scores, inverse=inverse,
                                inverse_scores=inverse_scores, verified=verified, accepted=keep,
                                reciprocal=reciprocal)
            item = dict(lag=lag, all_source_nodes=len(flat), exact_all_pair_fingerprint_comparisons=len(flat) * (len(flat) - 1),
                        accepted=int(keep.sum()), accepted_fraction=float(keep.mean()),
                        source_target_same_stimulus_accepted=same_stimulus,
                        concept_fold_crossing_accepted=cross_fold, graph=structure,
                        ordered_quantiles=quantiles,
                        fingerprint_best_cosine_quantiles=np.quantile(scores[:, 0], quantiles).tolist(),
                        held_difference_relative_error_quantiles=np.quantile(verified[:, 0], quantiles).tolist(),
                        held_difference_cosine_quantiles=np.quantile(verified[:, 1], quantiles).tolist(),
                        held_demeaned_raw_relative_error_quantiles=np.quantile(verified[:, 2], quantiles).tolist(),
                        reciprocal_matches=int(reciprocal.sum()), matches=ref(npz),
                        source_fingerprints=ref(lp), target_fingerprints=ref(rp), seconds=time.monotonic() - t0)
            write(output / f'lag{lag}_summary.json', item)
            results.append(item)
            progress = dict(status='D33_FULL_T_OVERLAP_PROGRESS', completed_lags=[x['lag'] for x in results],
                            results=results, seconds=time.monotonic() - START)
            temp = output / 'progress.pending'
            write(temp, progress)
            os.replace(temp, output / 'progress.json')
            print(json.dumps(dict(lag=lag, accepted=item['accepted'], fraction=item['accepted_fraction'], seconds=item['seconds'])), flush=True)
            del left, right, target, scores, inverse, inverse_scores, verified, keep, reciprocal
        by_lag = {x['lag']: x for x in results}
        positive = by_lag[50]
        maximum_control = max(by_lag[k]['accepted_fraction'] for k in (49, 51, 73))
        structure_supported = (positive['accepted_fraction'] >= .05 and maximum_control <= .001
                               and positive['graph']['cyclic_source_nodes'] == 0
                               and positive['graph']['maximum_path_nodes'] <= 20)
        budget()
        assert sha(trials_path) == spec['T_trials']['sha256']
        value = dict(status='D33_ALL13230_T_52920_PHYSICAL_TRIAL_OVERLAP_DIAGNOSTIC_COMPLETE',
                     UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), spec=ref(spec_path),
                     results=results, registered_strict_overlap_support=bool(structure_supported),
                     maximum_wrong_lag_accepted_fraction=maximum_control,
                     decision='PARTIAL_CONTINUOUS_SEGMENT_STRUCTURE_SUPPORTED_NEEDS_EVENT_AND_MODEL_VALIDATION' if structure_supported
                     else 'STOP_THIS_STRICT_SHARED_SEGMENT_RECOVERY_INSTANCE_NO_DECONVOLUTION_ADMISSION',
                     seconds=time.monotonic() - START, peak_extra_CPU_bytes=rss() - base_rss,
                     images=0, model_fits=0, SGD_updates=0, GPU_jobs=0,
                     target_neural_component_identified=False, reconstruction_gain_established=False,
                     neighboring_query_inputs_allowed_for_deployment=False,
                     event_identity_or_session_chain_independently_verified=False,
                     no_automatic_model_or_image_admission=True)
        write(output / 'completion.json', value)
        print(json.dumps(value), flush=True)
    except BaseException as exc:
        write(output / ('failure_' + str(time.time_ns()) + '.json'), dict(
            status='D33_FAILED_INCOMPLETE', stage=stage, completed_lags=[x['lag'] for x in results],
            error=repr(exc), traceback=traceback.format_exc(), seconds=time.monotonic() - START,
            peak_extra_CPU_bytes=rss() - base_rss))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('Finite CPU deadline')))
    run(args.spec)
