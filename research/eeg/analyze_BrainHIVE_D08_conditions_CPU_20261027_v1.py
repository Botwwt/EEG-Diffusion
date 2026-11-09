"""Post-result complete13230 condition geometry; no fit, images, or CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def unit(value): return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit-summary', 'native-binding', 'targets', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    assert not args.output.exists(); args.output.mkdir(parents=True)
    fitted = json.loads(args.fit_summary.read_bytes()); native = json.loads(args.native_binding.read_bytes())
    arm = fitted['arms']['repeat_reliable']; assert fitted['status'] == 'COMPLETE_REPEAT_RELIABLE_FULL_F_DECODER'
    assert sha(arm['predictions_path']) == arm['predictions_sha256']
    assert sha(native['predictions_path']) == native['predictions_sha256']
    assert sha(args.targets) == '363e8e74514ca8d1bcbfb29afc2023ed4a5c3a6c4f49bf8b1134e7ae7a090336'
    index = Path(fitted['prediction_index']); assert sha(index) == fitted['prediction_index_sha256'] == native['prediction_index_sha256']
    items = [json.loads(line) for line in index.read_text().splitlines() if line]; assert len(items) == 13230
    y = np.load(args.targets, allow_pickle=False).astype(np.float64); truth = unit(y)
    result = {}
    for name, path in (('native', native['predictions_path']), ('repeat_reliable', arm['predictions_path'])):
        x = np.load(path, allow_pickle=False).astype(np.float64); prediction = unit(x)
        output = []
        for fold in (0, 1):
            held = [i for i, row in enumerate(items) if row['fold'] == fold]
            fit = [i for i, row in enumerate(items) if row['fold'] != fold]
            groups = {}
            for i in held: groups.setdefault(items[i]['concept_id'], []).append(i)
            groups = list(groups.values()); assert all(len(ids) == 10 for ids in groups)
            correct = []; same_wrong = []; different_wrong = []
            for n, ids in enumerate(groups):
                positive = (prediction[ids] * truth[ids]).sum(1)
                negative_mean = (prediction[ids] @ truth[ids].sum(0) - positive) / 9
                other = groups[(n + 1) % len(groups)]
                correct.extend(positive); same_wrong.extend(negative_mean)
                different_wrong.extend((prediction[ids] * truth[other]).sum(1))
            vector = x[held].mean(0)
            energy_fraction = float(vector @ vector / np.square(x[held]).sum(1).mean())
            target_mean = y[fit].mean(0); target_mean /= np.linalg.norm(target_mean)
            output.append(dict(outer_fold=fold, queries=len(held), correct_cosine=float(np.mean(correct)),
                sameconcept_other9_cosine=float(np.mean(same_wrong)),
                correct_minus_sameconcept_other9=float(np.mean(np.array(correct) - same_wrong)),
                nextconcept_cosine=float(np.mean(different_wrong)),
                correct_minus_nextconcept=float(np.mean(np.array(correct) - different_wrong)),
                mean_cosine_to_outerFIT_visual_mean=float((prediction[held] @ target_mean).mean()),
                condition_population_mean_energy_fraction=energy_fraction,
                population_mean_energy_is_descriptive_not_individual_brain_signal_fraction=True))
        result[name] = output
    document = dict(status='COMPLETE_D08_ALL13230_POSTRESULT_CONDITION_GEOMETRY_CPU', rows=13230, results=result,
        fit_summary=ref(args.fit_summary), native_binding=ref(args.native_binding), targets=ref(args.targets), index=ref(index), source=ref(__file__),
        noise_or_public_component_causality_proved=False, model_or_sampling_changed=False, GPU_used=False,
        held_values_used_only_in_postresult_statistics=True, seconds=time.monotonic() - started)
    (args.output / 'summary.json').write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    print(json.dumps(document)); return 0


if __name__ == '__main__': raise SystemExit(main())
