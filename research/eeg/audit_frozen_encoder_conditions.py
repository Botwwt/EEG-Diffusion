"""CPU-only, full T/V diagnosis of frozen native conditions, not image metrics.

The T mean predictor supplies a useful baseline for anisotropic raw CLIP features.
No test EEG, final benchmark, model fitting, or GPU is used here.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import torch.nn.functional as F
from locked_metrics import fixed_gallery_metrics

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def norm_stats(x):
    n = x.norm(dim=-1)
    return dict(mean=float(n.mean()), quantiles_0_25_50_75_100=torch.quantile(
        n, torch.tensor([0., .25, .5, .75, 1.])).tolist())


def main():
    torch.set_num_threads(2)
    started = time.monotonic()
    config = json.loads((ROOT/'configs/full_reference_resolved.json').read_text())
    split = json.loads((ROOT/'configs/split_manifest.json').read_text())
    assert digest(ROOT/'configs/split_manifest.json') == config['split_sha256']
    assert digest(config['visual_features_path']) == config['visual_features_sha256']
    target = torch.load(config['visual_features_path'], weights_only=True, map_location='cpu')
    assert target.shape == (16540, 1024) and torch.isfinite(target).all()
    images = {int(r['array_row']): r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open())
              if r['dataset_split'] == 'train'}
    t_rows = split['image_rows']['T']
    mean = target[t_rows].mean(0)
    out = DATA/'diagnostics/frozen_encoder_native_v1'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'completion.json').exists(), 'Preserve an earlier completed audit'
    summary = []
    with (out/'per_query.csv').open('x', newline='') as file:
        writer = None
        for subject in split['subjects']:
            proof = json.loads((Path(config['output_root'])/'full'/subject/'completion.json').read_text())
            assert digest(proof['condition_cache']) == proof['condition_cache_sha256']
            cache = torch.load(proof['condition_cache'], weights_only=True, map_location='cpu')
            assert cache['encoder_sha256'] == proof['encoder']['checkpoint_sha256']
            assert cache['native_visual_sha256'] == config['visual_features_sha256']
            assert cache['training_role'] == 'full'
            mapping = {int(row): i for i, row in enumerate(cache['array_rows'])}
            for role in ['T', 'V']:
                rows = split['image_rows'][role]
                y = target[rows]
                prediction = cache['condition'][[mapping[i] for i in rows]]
                assert torch.isfinite(prediction).all()
                ids = [images[i]['stimulus_id'] for i in rows]
                metric = fixed_gallery_metrics(prediction, y, ids, ids)
                yn = F.normalize(y, dim=-1)
                pn = F.normalize(prediction, dim=-1)
                similarity = pn @ yn.T
                best = similarity.argmax(-1)
                concepts = [images[i]['concept_id'] for i in rows]
                mse_per_query = (prediction-y).square().mean(-1)
                mean_mse = (mean-y).square().mean(-1)
                prediction_trace = float(prediction.var(0, unbiased=False).sum())
                target_trace = float(y.var(0, unbiased=False).sum())
                centered_cos = F.cosine_similarity(prediction-mean, y-mean, dim=-1)
                record = dict(subject=subject, role=role, queries=len(rows), gallery_images=len(rows),
                    native_MSE=float(mse_per_query.mean()), T_mean_predictor_MSE=float(mean_mse.mean()),
                    MSE_reduction_relative_to_T_mean=1-float(mse_per_query.mean()/mean_mse.mean()),
                    native_cosine=float(metric['correct_cosine'].mean()),
                    T_mean_predictor_cosine=float(F.cosine_similarity(mean[None], y).mean()),
                    T_mean_centered_cosine=float(centered_cos.mean()),
                    native_stimulus_Top1=float(metric['top1'].float().mean()),
                    native_stimulus_Top5=float(metric['top5'].float().mean()),
                    native_concept_Top1=sum(concepts[int(j)] == concepts[i] for i,j in enumerate(best))/len(rows),
                    native_gallery_2AFC=float(metric['two_afc'].mean()),
                    condition_variance_trace=prediction_trace, target_variance_trace=target_trace,
                    conditional_to_target_variance_trace_ratio=prediction_trace/target_trace,
                    condition_norm=norm_stats(prediction), target_norm=norm_stats(y),
                    encoder_sha256=proof['encoder']['checkpoint_sha256'],
                    condition_cache_sha256=proof['condition_cache_sha256'])
                summary.append(record)
                for k, index in enumerate(rows):
                    row = dict(subject=subject, role=role, stimulus_id=images[index]['stimulus_id'],
                        concept_id=images[index]['concept_id'], array_row=index, gallery_images=len(rows),
                        native_MSE=float(mse_per_query[k]), T_mean_predictor_MSE=float(mean_mse[k]),
                        native_cosine=float(metric['correct_cosine'][k]),
                        native_stimulus_Top1=int(metric['top1'][k]), native_stimulus_Top5=int(metric['top5'][k]),
                        native_gallery_2AFC=float(metric['two_afc'][k]),
                        condition_norm=float(prediction[k].norm()), target_norm=float(y[k].norm()))
                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=list(row));writer.writeheader()
                    writer.writerow(row)
                print(json.dumps(record), flush=True)
                del similarity
    report = dict(status='COMPLETED_FULL_T_AND_V_NATIVE_CONDITION_DIAGNOSIS', subjects=10,
        training_seed=1729, test_or_benchmark_accessed=False, GPU_used=False,
        visual_feature_sha256=config['visual_features_sha256'],
        split_sha256=config['split_sha256'], T_mean_native_norm=float(mean.norm()),
        source_sha256=digest(__file__), per_subject_role=summary,
        interpretive_limits=[
            'Native CLIP retrieval is not generated-image performance or proof of interface correctness.',
            'T has13230 gallery images; V has1660. Their raw retrieval accuracies are not directly matched.',
            'The mean predictor is fitted on T only. CLIP anisotropy makes uncentered cosine potentially misleading.',
            'This measures the selected frozen checkpoint. It is not a controlled capacity expansion.',
            'Low conditional variance alone does not prove implementation failure: an MSE conditional mean can shrink under uncertainty.'
        ], seconds=time.monotonic()-started)
    (out/'completion.json').write_text(json.dumps(report, indent=2)+'\n')
    (ROOT/'audit/frozen_encoder_native_diagnosis.json').write_text(json.dumps(report, indent=2)+'\n')
    columns = [k for k,v in summary[0].items() if not isinstance(v,dict)]
    with (ROOT/'audit/frozen_encoder_native_summary.csv').open('x', newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(summary)


if __name__ == '__main__':
    main()
