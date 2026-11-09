"""Post-result D09 branch diversity and unavailable-oracle context, no fit."""
import argparse
from pathlib import Path
import numpy as np
import brainhive_gain_calibration_20261027_v1 as U


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    done = U.read(args.completion); assert done['stage'] == 400
    inputs = U.read(U.checked(done['inputs'])); panel = U.read(U.checked(inputs['panel']))
    g = np.load(U.checked(done['check_generated_features']), allow_pickle=False)
    t = np.load(U.checked(done['check_target_features']), allow_pickle=False)
    c = np.load(U.checked(done['check_conditions']), allow_pickle=False)
    assert g.shape == (400, 3, 768) and c.shape == (400, 3, 1024)
    g = g / np.linalg.norm(g, axis=2, keepdims=True)
    metrics = [U.S.metric_arrays(g[:, i], t, panel['Dcheck'], panel['Dcheck']) for i in range(3)]
    diversity = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        cosine = (g[:, i] * g[:, j]).sum(1)
        diversity[U.ARMS[i]+'__'+U.ARMS[j]] = dict(mean_DINO_feature_cosine=float(cosine.mean()),
            median_DINO_feature_cosine=float(np.median(cosine)), fraction_above_point95=float((cosine > .95).mean()))
    dino = np.stack([metric['DINO2AFC'] for metric in metrics], 1)
    top1 = np.stack([metric['gallery_top1'] for metric in metrics], 1)
    within = np.stack([metric['within_concept10_top1'] for metric in metrics], 1)
    result = dict(status='COMPLETE_D09_POSTRESULT_PORTFOLIO_LIMIT_DIAGNOSIS', source=U.ref(__file__),
        completion=U.ref(args.completion), diversity=diversity,
        all3_gallery_top1_miss_queries=int((top1.max(1) == 0).sum()),
        all3_within_concept_top1_miss_queries=int((within.max(1) == 0).sum()),
        all3_DINO2AFC_below_half_queries=int((dino.max(1) < .5).sum()),
        unavailable_perquery_DINO_oracle_mean=float(dino.max(1).mean()),
        unavailable_perquery_DINO_oracle_minus_risk=float((dino.max(1)-dino[:, 2]).mean()),
        oracle_uses_true_target_answers_and_is_not_an_inference_method=True,
        missing_object_identity_is_not_established_by_DINO_or_gallery_top1=True,
        model_or_choices_modified=False, GPU_used=False, SOTA_established=False)
    U.atomic(args.output, result); print(result)


if __name__ == '__main__': main()
