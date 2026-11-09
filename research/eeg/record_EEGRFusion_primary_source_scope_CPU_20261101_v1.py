"""Record a scoped source review. Does not import or execute author code."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audit/EEGRFusion_public_source_review_20261101_v1'


def main():
    provenance = json.loads((OUT / 'source_provenance.json').read_bytes())
    assert provenance['text_files'] == 81
    commit = provenance['commit']
    assert commit == 'a5a8ff0b30593208c0f2c73fff61fc92665bb5af'
    source_files = {p['path']: p for p in provenance['files']}
    findings = []

    def add(path, kind, start, end, required, finding, limitation):
        local = OUT / path
        data = local.read_bytes()
        text = data.decode('utf8')
        lines = text.splitlines()
        fragment = '\n'.join(lines[start-1:end])
        assert all(s in fragment for s in required), (path, start, end)
        source = path.removeprefix('source/')
        if path.startswith('source/'):
            assert hashlib.sha256(data).hexdigest() == source_files[source]['sha256']
        else:
            source = path.removeprefix('notebook_code/').replace('.code.py', '.ipynb')
            assert source in source_files
        findings.append(dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                             lines=[start, end], kind=kind, finding=finding, limitation=limitation,
                             pinned_author_URL=f'https://github.com/hcy1026/EEGRFusion/blob/{commit}/{source}'))

    add('source/Generation/MAMD_reconstruction.py', 'coordinate_interface', 275, 289,
        ['proj_dim=1024', 'nn.LayerNorm(proj_dim)'],
        'Generation encoder projects to a 1024-dimensional LayerNorm output.',
        'Dimensional equality with HIVE is not coordinate equality. Reusing the HIVE ordinary weights is not justified.')
    add('source/Generation/MAMD_reconstruction.py', 'epoch_reporting', 534, 586,
        ['test_dataloader', 'if v2_acc > best_v2_acc:', 'best_epoch_info'],
        'The training loop evaluates the test loader each epoch and records the best two-way epoch information.',
        'Periodic checkpoint saving is separate; this does not prove that the best-report checkpoint was saved or used for generation. Our method selection must use internal T validation.')
    add('notebook_code/Generation/Generation_metrics_sub8_EEGRFusion.code.py', 'condition_target', 568, 576,
        ['ViT-H-14_features_train.pt', 'view(1654, 10, 1, 1024)'],
        'The generation notebook uses CLIP H/14 training targets and expands four EEG repetitions.',
        'This target differs from the HIVE fusion target and requires an independent host baseline and verified identity mapping.')
    add('notebook_code/Generation/Generation_metrics_sub8_EEGRFusion.code.py', 'training_control', 742, 798,
        ['images_train[img_idx]', 'blurry_tensor = blurry_gt', 'blurry_tensor = blur_tensor(coarse_01'],
        'The training curriculum mixes training-image blur and blur from a generated coarse image.',
        'Training-image supervision is not evidence that test ground truth is fed into normal inference.')
    add('notebook_code/Generation/Generation_metrics_sub8_EEGRFusion.code.py', 'inference_and_cost', 858, 905,
        ['num_inference_steps=4', 'num_inference_steps=16', 'control_tensor = blur_tensor(coarse_01', 'for j in range(10):'],
        'The inference notebook generates a coarse image, blurs it for control, and generates ten refinements. The RF prior has sixteen steps and each image generator call has four steps.',
        'RF evaluations are not UNet evaluations. This is not our single-image four-UNet-call protocol. No query-ground-truth control was found in this reviewed inference block.')
    add('notebook_code/Generation/Reconstruction_Metrics_EEGRFusion.code.py', 'metric_reduction', 210, 241,
        ['reduce_recons="mean"', 'preds = preds.reshape(N, K, -1)', 'preds = preds.mean(axis=1)', 'np.corrcoef'],
        'The identification helper averages multiple reconstruction feature vectors before Pearson identification when their count is a multiple of target count.',
        'This is feature averaging, not demonstrated best-of-N selection or averaging of per-image identification scores.')
    add('notebook_code/Generation/Reconstruction_Metrics_EEGRFusion.code.py', 'metric_model', 502, 514,
        ['clip.load("ViT-L/14"', 'reduce_recons="mean"'],
        'The CLIP evaluation cell requests OpenAI ViT-L/14 and the feature-mean reduction.',
        'Training H/14 and evaluation L/14 are different roles. Pixel and SSIM preprocessing was not comprehensively reviewed.')

    review = dict(
        status='COMPLETE_SCOPED_PRIMARY_PAPER_AND_PINNED_SOURCE_REVIEW_NOT_REPRODUCTION',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=commit, source_tree=provenance['tree'],
        downloaded_text_files=81, downloaded_text_bytes=provenance['total_bytes'],
        entire_repository_comprehensively_reviewed=False, findings=findings,
        primary_sources=[
            dict(title='EEGRFusion, MICCAI 2026', URL='https://papers.miccai.org/miccai-2026/paper/4361_paper.pdf',
                 reviewed_scope='Method, retrieval epoch rule, generation Tables 3/4; parsed PDF text, not a visual replication.',
                 consequence='MAMD, rectified-flow prior and IP-Control fusion are existing author contributions. Table 3 concerns Subject 8; do not compare it directly with our ten-subject single-image aggregate.'),
            dict(title='Img2EEG, preprint version 1, 2026-09-18', URL='https://pmc.ncbi.nlm.nih.gov/articles/PMC13596387/',
                 reviewed_scope='Architecture and stimulus-specific retrieval method; explicitly not peer reviewed.',
                 consequence='Image-to-EEG modelling and common-waveform versus stimulus-specific distinctions are not new ideas. Candidate-image retrieval is a different task from EEG-only open image generation.'),
            dict(title='Walther et al., NeuroImage 2016', URL='https://www.diedrichsenlab.org/pubs/Walther_Neuroimage_2016.pdf',
                 reviewed_scope='Crossvalidated noise-normalized representational distances.',
                 consequence='Crossnobis/RSA/noise normalization are classical. Independent-noise assumptions do not establish neural specificity in our four-repeat bank.')],
        design_counterexamples='audit/EEG_geometry_design_assumptions_20261101_v1/actual_counterexamples_and_design_constraints.json',
        next_actions_changed=[
            'Do not relabel generic Mamba/RF/control, image-to-EEG, geometry alignment, or variance preservation as a new contribution.',
            'Do not migrate ordinary weights merely because another host also has 1024 output coordinates.',
            'Before a new host fit, verify assets/data rights and identity joins, define internal held-concept selection, and compare the same single-image sampling cost.',
            'Complete the independently registered noO study unchanged; use actual stimulus discrimination and images to decide any further EEG target mechanism.'],
        noO_scientific_sources_or_gates_changed=False, author_code_imported=False,
        author_code_executed=False, author_weights_loaded=False, author_models_or_metrics_reproduced=False,
        new_GPU_jobs=0, new_training_updates=0, new_images=0, novelty_established=False)
    destination = OUT / 'actual_primary_source_and_scope_review.json'
    with destination.open('xb') as f:
        f.write((json.dumps(review, indent=2, ensure_ascii=True, allow_nan=False)+'\n').encode())
    print(json.dumps(dict(status=review['status'], checked_source_findings=len(findings), output=str(destination))))


if __name__ == '__main__':
    main()
