"""Test training-only image-to-EEG guidance on complete held-concept R4 panels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_neural_encoding import pool_trials
from paper_render_cache import PinnedFiles, render_transaction, require, save_json
from paper_visual_likelihood import BLOCK, MICROBATCH, STRENGTH, UPDATE_STEP, image_alignment_step


ARMS = ('d2', 'native-guide', 'neural-guide', 'neural-guide-wrong')


def readout(path, subject, split, physical, files):
    record = json.loads(path.with_suffix('.json').read_bytes())
    require(record['training_only'] is True and record['subject'] == subject
            and record['training_split'] == split and record['training_eeg'] == physical['array_sha256']
            and record['weights_sha256'] == files.pin(path) and record['mixing'] == .25,
            'The guidance head must be fit before access to the held concepts and use the locked development weight')
    with np.load(path, allow_pickle=False) as data:
        shapes = {'eeg_center': (680,), 'visual_center': (1024,), 'lower': (680, 680), 'projection': (1024, 680)}
        states = {}
        for key, shape in shapes.items():
            array = data[key]
            require(array.shape == shape and array.dtype == np.float32 and np.isfinite(array).all(),
                    f'Invalid training-only neural guide tensor {key}')
            states[key] = torch.from_numpy(array).cuda()
        require(bool((states['lower'].diagonal() > 0).all()), 'Invalid ERP covariance factor')
    return states, record, files.pin(path.with_suffix('.json'))


@torch.no_grad()
def run(subject, weights, output):
    from common import ROOT, RUNS
    from transformers import CLIPVisionModelWithProjection

    from paper_d2_object_swap import d2_models
    from paper_horizon_readout import repeat_views
    from paper_identity_render import load_generator
    from paper_middle_teacher import CONFIG_SHA, MODEL, MODEL_SHAS, image_processor
    from paper_photo_top1 import DEFAULT_LAYOUT, render_inputs
    from paper_render import condition, draw_images, generator_inputs, layout, query_seed, render_code, runtime, tensor_fingerprint
    from paper_token_evidence import score_outputs
    from paper_uniform_baselines import reference_layout

    require(RUNS.name == 'shared', 'Use the registered shared D2 generator and layout')
    files = PinnedFiles()
    raw, rows, _, budget, repeats, _, split, _, physical = render_inputs(subject, 'fresh', files)
    require(budget == 'fresh-r4' and len(rows) == 500 and repeats == [[0, 1, 2, 3]],
            'Each participant must use the complete fresh panel and four recorded presentations')
    state, head, head_sha = readout(weights, subject, split, physical, files)
    eeg = torch.from_numpy(pool_trials(raw, np.arange(len(raw))).mean(1)).cuda() - state['eeg_center']
    neural_target = F.normalize(torch.linalg.solve_triangular(state['lower'], eeg.T, upper=False).T)
    encoder, _, _, source = d2_models(files)
    native = []
    for start in range(0, len(raw), 20):
        trials = torch.from_numpy(raw[start:start + 20].copy()).cuda().float()
        native.append(repeat_views(encoder, trials, subject)[1][:, 0])
    target = F.normalize(torch.cat(native))
    del encoder

    model_dir = ROOT / 'models' / MODEL
    vision = files.pin(model_dir / 'model.safetensors')
    require(vision in MODEL_SHAS and files.pin(model_dir / 'config.json') == CONFIG_SHA, 'Unexpected frozen H/14 critic')
    image_tower = CLIPVisionModelWithProjection.from_pretrained(model_dir, local_files_only=True, use_safetensors=True,
                                                                torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
    processor = image_processor()
    mean = torch.tensor(processor.image_mean, device='cuda')[None, :, None, None]
    std = torch.tensor(processor.image_std, device='cuda')[None, :, None, None]
    assets = generator_inputs(files)
    fitted, layout_proof = reference_layout(subject, split, files)
    starts = layout(fitted, raw, repeats)
    pipe, adapter, projector, prompt, vae = load_generator(torch.float16, False)
    tokens, checkpoint, base = condition('d2', subject, raw, repeats, adapter, projector, files=files, return_source=True)
    require(base == source and pipe.scheduler.timesteps.tolist() == [999, 749, 499, 249], 'D2 components disagree')
    seeds = [query_seed('fresh', index) for index in range(len(rows))]
    strength, keep = DEFAULT_LAYOUT
    code = render_code(files, 'd2')
    for name in (Path(__file__).name, 'paper_neural_encoding.py', 'paper_visual_likelihood.py'):
        code[name] = files.pin(Path(__file__).with_name(name))
    receipt = dict(render_cache_version=2, subject=subject, budget=budget, n=len(rows), row=rows, split_sha256=split,
                   repeats=repeats, generator=assets, layout_source=layout_proof, layout_sha256=tensor_fingerprint(starts),
                   base_token_sha256=tensor_fingerprint(tokens), d2_source=base, checkpoint_sha256=checkpoint,
                   physical_eeg=physical, seeds=seeds, prompt='', training='shared', neural_weights_sha256=head['weights_sha256'],
                   neural_head_sha256=head_sha, native_target_sha256=tensor_fingerprint(target),
                   neural_target_sha256=tensor_fingerprint(neural_target), h14_weights=vision, guidance_step=UPDATE_STEP,
                   max_update_rms=STRENGTH, preserved_latent_block=BLOCK, gradient_microbatch=MICROBATCH, draws=1,
                   unet_calls_per_draw=4, reference_pixels_available_to_arm=False, runtime=runtime(), render_code_sha256=code)
    folders = []
    for arm in ARMS:
        donors = np.roll(np.arange(len(rows)), len(rows) // 2) if arm == 'neural-guide-wrong' else np.arange(len(rows))
        neural_query = neural_target[donors]
        weight = head['mixing'] if arm in ('neural-guide', 'neural-guide-wrong') else 0.
        folder = output / 'images/shared' / arm / budget / f'sub-{subject:02d}'

        def update(batch, step, clean):
            if arm == 'd2' or step != UPDATE_STEP:
                return None
            anchor, brain = target[batch], neural_query[batch]

            def similarity(image, start):
                stop = start + len(image)
                score = (image * anchor[start:stop]).sum()
                if weight:
                    projected = F.normalize((image - state['visual_center']) @ state['projection'])
                    score = (1 - weight) * score + weight * (projected * brain[start:stop]).sum()
                return score

            return image_alignment_step(clean, vae, image_tower, mean, std, similarity)

        def draw(stage, batches, finish):
            draw_images(stage, pipe, vae, prompt, tokens, seeds, starts, batch_starts=batches, batch_complete=finish,
                        layout_strength=strength, layout_low_keep=keep, clean_update=update if arm != 'd2' else None)

        render_transaction(folder, dict(receipt, arm=arm, guidance_mix=weight, guidance_donors=donors.tolist()), draw, files.verify)
        folders.append(folder)
        print(f'Verified fresh R4 renders for participant {subject:02d}, {arm}, {len(rows)} images', flush=True)
    del image_tower, pipe, vae, adapter, projector
    torch.cuda.empty_cache()
    score_outputs(folders, 'fresh')
    save_json(output / 'neural-guidance.json', dict(proof=receipt, arms=list(ARMS)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, choices=range(1, 11), required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.subject, args.weights, args.output)
