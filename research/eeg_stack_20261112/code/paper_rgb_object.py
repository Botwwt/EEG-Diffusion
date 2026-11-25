"""Fine-tune a personal token-photo encoder through the four-step RGB renderer, using real training photos."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import time
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from paper_render_cache import PinnedFiles, atomic_file, key, require, save_json, sha

MODES = ('token-rgb-photo', 'token-rgb-sibling')
SEED = 20261125
NEGATIVES = 4
TEMPERATURE = .07
LEARNING_RATE = 1e-5
LAST_WEIGHT = 'backbone.block.layers.0.ff.w2.weight'


def make_plan(rows, labels, layout, steps, seed, negatives):
    """Choose each query concept once. Rank other concepts by their closest coarse-RGB photo."""
    rows, labels, layout = np.asarray(rows), np.asarray(labels), np.asarray(layout, dtype=np.float32)
    require(rows.ndim == labels.ndim == 1 and rows.shape == labels.shape and rows.dtype.kind in 'iu'
            and len(np.unique(rows)) == len(rows) and np.all(rows >= 0), 'Expected unique registered photo rows')
    require(layout.ndim == 2 and len(layout) == len(rows) and np.isfinite(layout).all(), 'Invalid layout descriptors')
    order = np.argsort(rows)
    rows, labels, layout = rows[order], labels[order], layout[order]
    objects, counts = np.unique(labels, return_counts=True)
    require(0 < steps <= len(objects) and 0 < negatives < len(objects) and np.all(counts >= 2),
            'Need distinct query concepts, same-object siblings and enough negative concepts')
    rng = np.random.default_rng(seed)
    plan = []
    for label in rng.choice(objects, steps, replace=False):
        query, sibling = rng.choice(np.flatnonzero(labels == label), 2, replace=False)
        distance = np.square(layout - layout[query]).mean(axis=1)
        other = []
        seen = {label}
        for i in np.lexsort((rows, distance)):
            if labels[i] not in seen:
                other.append(int(rows[i]))
                seen.add(labels[i])
                if len(other) == negatives:
                    break
        plan.append(dict(query=int(rows[query]), sibling=int(rows[sibling]), negatives=other,
                         noise_seed=int(rng.integers(0, 2**63 - 1))))
    return plan


def target_rows(episode, mode):
    require(mode in MODES, 'Unknown positive-photo rule')
    positive = episode['query' if mode == 'token-rgb-photo' else 'sibling']
    return [positive, *episode['negatives']]


def registered_photos(prep):
    from common import concept_folds

    index = json.loads((prep / 'train_ids.json').read_bytes())
    rows = np.asarray(json.loads((prep / 'paper_split.json').read_bytes())['train_rows'])
    labels = np.load(prep / 'train_concept.npy', allow_pickle=False)
    require(labels.shape == (16540,) and labels.dtype.kind in 'iu' and labels.min() == 0 and labels.max() == 1653,
            'Expected the full registered training label index')
    expected = np.flatnonzero(concept_folds(1654)[labels] != 0)
    require(rows.shape == (13230,) and rows.dtype.kind in 'iu' and np.array_equal(np.sort(rows), expected),
            'RGB fitting must use exactly the registered 13,230 train rows')
    objects, counts = np.unique(labels[rows], return_counts=True)
    require(len(objects) == 1323 and np.all(counts == 10), 'Expected 1,323 complete ten-photo training objects')
    require(len(index['ids']) == len(index['paths']) == 16540, 'Unexpected photo index size')
    members = {int(r): index['paths'][r] for r in rows}
    require(len(set(members.values())) == len(rows), 'Duplicated registered image')
    for row, name in members.items():
        path = PurePosixPath(name)
        require(path.parts[0] == 'training_images' and len(path.parts) == 3 and path.stem == index['ids'][row]
                and int(path.parent.name.split('_', 1)[0]) - 1 == labels[row], 'Training photo or object ID mismatch')
    return rows, labels[rows], members


def read_photo(archive, member, size):
    with Image.open(io.BytesIO(archive.read(member))) as im:
        require(im.width == im.height, f'Expected a square THINGS training photograph: {member}')
        if size == 4:
            # JPEG native-scale reduction precedes an area average in each of 16 cells.
            im.draft('RGB', (32, 32))
        method = Image.Resampling.BOX if size == 4 else Image.Resampling.BILINEAR
        return np.asarray(im.convert('RGB').resize((size, size), method))


def dino_pixels(rgb):
    require(rgb.ndim == 4 and rgb.shape[1] == 3 and rgb.shape[2] == rgb.shape[3], 'Expected square RGB tensors')
    resized = F.interpolate(rgb.float(), size=(256, 256), mode='bicubic', align_corners=False, antialias=True).clamp(0., 1.)
    crop = resized[:, :, 16:240, 16:240]
    mean = crop.new_tensor((.485, .456, .406))[None, :, None, None]
    std = crop.new_tensor((.229, .224, .225))[None, :, None, None]
    return (crop - mean) / std


def dino_cls(model, pixels):
    features = model(pixel_values=dino_pixels(pixels)).last_hidden_state[:, 0].float()
    norms = features.norm(dim=1, keepdim=True)
    require(torch.all(torch.isfinite(norms) & (norms > 0)).item(), 'Invalid DINO training representation')
    return features / norms


def prepare_targets(archive_path, rows, labels, members, steps, model, device):
    with zipfile.ZipFile(archive_path) as archive:
        layout = np.stack([read_photo(archive, members[int(r)], 4).ravel() for r in rows])
        plan = make_plan(rows, labels, layout, steps, SEED, NEGATIVES)
        wanted = sorted({r for episode in plan for mode in MODES for r in target_rows(episode, mode)})
        targets = {}
        with torch.no_grad():
            for start in range(0, len(wanted), 16):
                batch = wanted[start:start + 16]
                images = np.stack([read_photo(archive, members[r], 512) for r in batch])
                pixels = torch.from_numpy(images).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255.
                vectors = dino_cls(model, pixels).cpu()
                targets.update(zip(batch, vectors.unbind(), strict=True))
    return plan, targets


class CheckpointedUNet:
    """Use the renderer's U-Net call unchanged while rematerializing its activations in backward."""

    def __init__(self, unet):
        self.unet = unet
        self.dtype = unet.dtype

    def __call__(self, sample, timestep, *, encoder_hidden_states, added_cond_kwargs, cross_attention_kwargs):
        def predict(latent, tokens):
            return self.unet(latent, timestep, encoder_hidden_states=encoder_hidden_states,
                             added_cond_kwargs=added_cond_kwargs, cross_attention_kwargs={'ip_hidden_states': tokens}).sample

        value = checkpoint(predict, sample, cross_attention_kwargs['ip_hidden_states'],
                           use_reentrant=False, preserve_rng_state=False)
        return SimpleNamespace(sample=value)


def render_rgb(unet, scheduler, vae, prompts, tokens, noise, rollout):
    eps, *step_noises = noise.unbind(1)
    require(len(step_noises) == 4 and len(scheduler.timesteps) == 4, 'Expected exactly four SDXL-Turbo steps')
    latent = rollout(CheckpointedUNet(unet), scheduler, [tokens] * 4, torch.zeros_like(eps), eps,
                     step_noises, *prompts, 0., None)
    scaled = latent.float() / vae.config.scaling_factor
    decoded = checkpoint(lambda z: vae.decode(z).sample, scaled, use_reentrant=False, preserve_rng_state=False)
    return (decoded.float() * .5 + .5).clamp(0., 1.)


def train_arm(mode, source_state, eeg, plan, targets, pipeline, dino, rollout):
    from paper_instance_residual import make_encoder

    pipe, adapter, projection, prompts = pipeline
    device = eeg.device
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    encoder = make_encoder(device).float().eval().requires_grad_(False)
    encoder.load_state_dict(source_state, strict=True)
    weight = encoder.get_parameter(LAST_WEIGHT).requires_grad_(True)
    opt = torch.optim.AdamW([weight], lr=LEARNING_RATE, betas=(.9, .995), weight_decay=0.)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    log = []
    for i, episode in enumerate(plan):
        noise = torch.randn(1, 5, 4, 64, 64, generator=torch.Generator().manual_seed(episode['noise_seed'])).to(device)
        tokens = projection(adapter(encoder.backbone(eeg[i:i + 1]).half()))
        image = render_rgb(pipe.unet, pipe.scheduler, pipe.vae, prompts, tokens, noise, rollout)
        pred = dino_cls(dino, image)
        options = torch.stack([targets[r] for r in target_rows(episode, mode)]).to(device)
        scores = pred @ options.T
        loss = F.cross_entropy(scores / TEMPERATURE, torch.zeros(1, device=device, dtype=torch.long))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([weight], 1., error_if_nonfinite=True)
        require(float(norm) > 0., 'RGB loss produced no encoder gradient')
        opt.step()
        log.append(dict(loss=float(loss.detach()), grad_norm=float(norm), positive=float(scores[0, 0].detach()),
                        max_negative=float(scores[0, 1:].detach().max())))
        if i == 0 or (i + 1) % 8 == 0:
            print(f'{mode} {i + 1}/{len(plan)} loss={log[-1]["loss"]:.4f} grad_norm={float(norm):.4g}', flush=True)
    delta = (weight.detach().cpu() - source_state[LAST_WEIGHT]).norm().item()
    require(math.isfinite(delta) and delta > 0., 'Final BrainMLP weight did not change')
    result = dict(loss=[row['loss'] for row in log], updates=log, weight_delta_l2=delta,
                  wall_seconds=time.perf_counter() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device), gpu=torch.cuda.get_device_name(device))
    return {k: v.detach().cpu() for k, v in encoder.state_dict().items()}, result


def fit(subject, modes, steps, dino_path=None):
    from filelock import FileLock
    from common import PREP, ROOT, RUNS, TURBO
    from diffusers import AutoencoderKL
    from transformers import Dinov2Model
    from paper_dino import MODEL_FILES, REVISION, TRAIN_ZIPS
    from paper_render import generator, generator_inputs, rollout_4step_step_ips

    require(torch.cuda.is_available(), 'The four-step RGB fit requires CUDA')
    require(steps in (32, 64) and len(set(modes)) == len(modes) and set(modes) <= set(MODES) and modes,
            'Choose distinct registered arms and 32 or 64 updates')
    require(RUNS.name == f'personal-{subject:02d}', 'Expected the fitting directory of this participant')
    dino_path = Path(dino_path) if dino_path is not None else ROOT / 'models/paper-dinov2-small'
    archive_path = Path(os.environ['ARCHIVE']) / 'assets/things_images_v254/training_images.zip'
    dest = RUNS / f'enc/object_tokens/sub-{subject:02d}'
    source = dest / 'token-photo.pt'
    eeg_path = PREP / f'sub-{subject:02d}_train.npy'
    files = PinnedFiles()
    proof = {'token_photo': files.pin(source), 'token_photo_receipt': files.pin(source.with_suffix('.json')),
             'eeg': files.pin(eeg_path), 'eeg_receipt': files.pin(eeg_path.with_suffix('.json')),
             'eeg_ids': files.pin(PREP / 'eeg_train_ids.json'), 'photo_ids': files.pin(PREP / 'train_ids.json'),
             'split': files.pin(PREP / 'paper_split.json'), 'concept': files.pin(PREP / 'train_concept.npy'),
             'archive': files.pin(archive_path), 'dino': {name: files.pin(dino_path / name) for name in MODEL_FILES},
             'generator': generator_inputs(files), 'code': files.pin(__file__)}
    for name in ('common', 'exp6_draft_intra4', 'paper_training', 'paper_instance_residual', 'paper_dino'):
        proof[name] = files.pin(Path(__file__).with_name(name + '.py'))
    base = json.loads(source.with_suffix('.json').read_bytes())
    require(base['mode'] == 'token-photo' and base['subject'] == subject and base['checkpoint_sha256'] == proof['token_photo']
            and base['photos'] == 13230 and base['concepts'] == 1323, 'Expected the registered personal token-photo starting fit')
    require(all(base['sources'][k] == proof[k] for k in ('eeg', 'split', 'concept'))
            and base['sources']['adapter'] == proof['generator']['prior/fusion_encoder/diffusion_pytorch_model.safetensors']
            and base['sources']['projection'] == proof['generator']['prior/proj/diffusion_pytorch_model.safetensors'],
            'Source encoder used different EEG, split or token projection')
    require(proof['dino'] == MODEL_FILES and proof['archive'] in TRAIN_ZIPS, 'Unexpected DINO weights or original training image archive')
    require(json.loads(eeg_path.with_suffix('.json').read_bytes())['output_sha256'] == proof['eeg']
            and json.loads((PREP / 'eeg_train_ids.json').read_bytes())['ids'] == json.loads((PREP / 'train_ids.json').read_bytes())['ids'],
            'EEG hash or row identity differs from the training archive')
    request = dict(seed=SEED, steps=steps, negatives=NEGATIVES, temperature=TEMPERATURE, lr=LEARNING_RATE,
                   subject=subject, weight=LAST_WEIGHT, sources_sha256=key(proof))
    with FileLock(str(dest / '.rgb-object.lock')):
        files.verify()
        pending = []
        for mode in modes:
            path = dest / f'{mode}.pt'
            receipt = path.with_suffix('.json')
            if receipt.is_file():
                previous = json.loads(receipt.read_bytes())
                require(previous['mode'] == mode and previous['request'] == request and path.is_file()
                        and previous['checkpoint_sha256'] == sha(path), f'Existing {mode} checkpoint or training request differs')
            else:
                pending.append(mode)
        if not pending:
            return
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = torch.device('cuda')
        dino = Dinov2Model.from_pretrained(str(dino_path), local_files_only=True, use_safetensors=True).to(device).float().eval().requires_grad_(False)
        require((dino.config.patch_size, dino.config.hidden_size, dino.config.num_hidden_layers) == (14, 384, 12),
                'Expected frozen DINOv2-S/14')
        rows, labels, members = registered_photos(PREP)
        plan, targets = prepare_targets(archive_path, rows, labels, members, steps, dino, device)
        target_sha = hashlib.sha256(torch.stack([targets[r] for r in sorted(targets)]).contiguous().numpy().astype('<f4').tobytes()).hexdigest()
        raw = np.load(eeg_path, mmap_mode='r', allow_pickle=False)
        require(raw.shape == (16540, 4, 17, 250) and raw.dtype.kind == 'f', 'Expected four registered EEG presentations per training photograph')
        eeg = torch.from_numpy(raw[[item['query'] for item in plan]]).to(device=device, dtype=torch.float32).mean(1)
        require(torch.isfinite(eeg).all().item(), 'Nonfinite EEG among the selected training rows')
        source_state = torch.load(source, map_location='cpu', weights_only=True)
        pipe, adapter, projection, text, pooled, dims = generator(torch.float16, training=True)
        pipe.vae = None
        pipe.vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder='vae', variant='fp16', torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
        pipe.vae.enable_gradient_checkpointing()
        pipe.vae.enable_slicing()
        require(math.isclose(float(pipe.scheduler.init_noise_sigma), float(pipe.scheduler.sigmas[0]), rel_tol=1e-6),
                'Noise-only initialization differs from the native renderer')
        pipeline = (pipe, adapter, projection, (text, pooled, dims))
        for mode in pending:
            state, measured = train_arm(mode, source_state, eeg, plan, targets, pipeline, dino, rollout_4step_step_ips)
            files.verify()
            path = dest / f'{mode}.pt'
            atomic_file(path, lambda f: torch.save(state, f))
            meta = dict(mode=mode, subject=subject, checkpoint_sha256=sha(path), sources=proof, request=request,
                        seed=SEED, steps=steps, photos=len(rows), concepts=len(set(labels)), episodes=plan, plan_sha256=key(plan),
                        unique_target_rows=sorted(targets), target_vectors_sha256=target_sha, training_supervision='real training photo DINOv2-S/14 final CLS',
                        positive='own query photo' if mode == MODES[0] else 'distinct registered photograph of the same training object',
                        negatives='four distinct other objects, MSE to query on 4x4 RGB cell means, ties by photo row',
                        loss_definition='five-way cross-entropy of unit DINO CLS cosine / 0.07, positive first',
                        target_preprocessing='bilinear 512 RGB; shared differentiable bicubic 256, center 224, ImageNet normalization',
                        dino_revision=REVISION, frozen_generator=True, trainable_parameters=[LAST_WEIGHT], frozen_evaluation_mode=True,
                        inference='personal BrainMLP -> native adapter and projection -> four SDXL-Turbo steps -> VAE',
                        gradient='complete chain, all four Euler ancestral U-Net calls, fp32 VAE, continuous clamped RGB, frozen fp32 DINO; activation rematerialization; no detach or surrogate',
                        generator_dtype='fp16 U-Net and IP modules; fp32 Euler state, VAE and EEG encoder', text_prompt='',
                        initial_latent='seeded Gaussian noise', draws_per_update=1, denoising_steps_per_draw=4,
                        nominal_unet_forward_calls=4 * steps, gradient_clip_norm=1., adamw_betas=[.9, .995], weight_decay=0.,
                        torch_version=str(torch.__version__), **measured)
            save_json(path.with_suffix('.json'), meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, required=True, choices=range(1, 11))
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--steps', type=int, choices=(32, 64), default=32)
    parser.add_argument('--dino', type=Path, help='Frozen facebook/dinov2-small directory; defaults to RAD_ROOT/models/paper-dinov2-small')
    args = parser.parse_args()
    fit(args.subject, args.modes, args.steps, args.dino)
