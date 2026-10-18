"""Evaluation-only ATM-convention image metrics; no EEG or generation imports."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

ROOT = Path(__file__).resolve().parent
WEIGHT_ROWS = json.loads((ROOT / 'evaluator_references/weight_source_manifest.json').read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_weight(assets, name):
    row = next(r for r in WEIGHT_ROWS if Path(r['path']).name == name)
    path = Path(assets) / name
    assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
    return path


def load_inception(assets, device='cpu'):
    path = verified_weight(assets, 'inception_v3_google-0cc3c7bd.pth')
    model = models.inception_v3(weights=None, aux_logits=True, transform_input=True, init_weights=False)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return create_feature_extractor(model, return_nodes=['avgpool']).to(device).eval().requires_grad_(False)


def load_swav(assets, device='cpu'):
    path = verified_weight(assets, 'swav_800ep_pretrain.pth.tar')
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Identity()
    raw = torch.load(path, map_location='cpu', weights_only=True)
    raw = {k.removeprefix('module.'): v for k, v in raw.items()}
    wanted = model.state_dict()
    assert wanted.keys() <= raw.keys()
    model.load_state_dict({k: raw[k] for k in wanted}, strict=True)
    return create_feature_extractor(model, return_nodes=['avgpool']).to(device).eval().requires_grad_(False)


def images_256(images, clamp_generated=False):
    assert images.dtype == torch.float32 and images.ndim == 4 and images.shape[1] == 3
    assert torch.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
    output = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR,
                               antialias=True)(images)
    return output.clamp(0, 1) if clamp_generated else output


def feature_preprocess(size):
    assert size in [224, 342]
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


@torch.no_grad()
def extract_features(images, model, size, batch_size=200):
    """Use source per-image preprocessing on the model's execution device."""
    assert not model.training and batch_size in [1, 2, 8, 16, 200]
    device = next(model.parameters()).device
    transform = feature_preprocess(size)
    result = []
    for batch in images.split(batch_size):
        batch = batch.to(device)
        prepared = torch.stack([transform(row) for row in batch])
        values = model(prepared)['avgpool'].float().flatten(1).cpu()
        assert torch.isfinite(values).all()
        result.append(values)
    return torch.cat(result).numpy()


def identification_comparisons(target_features, generated_features):
    """Rows are real distractors, columns are generated queries; ties fail."""
    real = np.asarray(target_features, dtype=np.float32)
    generated = np.asarray(generated_features, dtype=np.float32)
    assert real.shape == generated.shape and real.ndim == 2 and len(real) >= 2
    assert np.isfinite(real).all() and np.isfinite(generated).all()
    assert np.all(np.std(real, axis=1) > 0) and np.all(np.std(generated, axis=1) > 0)
    count = len(real)
    corr = np.corrcoef(real, generated)[:count, count:]
    success = corr < np.diag(corr)
    assert not np.diag(success).any()
    return success


def identification(target_features, generated_features):
    """Exact author's numpy.corrcoef convention, including strict ties."""
    success = identification_comparisons(target_features, generated_features)
    scores = np.sum(success, axis=0) / (len(success)-1)
    assert np.isfinite(scores).all() and scores.min() >= 0 and scores.max() <= 1
    return scores


def correlation_distance(target_features, generated_features):
    from scipy.spatial.distance import correlation
    real = np.asarray(target_features, dtype=np.float32)
    generated = np.asarray(generated_features, dtype=np.float32)
    assert real.shape == generated.shape
    result = np.array([correlation(a, b) for a, b in zip(real, generated)])
    assert np.isfinite(result).all()
    return result


def pixel_metrics(targets, generated):
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity
    assert targets.shape == generated.shape and targets.shape[1:] == (3, 256, 256)
    resize = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    pixcorr, ssim = [], []
    # Streaming avoids materializing two 200x425x425x3 arrays at once. Pixel
    # operations for one query are identical to the source's batched version.
    for a, b in zip(targets, generated):
        a, b = resize(a).cpu(), resize(b).cpu()
        af, bf = a.flatten().numpy(), b.flatten().numpy()
        value = float(np.corrcoef(af, bf)[0, 1]) if af.std() > 0 and bf.std() > 0 else None
        pixcorr.append(value)
        ag, bg = (rgb2gray(x.permute(1, 2, 0).numpy()) for x in [a, b])
        ssim.append(float(structural_similarity(bg, ag, gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False, data_range=1., channel_axis=None)))
    assert all(v is None or np.isfinite(v) for v in pixcorr) and np.isfinite(ssim).all()
    # A constant image makes source Pearson undefined. Keep that query and mark
    # it undefined; never drop failed outputs or silently replace NaN with .5.
    return {'PixCorr': pixcorr, 'SSIM': ssim}
