"""Check whether an auxiliary real-repeat mixture helps the frozen H/14 condition."""

import argparse
import csv
import json
from pathlib import Path
import tarfile

import numpy as np
import torch
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci, units
from paper_fusion_siglip_probe import measure, predict, rank_credit, ridge_system, solve_ridge
from paper_horizon_readout import VIEWS
from paper_r80_value_guidance import guidance_target
from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_tangent_calibration import read_inputs
from paper_temporal_readout import held_concepts


ARMS = ('real-control', 'same-object-mixture')
METRICS = ('h14_name_top1', 'h14_name_top5', 'siglip2_name_top1', 'siglip2_name_top5', 'h14_original_top1')


def categories(filename):
    with filename.open(encoding='utf-8-sig', newline='') as stream:
        return np.asarray([row['unique_id'] for row in csv.DictReader(stream)])


def native_conditions(native, guidance, held, sources, pins):
    r4, r80, receipts = [], [], {}
    repeats = [list(range(i, i + 4)) for i in range(0, 80, 4)]
    people = sources['physical']
    for subject in range(1, 11):
        key = f'sub-{subject:02d}'
        meta, path = native / f'{key}.json', native / f'{key}.h14.npz'
        proof = json.loads(meta.read_bytes())
        registration = proof['proof']['registration']
        require(pins.pin(path) == proof['h14_sha256'] and proof['proof']['subject'] == subject
                and proof['proof']['views'] == list(VIEWS) and proof['proof']['repeats'] == list(range(4))
                and all(registration[k] == sources['visual'][v] for k, v in
                        (('split', 'split'), ('development', 'dev'), ('labels', 'labels'))), 'Changed native readout')
        physical = people[str(subject)]
        require(all(proof['proof']['eeg'][k] == v for k, v in physical['train'].items()), 'Different physical R4 inputs')
        with np.load(path, allow_pickle=False) as value:
            require(np.array_equal(value['row'], held) and value['embedding'].shape == (len(held), len(VIEWS), 1024),
                    'Native R4 row order changed')
            r4.append(torch.from_numpy(value['embedding'][:, 0].copy()))
        archive = guidance / f'{key}.compact.tar'
        with tarfile.open(archive, 'r') as pack:
            physical_r80 = json.load(pack.extractfile('h14-guidance.json'))['record']['physical_eeg']
        require(all(physical_r80[k] == v for k, v in physical['test'].items()), 'Different physical R80 inputs')
        binding = dict(row=list(range(200)), repeats=repeats, physical_eeg=physical_r80,
                       split_sha256=sources['visual']['split'])
        estimate, receipt = guidance_target(archive, subject, binding, pins)
        r80.append(estimate)
        receipts[key] = dict(r4=pins.pin(meta), r80=receipt)
    return {'held': torch.stack(r4), 'official': torch.stack(r80)}, receipts


@torch.inference_mode()
def run(benchmark, mixture, native, guidance, device):
    pins = PinnedFiles()
    source, train, dev, fused, _, _, _, _, audit, _ = read_inputs(benchmark, pins)
    split = benchmark / 'verified/data/prep/paper_split.json'
    label = benchmark / 'verified/data/prep/train_concept.npy'
    held = held_concepts(benchmark, train, dev, np.load(label, allow_pickle=False))
    image_set = benchmark / 'inputs/image_set'
    train_registry, test_registry, concept_registry = (
        image_set / name for name in ('train_photos.csv', 'test_photos.csv', 'concepts1854.csv'))
    registry = {path.name: pins.pin(path) for path in (train_registry, test_registry, concept_registry)}

    pack = source / 'alignvis-20261127/alignvis-image-teachers.npz'
    image_receipt = json.loads(pack.with_suffix('.json').read_bytes())
    require(pins.pin(pack) == image_receipt['array_sha256'] and image_receipt['photo_registries'] ==
            {'train': registry[train_registry.name], 'test': registry[test_registry.name]}, 'Changed visual reference pack')

    def directions(values):
        return F.normalize(torch.as_tensor(np.asarray(values).copy(), device=device, dtype=torch.float32), dim=-1)

    with np.load(pack, allow_pickle=False) as value:
        names, official = value['train_word'], value['test_word']
        require(np.array_equal(names, categories(train_registry)) and np.array_equal(official, categories(test_registry))
                and set(names[train]).isdisjoint(official), 'Unexpected concept overlap or order')
        image_train, image_test = directions(value['train/h14']), directions(value['test/h14'])
    h14 = source / 'h14-text-bank.npz'
    require(pins.pin(h14) == json.loads(h14.with_suffix('.json').read_bytes())['bank_sha256'], 'Changed H/14 text bank')
    with np.load(h14, allow_pickle=False) as words:
        h_text = directions(words['text'])
        h_index = {word: i for i, word in enumerate(words['concept_id'])}
    siglip = benchmark / 'token-rank-check/siglip-independent/siglip-fused-evaluation-bridge.pt'
    sig_receipt = json.loads(siglip.with_name('siglip-fused-probe.json').read_bytes())
    require(pins.pin(siglip) == sig_receipt['evaluation_bridge_sha256']
            and registry[concept_registry.name] == sig_receipt['vision_and_text']['test']['concepts'], 'Changed SigLIP2 bridge')
    scorer = torch.load(siglip, map_location=device, weights_only=True)
    s_index = {word: i for i, word in enumerate(categories(concept_registry))}
    require(len(s_index) == len(h_index) == 1854, 'Full open vocabulary required for scoring')
    # Only reference images of the registered training concepts fit the representation bridge.
    bridge = solve_ridge(ridge_system(image_train[train][None], directions(fused[train])), .01)

    query, provenance, proof = {}, {}, None
    for name in ARMS:
        query[name], provenance[name] = {}, {}
        for panel, suffix, rows in (('held', '', held), ('official', '-official', np.arange(200))):
            estimate = mixture / f'{name}{suffix}.npz'
            receipt = estimate.with_suffix('.json')
            document = json.loads(receipt.read_bytes())
            proof = document['sources'] if proof is None else proof
            archive_hash = document['official_sha256' if suffix else 'predictions_sha256']
            require(document['sources'] == proof and document['arm'] == name and pins.pin(estimate) == archive_hash
                    and proof['visual']['split'] == audit['split'] and proof['real_repeat_halves'] == [[0, 1], [2, 3]],
                    'Changed real-repeat predictions')
            with np.load(estimate, allow_pickle=False) as prediction:
                require(np.array_equal(prediction['row'], rows) and prediction['embedding'].shape == (10, len(rows), 1024),
                        'Mixture values or order incomplete')
                query[name][panel] = directions(prediction['embedding'])
            provenance[name][panel] = pins.pin(receipt)
    frozen, native_receipts = native_conditions(native, guidance, held, proof, pins)
    results, paired_rows = {}, {}
    for panel, key, selector, truth, candidates in (
        ('dev', 'held', slice(0, 200), names[dev], image_train[dev]),
        ('fresh', 'held', slice(200, 700), names[held[200:]], image_train[held[200:]]),
        ('official', 'official', slice(None), official, image_test),
    ):
        current = directions(frozen[key][:, selector])
        scores = {'native': current}
        scores.update({name: F.normalize(.75 * current + .25 * query[name][key][:, selector], dim=-1) for name in ARMS})
        ts = np.asarray([s_index[x] for x in truth])
        th = np.asarray([h_index[x] for x in truth])
        observed = {}
        for name, vectors in scores.items():
            s = measure(predict(predict(vectors, bridge), scorer['readout']), scorer['text_bank'], scorer['names'].cpu().numpy(), ts)
            h = measure(vectors, h_text, np.arange(1854), th)
            p = rank_credit((vectors @ candidates.T).cpu().numpy(), np.arange(len(truth)), 1)
            observed[name] = np.concatenate((h, s, p[..., None]), axis=-1)
            paired_rows[f'{panel}/{name}'] = observed[name]
        results[panel] = dict(n=10 * len(truth), hits={k: dict(zip(METRICS, v.sum((0, 1)).tolist(), strict=True))
                                                     for k, v in observed.items()}, paired={})
        for name, reference in ((ARMS[1], ARMS[0]), (ARMS[1], 'native'), (ARMS[0], 'native')):
            delta = observed[name] - observed[reference]
            lo, hi = bootstrap_ci(units(delta, truth, panel))
            results[panel]['paired'][f'{name} minus {reference}'] = dict(
                zip(METRICS, np.stack((delta.mean((0, 1)), lo, hi), axis=-1).tolist(), strict=True))
    scores = mixture / 'mixture-native-composition-scores.npz'
    destination = mixture / 'mixture-native-composition.json'
    atomic_file(scores, lambda stream: np.savez_compressed(stream, **paired_rows))
    save_json(destination, dict(paired=results, score_sha256=pins.pin(scores), code=pins.pin(Path(__file__)), blend=.25,
                                bridge_ridge=.01, fit_photos=len(train), native=native_receipts,
                                inputs=provenance, registries=registry, image_pack=image_receipt['array_sha256'],
                                siglip_bridge=sig_receipt['evaluation_bridge_sha256'], h14_bank=pins.pin(h14)))
    pins.verify()
    for panel, result in results.items():
        print(panel, json.dumps(result['hits']), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('benchmark', type=Path)
    cli.add_argument('mixture', type=Path)
    cli.add_argument('native', type=Path)
    cli.add_argument('guidance', type=Path)
    cli.add_argument('--device', default='cuda')
    args = cli.parse_args()
    run(args.benchmark, args.mixture, args.native, args.guidance, args.device)
