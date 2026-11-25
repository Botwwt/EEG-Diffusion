"""Make fixed single-person EEG features for the 416/96 pose test and final personal12 inference."""

import argparse
from pathlib import Path

import numpy as np
from measure_personal_pose import LOCK
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz


def features(a):
    import torch
    from fit_shared_trial_encoder import torch_setup
    from fit_trial_object_encoder import TrialObjectEncoder, fit, source_data
    from safetensors.torch import load_file

    torch_setup()
    sub = a.subject
    d = read(a.out / "design.json")
    check(sha(a.out / "design.json") == LOCK and sub in d["subjects"], "Wrong study")
    data = a.data
    old = sub == "sub-04"
    mix = data / ("trial-object-encoder/personal12" if old else f"trial-object-replication/{sub}")
    checkpoint512 = (
        data / ("trial-object-encoder/target" if old else f"trial-object-replication/{sub}/fit")
    ) / "model.safetensors"
    checkpoint416 = data / "shared-trial-encoder/individual" / (sub + ".safetensors")
    split = data / "episodic-object" / sub / "model.npz"
    expanded = data / "photo-structure/expanded/features" / (sub + ".json")
    dates = data / "object-transfer/sessions.json"
    check(
        sha(checkpoint512) == d["people"][sub]["fit512"]
        and sha(checkpoint416) == d["people"][sub]["fit416"]
        and sha(split) == d["people"][sub]["split"]
        and sha(expanded) == d["people"][sub]["generic_source"]
        and sha(dates) == d["inputs"]["schedules"]
        and sha(Path(__file__).with_name("fit_trial_object_encoder.py"))
        == d["inputs"]["training_code"],
        "Training split, original feature model or physical IDs changed",
    )
    src = data / ("trial-object-inputs/sub-04" if old else f"trial-object-replication/{sub}/trials")
    table = src / ("generic512.json" if old else "receipt.json")
    physical = read(table)["blocks"]
    with np.load(split) as z:
        ids = z["train_ids"]
        held = z["generic_heldout_ids"]
    train = ids[~np.isin(ids, held)]
    check(
        len(train) == 416
        and len(held) == 96
        and len(physical) == 512
        and all(
            physical[c]["source_raw_sha256"] == expected
            for c, expected in read(expanded)["raw_block_sha256"].items()
        ),
        "Wrong EEG range manifest",
    )
    source_receipt = dict(
        registration_sha256=LOCK,
        subject=sub,
        trial_manifest_sha256=sha(table),
        fit512_sha256=sha(checkpoint512),
        fit416_sha256=sha(checkpoint416),
        training_code_sha256=d["inputs"]["training_code"],
    )
    dest = a.out / "features" / sub
    if (dest / "receipt.json").exists():
        r = read(dest / "receipt.json")
        check(
            all(r[k] == v for k, v in source_receipt.items())
            and r["packet_sha256"] == sha(dest / "features.npz"),
            "Features changed",
        )
        return
    device = torch.device(a.device)
    eeg = source_data(src, ids, physical, device)
    ti = np.searchsorted(ids, train)
    schedule = np.asarray(read(dates)["subjects"][sub]["session_pairs"])
    _, groups = np.unique(schedule[ids], axis=0, return_inverse=True)

    def apply(net, mu, sd, raw):
        net.eval()
        with torch.inference_mode():
            z = torch.cat(
                [net(((block - mu) / sd).clamp(-12, 12)) for block in raw.flatten(0, 1).split(128)]
            )
        return z.cpu().numpy().reshape(len(raw), 10, 96)

    def original(path):
        w = load_file(path, device=str(device))
        mu, sd = w.pop("input_mean"), w.pop("input_scale")
        net = TrialObjectEncoder().to(device)
        net.load_state_dict(w, strict=True)
        return apply(net, mu, sd, eeg)

    deep416, deep512 = original(checkpoint416), original(checkpoint512)
    if old:
        with np.load(data / "trial-object-encoder/generic/embeddings.npz") as z:
            # A full bank is batched in 128s. Changed batch boundaries permit GPU rounding.
            check(
                np.array_equal(held, z["ids"])
                and np.allclose(
                    deep416[np.searchsorted(ids, held)], z["features"], rtol=0, atol=2e-5
                ),
                "Historical held-out representations differ",
            )
    oof = []
    fold = np.full(416, -1)
    for f, which in enumerate(np.array_split(np.random.default_rng(260564).permutation(416), 4)):
        fold[which] = f
        ix = ti[~np.isin(np.arange(416), which)]
        check(
            len(ix) == 312 and not np.isin(ids[ix], held).any(),
            "OOF fit overlaps readout evaluation",
        )
        raw = eeg[ix]
        sd, mu = torch.std_mean(raw, dim=(0, 1, 2, 4), unbiased=False)
        sd = sd.clamp_min(1e-5)[:, None]
        mu = mu[:, None]
        raw.sub_(mu).div_(sd).clamp_(-12, 12)
        model, _ = fit(raw, groups[ix], {"seed": 260553}, device)
        oof.append(apply(model, mu, sd, eeg[ti]))
        print(sub, "inner", f + 1, "of 4", flush=True)
    check(
        sha(mix / "decisions.npz") == d["people"][sub]["mixed_decisions"],
        "Different target EEG representations",
    )
    with np.load(mix / "decisions.npz") as z:
        target_rows = z["embedding_rows"]
        target = z["deep_embeddings"]
    check(
        target.shape == (120, 96) and not set(target_rows // 10) & set(ids),
        "Target object contamination",
    )
    out = dest / "features.npz"
    frozen_npz(
        out,
        dict(
            ids=ids,
            training=train,
            held=held,
            inner_folds=fold,
            deep416=deep416,
            deep512=deep512,
            deep_oof=np.stack(oof),
            target_rows=target_rows,
            target_deep=target,
        ),
    )
    freeze(
        dest / "receipt.json",
        source_receipt
        | dict(
            packet_sha256=sha(out),
            inner_train_classes=312,
            inner_held_classes=104,
            inner_folds=4,
            trial_fit_seed=260553,
            fold_seed=260564,
        ),
    )
    print(sub, "features ready", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", choices=["sub-04", "sub-05", "sub-01"], required=True)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/personal-pose"))
    p.add_argument("--device", default="cuda")
    features(p.parse_args())
