"""Verify a registered ten-person comparison against individual layouts."""
import argparse
from hashlib import sha256
from importlib import util
import io
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

import numpy as np

PEOPLE = tuple(f"sub-{person:02d}" for person in range(1, 11))
PAIRED_FIELDS = (
    "budget", "development_split_sha256", "draws", "eeg_donor", "layout_mode",
    "layout_source", "n", "physical_eeg", "prompt", "render_code_sha256",
    "repeats", "row", "sampling", "seeds", "semantic_EEG", "split_sha256",
    "subject", "unet_calls_per_draw",
)


def load_aggregate(source):
    spec = util.spec_from_file_location("registered_paper_aggregate", source)
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_member(archive, name):
    entry = archive.getmember(name)
    if not entry.isfile():
        raise ValueError(f"Expected a regular archive member at {name}")
    return archive.extractfile(entry).read()


def digest(blob):
    return sha256(blob).hexdigest()


def read_registered_panel(archive):
    split_bytes = read_member(archive, "registry/paper_split.json")
    dev_bytes = read_member(archive, "registry/paper_development.json")
    split, dev = json.loads(split_bytes), json.loads(dev_bytes)
    concepts = np.load(io.BytesIO(read_member(archive, "registry/train_concept.npy")), allow_pickle=False)
    rows = np.array(dev["dev_rows"])
    if len(rows) != 200 or len(np.unique(rows)) != 200:
        raise ValueError("The registered development panel needs all 200 distinct images")
    held = np.unique(concepts[rows])
    counts = np.unique(concepts[rows], return_counts=True)[1]
    if not np.array_equal(np.sort(dev["concepts"]), held) or len(held) != 20 or np.any(counts != 10):
        raise ValueError("The development registration differs from the stored concepts")
    other = np.concatenate((concepts[split["train_rows"]], concepts[split["fresh_rows"]]))
    if np.intersect1d(held, other).size:
        raise ValueError("Development concepts overlap training or independent replication")
    return rows, digest(split_bytes), digest(dev_bytes)


def read_case(archive, arm, subject, output, aggregate, panel, checkpoints):
    prefix = f"{arm}/{subject}"
    folder = output / prefix
    folder.mkdir(parents=True)
    for name in ("render.json", "scores.json", "observations.npz"):
        (folder / name).write_bytes(read_member(archive, f"{prefix}/{name}"))
    render_bytes = (folder / "render.json").read_bytes()
    render = json.loads(render_bytes)
    names, rows, concepts, orig, values, score, sources = aggregate.read_scores(folder, "dev")

    registered, split_sha, dev_sha = panel
    expected = {
        "arm": "d2", "training": "shared", "budget": "dev-r4", "n": 200,
        "subject": int(subject[-2:]), "draws": 1, "unet_calls_per_draw": 4, "prompt": "",
        "repeats": [[0, 1, 2, 3]], "row": registered.tolist(), "eeg_donor": list(range(200)),
        "seeds": aggregate.panel_seeds("dev", 200).tolist(), "checkpoint_sha256": checkpoints[arm],
        "split_sha256": split_sha, "development_split_sha256": dev_sha,
    }
    for key, expected_value in expected.items():
        if render.get(key) != expected_value:
            raise ValueError(f"{prefix} disagrees with its registered {key}")
    if score["render_sha256"] != digest(render_bytes):
        raise ValueError(f"{prefix} score belongs to a different render receipt")
    hashes = render["pixels_sha256"]
    if len(hashes) != 200 or not all(aggregate.valid_sha(item) for item in hashes):
        raise ValueError(f"{prefix} must bind 200 PNG hashes")
    aggregate.check_scored_images(score, hashes, sources)
    if not np.array_equal(rows, np.sort(registered)):
        raise ValueError(f"{prefix} observations cover different images")
    sources["render_sha256"] = digest(render_bytes)
    return render, names, rows, concepts, orig, values, sources


def compare(packet, aggregate, output, challenger="group-mean"):
    if challenger == "individual":
        raise ValueError("Select a challenger different from the individual-layout control")
    arms = ("individual", challenger)
    cases = {arm: [] for arm in arms}
    with tarfile.open(packet, "r:*") as archive, TemporaryDirectory(dir=output.parent) as temporary:
        panel = read_registered_panel(archive)
        checkpoints = json.loads(read_member(archive, "registry/checkpoints.json"))
        if set(checkpoints) != set(arms) or len(set(checkpoints.values())) != 2:
            raise ValueError("Expected two distinct registered condition checkpoints")
        if not all(aggregate.valid_sha(value) for value in checkpoints.values()):
            raise ValueError("Checkpoint hashes are invalid")
        for subject in PEOPLE:
            for arm in arms:
                cases[arm].append(read_case(archive, arm, subject, Path(temporary), aggregate, panel, checkpoints))
            control, candidate = (cases[arm][-1] for arm in arms)
            for key in PAIRED_FIELDS:
                if control[0][key] != candidate[0][key]:
                    raise ValueError(f"{subject} has unmatched {key}")
            for a, b in zip(control[1:5], candidate[1:5]):
                if not np.array_equal(a, b):
                    raise ValueError(f"{subject} has unmatched stimulus or metric identifiers")

    template = cases[arms[0]][0]
    for arm in arms:
        for case in cases[arm]:
            if not all(np.array_equal(a, b) for a, b in zip(template[1:5], case[1:5])):
                raise ValueError("The participant tables disagree on stimuli or metrics")
    values = {arm: np.stack([case[5] for case in cases[arm]]) for arm in arms}
    totals, people = aggregate.describe(values[arms[1]], values[arms[0]], PEOPLE, template[3],
                                        "dev", template[1], 10000, 20261127)
    report = {
        "comparison": f"{challenger} minus individual",
        "panel": "dev-r4",
        "n_people": len(PEOPLE),
        "n_per_arm": sum(values[arm].shape[0] * values[arm].shape[1] for arm in arms) // 2,
        "n_concepts": 20,
        "query_input": "Same four physical EEG presentations for both arms",
        "seed": 20261127,
        "bootstrap_resamples": 10000,
        "bootstrap_units": "Crossed participant and whole concept resampling",
        "checkpoint_sha256": checkpoints,
        "archive_sha256": digest(packet.read_bytes()),
        "metrics": totals,
        "per_person": people,
        "sources": {arm: dict(zip(PEOPLE, (case[6] for case in cases[arm]))) for arm in arms},
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for metric in ("generated_name1854", "clip_image_top1", "clip_within10", "ssim"):
        result = totals[metric]
        print(metric, result["mean_b"], result["mean_a"], result["crossed_ci95"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--aggregate-source", type=Path, required=True)
    parser.add_argument("--challenger", default="group-mean")
    opts = parser.parse_args()
    compare(opts.archive, load_aggregate(opts.aggregate_source), opts.output, opts.challenger)
