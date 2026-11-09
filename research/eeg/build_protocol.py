"""Freeze full concept partitions and every query ID before new results exist."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SEED = 1729
VERSION = 'eeg_full10_concept_protocol_v1'


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, raw):
    if path.exists():
        assert path.read_bytes() == raw, ('Protected manifest already exists with different contents', path)
    else:
        with path.open('xb') as f:
            f.write(raw)


def main():
    audit = json.loads((ROOT / 'audit/data_audit.json').read_text())
    assert audit['images_verified'] == 16740 and audit['train_test_concepts_disjoint']
    assert sha(ROOT / 'audit/image_manifest.csv') == audit['image_manifest_sha256']
    with (ROOT / 'audit/image_manifest.csv').open() as f:
        images = list(csv.DictReader(f))
    train = [r for r in images if r['dataset_split'] == 'train']
    concepts = sorted({r['concept_id'] for r in train}, key=lambda c: hash_text(f'{SEED}|partition|{c}'))
    assert len(concepts) == 1654
    t = concepts[:1323]; d = concepts[1323:1488]; v = concepts[1488:]
    assert (len(t), len(d), len(v)) == (1323, 165, 166)
    assert not set(t) & set(d) and not set(t) & set(v) and not set(d) & set(v)
    roles = {c: role for role, group in [('T', t), ('D', d), ('V', v)] for c in group}
    folds = {c: i % 2 for i, c in enumerate(sorted(t, key=lambda c: hash_text(f'{SEED}|oof|{c}')))}
    calibration = sorted(d, key=lambda c: hash_text(f'{SEED}|calibration|{c}'))
    dfit = set(calibration[:82]); dcheck = set(calibration[82:])
    protocol = dict(version=VERSION, training_seed=SEED, number_of_training_seeds=1,
        subjects=[f'sub-{i:02d}' for i in range(1,11)],
        train_concepts={role:sorted(group) for role,group in [('T',t),('D',d),('V',v)]},
        image_rows={role:[int(r['array_row']) for r in train if roles[r['concept_id']] == role] for role in ['T','D','V']},
        oof_fold_by_concept=folds,
        d_fit_concepts=sorted(dfit), d_check_concepts=sorted(dcheck),
        expected_queries=dict(T=132300,D=16500,V=16600,official_benchmark=2000),
        expected_images_per_method=dict(full_V_R4=16600,official_R4=2000,official_R1=2000,official_R80=2000),
        real_gallery_unique_images=dict(V=1660,official_benchmark=200),
        image_manifest_sha256=audit['image_manifest_sha256'],
        final_independent_H=[], official_test_status='previously exposed benchmark; not independent confirmation',
        previous_exposure='Historical models and pilots used training concepts and official 200 stimuli; new partitions do not erase that exposure',
        alignment_limit=audit['event_alignment_limit'],
        train_eeg_protocol='mean of all four physical training repetitions; all 16540 training stimuli partitioned by concept',
        official_repro_scope='official develop mean4 training option; original paper individual-trial training is a separate unmatched protocol',
        official_test_protocols=dict(R1=[0],R4=[0,1,2,3],R80=list(range(80))),
        query_seed_rule='int(first16hex(SHA256(seed|subject|stimulus_filename|R|protocol_version)),16) modulo (2**63-1)',
        images_per_query=1, best_of_N=False, no_test_gallery_at_inference=True,
        dino_direction='generated query -> fixed unique real-image gallery; exclude its correct image; ties=0.5',
        dino_backbone='existing frozen DINOv3 ViT-S/16 evaluator; this is DINO two-alternative forced choice, not a claim of DINOv2 weights',
        final_lock_status='SPLITS_AND_QUERY_RNG_LOCKED; generator resolution/precision/cost and evaluator gates pending S0 profiling',
        source_supervision_policy='encoder, prior and adapters train on T only; Dfit calibrates, Dcheck validates error model, V selects methods; no prior training on D/V',
        full_fit_policy='After method and epoch rules lock, allow all official training concepts for every matched strong method, rebuild OOF caches; no new training seed')
    freeze(ROOT / 'configs/split_manifest.json', (json.dumps(protocol, indent=2, sort_keys=True)+'\n').encode())
    fields = ['query_id','subject_id','dataset_split','role','calibration_role','stimulus_id','concept_id',
              'array_row','repeat_protocol','generation_seed','oof_fold','image_zip','image_member','image_sha256']
    query_path = ROOT / 'audit/query_manifest_R4.csv'
    if not query_path.exists():
        with query_path.open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for subject in protocol['subjects']:
                for image in images:
                    concept = image['concept_id']; role = roles[concept] if image['dataset_split']=='train' else 'benchmark_exposed'
                    stimulus = image['stimulus_id']; seed = int(hash_text(f'{SEED}|{subject}|{stimulus}|R4|{VERSION}')[:16],16) % (2**63-1)
                    writer.writerow(dict(query_id=f'{subject}/{stimulus}/R4',subject_id=subject,
                        dataset_split=image['dataset_split'],role=role,
                        calibration_role='fit' if concept in dfit else ('check' if concept in dcheck else ''),
                        stimulus_id=stimulus,concept_id=concept,array_row=image['array_row'],repeat_protocol='R4',
                        generation_seed=seed,oof_fold=folds.get(concept,''),image_zip=image['image_zip'],
                        image_member=image['image_member'],image_sha256=image['image_sha256']))
    with query_path.open() as f:
        query_rows=list(csv.DictReader(f))
    assert len(query_rows)==167400 and len({r['query_id'] for r in query_rows})==167400
    counts={role:sum(r['role']==role for r in query_rows) for role in ['T','D','V','benchmark_exposed']}
    assert counts==dict(T=132300,D=16500,V=16600,benchmark_exposed=2000)
    receipt=dict(status='FULL_PARTITIONS_LOCKED',protocol_version=VERSION,counts=counts,
                 split_manifest_sha256=sha(ROOT/'configs/split_manifest.json'),
                 query_manifest_sha256=sha(query_path),new_image_evaluations_before_lock=0,
                 independent_H_established=False)
    freeze(ROOT/'audit/protocol_manifest_receipt.json',(json.dumps(receipt,indent=2)+'\n').encode())
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
