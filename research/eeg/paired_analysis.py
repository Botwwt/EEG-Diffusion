"""Paired full-query differences with crossed subject/concept uncertainty.

This script cannot train, generate, tune checkpoints, or unlock the final test.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
METRICS = ['DINO2AFC', 'DINO_cosine', 'DINO_Top1', 'DINO_Top5',
           'EffNet', 'CapSim', 'ObjectF1', 'SEED', 'pixel_std',
           'near_black_fraction', 'near_white_fraction', 'blank_proxy']
PAIR_FIELDS = ['query_id', 'subject', 'stimulus_id', 'concept_id', 'array_row',
    'repeat_protocol', 'partition', 'image_seed', 'prior_seed',
    'initial_image_noise_sha256', 'target_image_sha256', 'target_image_zip',
    'target_image_member', 'encoder_sha256', 'generator_revision',
    'steps', 'CFG', 'IP_scale', 'UNet_NFE', 'image_interface_config_sha256']


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def load_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    result = {r['query_id']: r for r in rows}
    assert len(result) == len(rows), 'Repeated query IDs'
    return result


def load_metrics(path):
    with Path(path).open() as file:
        rows = list(csv.DictReader(file))
    result = {r['query_id']: r for r in rows}
    assert len(result) == len(rows), 'Repeated query metric IDs'
    return result


def scalar(value):
    return float(value == 'True') if value in ['True', 'False'] else float(value)


def paired_summary(values, subjects, concepts, draws=1000, seed=1729):
    """values has one row/query and one column/metric; positive=candidate-reference."""
    unique_s = sorted(set(subjects));unique_c = sorted(set(concepts))
    si = {s:i for i,s in enumerate(unique_s)};ci = {c:i for i,c in enumerate(unique_c)}
    sums = np.zeros((len(si), len(ci), values.shape[1]), dtype=np.float64)
    count = np.zeros((len(si), len(ci)), dtype=np.int64)
    for k,(s,c) in enumerate(zip(subjects, concepts)):
        sums[si[s], ci[c]] += values[k];count[si[s], ci[c]] += 1
    assert (count > 0).all() and np.unique(count).size == 1, 'Require complete balanced paired partitions'
    cell = sums/count[:,:,None]
    np.testing.assert_allclose(cell.mean((0,1)), values.mean(0), rtol=1e-12, atol=1e-12)
    generator = np.random.default_rng(seed);boot = np.empty((draws, values.shape[1]))
    for k in range(draws):
        ss = generator.integers(len(si), size=len(si));cc = generator.integers(len(ci), size=len(ci))
        boot[k] = cell[ss[:,None], cc[None,:]].mean((0,1))
    return dict(mean=values.mean(0).tolist(), CI95=np.quantile(boot,[.025,.975],axis=0).T.tolist(),
                per_subject={s:cell[i].mean(0).tolist() for i,s in enumerate(unique_s)},
                subjects=len(si), concept_clusters=len(ci), queries_per_subject_concept=int(count[0,0]))


def main():
    parser=argparse.ArgumentParser()
    for arg in ['reference_manifest','candidate_manifest','reference_metrics','candidate_metrics','output']:
        parser.add_argument('--'+arg.replace('_','-'), required=True)
    args=parser.parse_args();started=time.monotonic()
    plan_path=ROOT/'configs/paired_statistics_plan.json';plan=json.loads(plan_path.read_text())
    a=load_manifest(args.reference_manifest);b=load_manifest(args.candidate_manifest)
    ma=load_metrics(args.reference_metrics);mb=load_metrics(args.candidate_metrics)
    assert a.keys()==b.keys()==ma.keys()==mb.keys(), 'Never compare unequal subsets'
    keys=sorted(a);assert len(keys) in [16600,2000]
    assert len({a[k]['subject'] for k in keys})==10
    assert len({a[k]['repeat_protocol'] for k in keys})==1
    for key in keys:
        for field in PAIR_FIELDS:
            assert a[key][field]==b[key][field], (key,field,'Pairing or sampling contract mismatch')
        for manifest,metrics in [(a,ma),(b,mb)]:
            assert manifest[key]['image_seed']==int(metrics[key]['image_seed'])
            assert manifest[key]['stimulus_id']==metrics[key]['stimulus_id']
            assert manifest[key]['generated_path']==metrics[key]['generated_path']
    x=np.array([[scalar(ma[k][m]) for m in METRICS] for k in keys])
    y=np.array([[scalar(mb[k][m]) for m in METRICS] for k in keys])
    assert np.isfinite(x).all() and np.isfinite(y).all()
    delta=y-x;subjects=[a[k]['subject'] for k in keys];concepts=[a[k]['concept_id'] for k in keys]
    stat=paired_summary(delta,subjects,concepts,plan['bootstrap_draws'],plan['bootstrap_seed'])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'summary.json').exists(), 'Preserve completed comparisons'
    with (out/'paired_per_query.csv').open('x',newline='') as file:
        columns=['query_id','subject','stimulus_id','concept_id']+[f'{m}_{v}' for m in METRICS for v in ['reference','candidate','delta']]
        writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader()
        for i,k in enumerate(keys):
            row={f:a[k][f] for f in columns[:4]}
            for j,m in enumerate(METRICS):
                row.update({f'{m}_reference':x[i,j],f'{m}_candidate':y[i,j],f'{m}_delta':delta[i,j]})
            writer.writerow(row)
    with (out/'per_subject.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['subject']+METRICS);writer.writeheader()
        writer.writerows(dict(subject=s,**dict(zip(METRICS,v))) for s,v in stat['per_subject'].items())
    fields=['query_id','subject','stimulus_id','concept_id','DINO2AFC_delta']
    ordered=sorted(range(len(keys)),key=lambda i:(-delta[i,0],keys[i]))
    for name,positions in [('largest_gain',ordered),('largest_decline',list(reversed(ordered)))]:
        with (out/f'{name}_all_cases.csv').open('x',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
            writer.writerows(dict(**{f:a[keys[i]][f] for f in fields[:-1]},DINO2AFC_delta=delta[i,0]) for i in positions)
    report=dict(status='COMPLETED_PAIRED_FULL_PARTITION',queries=len(keys),subjects=stat['subjects'],
        concept_clusters=stat['concept_clusters'],queries_per_subject_concept=stat['queries_per_subject_concept'],
        reference=dict(method=a[keys[0]]['method'],intervention=a[keys[0]]['intervention']),
        candidate=dict(method=b[keys[0]]['method'],intervention=b[keys[0]]['intervention']),
        partition=a[keys[0]]['partition'],repeat_protocol=a[keys[0]]['repeat_protocol'],
        metrics={m:dict(reference=float(x[:,j].mean()),candidate=float(y[:,j].mean()),
            paired_delta=stat['mean'][j],CI95_subject_concept=stat['CI95'][j]) for j,m in enumerate(METRICS)},
        plan_sha256=sha(plan_path),source_sha256=sha(__file__),
        inputs={name:dict(path=str(getattr(args,name)),sha256=sha(getattr(args,name))) for name in
            ['reference_manifest','candidate_manifest','reference_metrics','candidate_metrics']},
        criteria=dict(DINO_practical_gain_met=stat['mean'][0]>=plan['practical_DINO_gain'],
            DINO_descriptive_CI_lower_positive=stat['CI95'][0][0]>0,
            SEED_mean_change_positive=stat['mean'][METRICS.index('SEED')]>0,
            ObjectF1_mean_change_positive=stat['mean'][METRICS.index('ObjectF1')]>0,
            quality_noninferiority_established=False,independent_confirmation=False,SOTA_established=False),
        training_seed_variability_covered=False,
        limits=['One training seed; intervals describe subject/concept resampling only.',
            'V intervals are descriptive after selection; the official benchmark was previously exposed.',
            'DINO2AFC gain is not correct-object percentage; gallery size is fixed within this comparison.',
            'The two methods have equal image UNet NFE; any extra prior evaluations and total time must be reported separately.',
            'Positive blank/near-black/near-white changes can indicate worse quality; no human ratings are inferred.'],
        seconds=time.monotonic()-started)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
