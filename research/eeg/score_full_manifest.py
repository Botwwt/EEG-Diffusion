"""Full query scoring by stimulus ID with frozen real galleries and semantic models."""
from image_evaluation import *
from extract_image_features import unpack_detection
from locked_metrics import fixed_gallery_metrics,VERSION
import argparse
import csv
import json
import time


def load_stage(directory,stage):
    pieces=[torch.load(p,weights_only=True,map_location='cpu') for p in sorted((Path(directory)/stage).glob('*.pt'))]
    keys=sum([p['keys'] for p in pieces],[]);assert len(set(keys))==len(keys)
    if stage=='ObjectF1':values=[unpack_detection(p,i) for p in pieces for i in range(len(p['keys']))]
    else:values=torch.cat([p['features'] for p in pieces])
    return keys,values


def interval(rows,key):
    subjects=sorted({r['subject'] for r in rows});concepts=sorted({r['concept_id'] for r in rows})
    # All subjects share concepts; images within a concept are never independent clusters.
    matrix=np.empty((len(subjects),len(concepts)))
    for i,s in enumerate(subjects):
        for j,c in enumerate(concepts):matrix[i,j]=np.mean([r[key] for r in rows if r['subject']==s and r['concept_id']==c])
    generator=np.random.default_rng(1729);draws=[]
    for _ in range(1000):
        si=generator.integers(len(subjects),size=len(subjects));ci=generator.integers(len(concepts),size=len(concepts))
        draws.append(matrix[si[:,None],ci[None,:]].mean())
    return dict(mean=float(matrix.mean()),CI95_subject_and_concept_bootstrap=np.quantile(draws,[.025,.975]).tolist(),
                training_seed_variability_covered=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--features',required=True)
    parser.add_argument('--gallery-manifest',required=True);parser.add_argument('--gallery-features',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(2);started=time.monotonic()
    rows=[json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    gallery=[json.loads(line) for line in Path(args.gallery_manifest).read_text().splitlines()]
    assert len(rows) in [16600,2000] and len({r['subject'] for r in rows})==10
    for directory in [args.features,args.gallery_features]:assert json.loads((Path(directory)/'completion.json').read_text())['status']=='COMPLETED'
    qkeys=[r['query_id'] for r in rows];gkeys=[r['stimulus_id'] for r in gallery]
    vectors={};g_vectors={}
    for stage in ['DINO','EffNet','CapSim','ObjectF1']:
        keys,data=load_stage(args.features,stage);lookup={key:i for i,key in enumerate(keys)}
        vectors[stage]=data[[lookup[k] for k in qkeys]] if isinstance(data,torch.Tensor) else [data[lookup[k]] for k in qkeys]
        keys,data=load_stage(args.gallery_features,stage);lookup={key:i for i,key in enumerate(keys)}
        g_vectors[stage]=data[[lookup[k] for k in gkeys]] if isinstance(data,torch.Tensor) else [data[lookup[k]] for k in gkeys]
    positions={key:i for i,key in enumerate(gkeys)};correct=[positions[r['stimulus_id']] for r in rows]
    metric=fixed_gallery_metrics(vectors['DINO'],g_vectors['DINO'],[r['stimulus_id'] for r in rows],gkeys)
    self_check=fixed_gallery_metrics(g_vectors['DINO'],g_vectors['DINO'],gkeys,gkeys)
    random_generator=np.random.default_rng(1729)
    random_ids=[gkeys[i] for i in random_generator.integers(len(gkeys),size=len(rows))]
    shuffled=fixed_gallery_metrics(vectors['DINO'],g_vectors['DINO'],random_ids,gkeys)
    eff=correlation_similarity(vectors['EffNet'].numpy(),g_vectors['EffNet'][correct].numpy())
    cap=(vectors['CapSim']*g_vectors['CapSim'][correct]).sum(-1).numpy()
    values=[]
    for i,row in enumerate(rows):
        target=gallery[correct[i]];assert row['target_image_sha256']==target['image_sha256']
        f1=object_f1_pair(vectors['ObjectF1'][i],g_vectors['ObjectF1'][correct[i]])
        q=quality_statistics(row['generated_path'])
        out=dict(query_id=row['query_id'],subject=row['subject'],stimulus_id=row['stimulus_id'],concept_id=row['concept_id'],
            EEG_source_stimulus_id=row['EEG_source_stimulus_id'],target_image_path=target['image_path'],
            generated_path=row['generated_path'],score_target_path=target['image_path'],
            image_seed=row['image_seed'],DINO2AFC=float(metric['two_afc'][i]),DINO_cosine=float(metric['correct_cosine'][i]),
            DINO_Top1=int(metric['top1'][i]),DINO_Top5=int(metric['top5'][i]),EffNet=float(eff[i]),CapSim=float(cap[i]),
            ObjectF1=f1,SEED=(f1+float(eff[i])+float(cap[i]))/3,
            GT_has_detected_COCO_gender_category=bool(detected_categories(g_vectors['ObjectF1'][correct[i]],.05)),
            generated_has_detected_COCO_gender_category=bool(detected_categories(vectors['ObjectF1'][i],.05)),**q)
        values.append(out)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'per_query.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    metrics=['DINO2AFC','DINO_cosine','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED']
    summary={key:interval(values,key) for key in metrics}
    per_subject={s:{key:float(np.mean([r[key] for r in values if r['subject']==s])) for key in metrics} for s in sorted({r['subject'] for r in values})}
    report=dict(status='COMPLETED',queries=len(rows),subjects=10,gallery_images=len(gallery),method=rows[0]['method'],
        intervention=rows[0]['intervention'],repeat_protocol=rows[0]['repeat_protocol'],metric_version=VERSION,
        metric_source_sha256=hashlib.sha256((ROOT/'locked_metrics.py').read_bytes()).hexdigest(),
        summary=summary,per_subject=per_subject,
        real_image_self_pair_DINO2AFC=float(self_check['two_afc'].mean()),random_label_DINO2AFC=float(shuffled['two_afc'].mean()),
        blank_proxy_count=sum(r['blank_proxy'] for r in values),
        GT_object_detector_empty_count=sum(not r['GT_has_detected_COCO_gender_category'] for r in values),
        generated_object_detector_empty_count=sum(not r['generated_has_detected_COCO_gender_category'] for r in values),
        human_quality_rating_performed=False,
        interpretive_limits=['DINO2AFC is relative gallery preference, not correct-object percentage.',
            'ObjectF1 uses COCO+gender82 classes and the official empty-detection convention; it is not THINGS1654-class recognition accuracy.',
            'A single training seed; intervals resample subjects/concepts only; previously exposed official benchmark is not independent H.'],
        seconds=time.monotonic()-started)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['summary','per_subject','interpretive_limits']}),flush=True)


if __name__=='__main__':main()
