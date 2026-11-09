"""Resumable full-manifest independent image evaluation; no query subsampling."""
from image_evaluation import *
from pathlib import Path
import argparse
import gc
import json
import os
import time


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def pack_detections(rows):
    n=max(len(r['scores']) for r in rows)
    scores=torch.zeros(len(rows),n);boxes=torch.zeros(len(rows),n,4)
    labels=torch.zeros(len(rows),n,dtype=torch.int16);lengths=torch.zeros(len(rows),dtype=torch.int16)
    for i,row in enumerate(rows):
        k=len(row['scores']);lengths[i]=k
        if not k:continue
        scores[i,:k]=torch.tensor(row['scores']);labels[i,:k]=torch.tensor(row['labels'],dtype=torch.int16)
        boxes[i,:k]=torch.tensor(row['bboxes'],dtype=torch.float32)
    return dict(scores=scores,boxes=boxes,labels=labels,lengths=lengths)


def unpack_detection(shard,index):
    n=int(shard['lengths'][index]);return dict(scores=shard['scores'][index,:n].tolist(),
        labels=shard['labels'][index,:n].long().tolist(),bboxes=shard['boxes'][index,:n].tolist())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--real-gallery',action='store_true');args=parser.parse_args()
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(1729)
    manifest=Path(args.manifest);rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    paths=[Path(row['image_path'] if args.real_gallery else row['generated_path']) for row in rows]
    keys=[row['stimulus_id'] if args.real_gallery else row['query_id'] for row in rows]
    assert len(set(keys))==len(keys)
    for row,path in zip(rows,paths):assert sha(path)==row['image_sha256' if args.real_gallery else 'generated_sha256']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=dict(manifest=str(manifest),manifest_sha256=sha(manifest),keys=keys,
        real_gallery=args.real_gallery,DINO_weight_sha256=DINO_SHA256,
        dino_preprocessing='Historical routes CPU resize real/CUDA resize generated; 256 bilinear antialias; layer11 CLS+mean patch concatenated and normalized',
        SEED_source_commit='5327a134b90b04fb1b37066d98b309c9cc533a02',
        ObjectF1_grid='public code np.arange(.05,.95,.05); area50; preserve empty detections convention',
        semantic_model_receipts=[sha(ROOT/'audit/seed_hf_acquisition_completion.json'),sha(ROOT/'audit/seed_direct_assets_receipt.json')])
    frozen=out/'config.json'
    if frozen.exists():assert json.loads(frozen.read_text())==config
    else:frozen.write_text(json.dumps(config,indent=2)+'\n')
    stages=['DINO','EffNet','CapSim','ObjectF1']
    durations={}
    for stage in stages:
        tick=time.monotonic();folder=out/stage;folder.mkdir(exist_ok=True)
        starts=list(range(0,len(paths),256))
        if all((folder/f'{i:08d}.pt').exists() for i in starts):continue
        if stage=='DINO':model=dino_model()
        elif stage=='EffNet':model,preprocess=effnet_model()
        elif stage=='CapSim':processor,model,text,info=caption_models()
        else:model,classes,cfg=detection_model()
        for start in starts:
            target=folder/f'{start:08d}.pt';stop=min(start+256,len(paths))
            if target.exists():
                old=torch.load(target,weights_only=True);assert old['keys']==keys[start:stop];continue
            batch=paths[start:stop]
            if stage=='DINO':values=dict(features=dino_features(batch,model,real_images=args.real_gallery))
            elif stage=='EffNet':values=dict(features=torch.from_numpy(effnet_features(batch,model,preprocess)))
            elif stage=='CapSim':
                captions,features=caption_features(batch,processor,model,text)
                values=dict(features=torch.from_numpy(features),captions=captions)
            else:values=pack_detections(detection_features(batch,model,classes,batch_size=4))
            temp=target.with_name(target.name+'.tmp');torch.save(values|dict(keys=keys[start:stop]),temp);os.replace(temp,target)
            state=dict(status='RUNNING',stage=stage,completed=stop,expected=len(paths),seconds=time.monotonic()-started)
            (out/'progress.json').write_text(json.dumps(state)+'\n');print(json.dumps(state),flush=True)
        durations[stage]=time.monotonic()-tick
        del model
        if stage=='CapSim':del processor,text
        gc.collect();torch.cuda.empty_cache()
    completion=dict(status='COMPLETED',queries=len(paths),manifest_sha256=sha(manifest),seconds=time.monotonic()-started,
        stage_seconds_this_attempt=durations,all_queries_used=True,query_subsampling=False)
    (out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n');print(json.dumps(completion),flush=True)


if __name__=='__main__':main()
