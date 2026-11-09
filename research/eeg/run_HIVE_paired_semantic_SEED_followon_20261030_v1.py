"""Conditional secondary evaluation of the SAME fixed400/all7 saved images."""
import argparse,csv,gc,hashlib,io,json,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
from brainhive_paired_semantic_study_contract_CPU_20261030_v1 import ARMS,IMAGE_ARMS,atomic,checked,read,ref

def load(path):
    s=read(path);assert s['status']=='REGISTERED_CONDITIONAL_ALL7_FULL400_SEED_AND_LOWLEVEL_FOLLOWON'
    for p in s['source_pins']:checked(p)
    e=read(checked(s['upstream_image_execution']));up=read(checked(s['upstream_protocol']));root=Path(s['output_root']);image_root=Path(e['output_root'])
    d=read(checked(e['D09_completion']));inputs=read(checked(d['inputs']));meta=read(checked(s['target_metadata']))['Dcheck'];q=read(checked(inputs['panel']))['Dcheck'];assert len(q)==len(meta)==400
    return s,e,up,root,image_root,q,meta
def runtime(s):
    sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C;C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
def promote(s,image_root):
    upstream=read(Path(s['upstream_output_root'])/'controller_completion.json');assert upstream['registration']==s['upstream_protocol'] and upstream['status']=='COMPLETE_FIXED_FULL_T_AND_NATIVE_IMAGE_STUDY'
    audit=read(image_root/'independent_images400_CPU_audit.json');score=read(checked(audit['score']));assert audit['execution']==s['upstream_image_execution'] and audit['all_images_verified'] and audit['images']==2800
    assert upstream['final_audit']==audit and audit['decision']==score['decision'] and audit['decision']['action']=='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION'
    return audit
def rows_for(root,image_root,q,arm):
    return read(root/'targets.json') if arm=='target' else [read(image_root/'records'/f'{i:03d}_{arm}.json') for i in range(400)]
def prepare(path):
    s,e,up,root,image_root,q,meta=load(path);promote(s,image_root);root.mkdir(exist_ok=True);targets=[];archives={};folder=root/'target_bytes';folder.mkdir(exist_ok=True)
    for i,(query,row) in enumerate(zip(q,meta)):
        assert all(row[k]==v for k,v in query.items());z=row['image_zip']
        if z not in archives:archives[z]=zipfile.ZipFile(z)
        b=archives[z].read(row['image_member']);assert len(b)==int(row['image_bytes']) and hashlib.sha256(b).hexdigest()==row['image_sha256'];p=folder/f'{i:03d}.jpg';assert not p.exists();p.write_bytes(b)
        targets.append(dict(**query,query_index=i,image_path=str(p),image_sha256=row['image_sha256'],source_zip=z,source_member=row['image_member']))
    for archive in archives.values():archive.close()
    atomic(root/'targets.json',targets);atomic(root/'prepare_completion.json',dict(status='COMPLETE_EXACT_400_TARGET_BYTES_NO_PREPROCESS_CHANGE',targets=ref(root/'targets.json'),training_or_image_generation_calls=0,upstream_audit=ref(image_root/'independent_images400_CPU_audit.json')))

def features(args):
    s,e,up,root,image_root,q,meta=load(args.protocol);promote(s,image_root);runtime(s)
    import numpy as np,torch
    from PIL import Image
    import semantic_evaluator as E
    from extract_image_features import pack_detections
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==up['GPU_UUID']
    for p in s['asset_pins']:checked(p)
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    start=time.monotonic();memory=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
    attempt=root/'attempts'/f'{args.arm}_{args.stage}_{args.slot}';attempt.mkdir(parents=True,exist_ok=False)
    def budget():
        rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
        assert time.monotonic()-start<820 and rss-memory<10<<30 and torch.cuda.max_memory_allocated()<20<<30
        assert shutil.disk_usage(root).free>58<<30 and sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<s['output_cap_bytes']
    try:
        rows=rows_for(root,image_root,q,args.arm);paths=[]
        for i,row in enumerate(rows):
            assert row['query_id']==q[i]['query_id'] and all(row[k]==v for k,v in q[i].items())
            p=Path(row['image_path'] if args.arm=='target' else row['generated_path']);assert ref(p)['sha256']==row['image_sha256' if args.arm=='target' else 'generated_sha256']
            if args.arm!='target':assert row['execution']==s['upstream_image_execution'] and not row['normal_inference_target_access']
            paths.append(p)
        out=root/'features'/args.arm/args.stage;out.mkdir(parents=True,exist_ok=True)
        config=dict(protocol=ref(args.protocol),stage=args.stage,arm=args.arm,ordered_query_ids=[r['query_id'] for r in rows],source_image_sha256=[ref(p)['sha256'] for p in paths],chunks=32,model_math_unchanged=True)
        cp=out/'configuration.json'
        if cp.exists():assert read(cp)==config
        else:atomic(cp,config)
        if args.stage=='EffNet':model,preprocess=E.effnet_model()
        elif args.stage=='CapSim':processor,model,text,info=E.caption_models()
        else:model,classes,cfg=E.detection_model()
        for lo in range(0,400,32):
            budget();hi=min(lo+32,400);p=out/f'{lo:03d}.pt'
            if p.exists():
                certificate=read(out/f'{lo:03d}.json');checked(certificate['artifact']);assert certificate['configuration_sha256']==ref(cp)['sha256']
                old=torch.load(p,weights_only=True,map_location='cpu');assert old['configuration_sha256']==ref(cp)['sha256'] and old['query_ids']==config['ordered_query_ids'][lo:hi];continue
            if time.monotonic()-start>650:
                atomic(attempt/'completion.json',dict(status='COMPLETE_PRESERVED_FEATURE_PREFIX',full_stage=False,arm=args.arm,stage=args.stage,seconds=time.monotonic()-start));return
            if args.stage=='EffNet':values=dict(features=torch.from_numpy(E.effnet_features(paths[lo:hi],model,preprocess,batch_size=32)))
            elif args.stage=='CapSim':
                captions,vectors=E.caption_features(paths[lo:hi],processor,model,text,batch_size=16);values=dict(features=torch.from_numpy(vectors),captions=captions)
            else:values=pack_detections(E.detection_features(paths[lo:hi],model,classes,batch_size=4))
            values.update(query_ids=config['ordered_query_ids'][lo:hi],configuration_sha256=ref(cp)['sha256']);tmp=p.with_suffix('.pt.pending');assert not tmp.exists()
            with tmp.open('xb') as stream:torch.save(values,stream);stream.flush();os.fsync(stream.fileno())
            tmp.rename(p);atomic(out/f'{lo:03d}.json',dict(artifact=ref(p),configuration_sha256=ref(cp)['sha256'],query_ids=config['ordered_query_ids'][lo:hi]))
        shard_refs=[]
        for lo in range(0,400,32):
            certificate=read(out/f'{lo:03d}.json');checked(certificate['artifact']);assert certificate['configuration_sha256']==ref(cp)['sha256'] and certificate['query_ids']==config['ordered_query_ids'][lo:min(lo+32,400)];shard_refs.append(certificate['artifact'])
        complete=dict(status='COMPLETE_ALL400_FROZEN_SEMANTIC_FEATURES',arm=args.arm,stage=args.stage,queries=400,configuration=ref(cp),shards=shard_refs)
        atomic(out/'completion.json',complete);budget();atomic(attempt/'completion.json',dict(status='COMPLETE_FIXED_FEATURE_STAGE',full_stage=True,completion=ref(out/'completion.json'),seconds=time.monotonic()-start))
    except BaseException as ex:atomic(attempt/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise

def load_features(root,arm,stage,torch):
    from extract_image_features import unpack_detection
    folder=root/'features'/arm/stage;c=read(folder/'completion.json');assert c['queries']==400 and c['arm']==arm and c['stage']==stage;config=read(checked(c['configuration']));rows=[];values=[];captions=[]
    for pin in c['shards']:
        value=torch.load(checked(pin),map_location='cpu',weights_only=True);assert value['configuration_sha256']==c['configuration']['sha256'];rows+=value['query_ids']
        if stage=='ObjectF1':values.extend(unpack_detection(value,i) for i in range(len(value['query_ids'])))
        else:values.append(value['features'])
        if stage=='CapSim':captions+=value['captions']
    assert rows==config['ordered_query_ids'] and len(rows)==400
    return values if stage=='ObjectF1' else torch.cat(values).numpy()

def panels(root,image_root,q,indices,Image):
    canvas=Image.new('RGB',(8*192,len(indices)*218+35),'white');from PIL import ImageDraw
    draw=ImageDraw.Draw(canvas);labels=['target']+list(IMAGE_ARMS)
    for col,label in enumerate(labels):draw.text((col*192+3,4),label,fill='black')
    targets=read(root/'targets.json')
    for line,i in enumerate(indices):
        y=35+line*218;draw.text((3,y),f"{i:03d} {q[i]['stimulus_id']}",fill='black')
        paths=[targets[i]['image_path']]+[read(image_root/'records'/f'{i:03d}_{a}.json')['generated_path'] for a in IMAGE_ARMS]
        for col,p in enumerate(paths):
            with Image.open(p) as im:canvas.paste(im.convert('RGB').resize((188,188),Image.Resampling.LANCZOS),(col*192+2,y+24))
    p=root/'fixed6_query_all7_panel.png';assert not p.exists();canvas.save(p);return ref(p)

def score(path):
    s,e,up,root,image_root,q,meta=load(path);audit=promote(s,image_root);runtime(s);os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np,torch
    from PIL import Image
    import image_evaluation as E
    import score_low_level as Low
    from brainhive_paired_semantic_SEED_statistics_CPU_20261030_v1 import METRICS,statistics,next_action
    torch.set_num_threads(2);Low.numerical_checks();target={k:load_features(root,'target',k,torch) for k in ('EffNet','CapSim','ObjectF1')};targets=read(root/'targets.json');values={};counts={};start=time.monotonic()
    for arm in IMAGE_ARMS:
        generated={k:load_features(root,arm,k,torch) for k in target};eff=E.correlation_similarity(generated['EffNet'],target['EffNet']);cap=(generated['CapSim']*target['CapSim']).sum(-1).astype(np.float64);f1=np.array([E.object_f1_pair(a,b) for a,b in zip(generated['ObjectF1'],target['ObjectF1'])]);rows=[];empty=0
        for i in range(400):
            record=read(image_root/'records'/f'{i:03d}_{arm}.json');rgb,y=Low.pixels(record['generated_path']);trgb,ty,tm=Low.target_pixels(targets[i]['image_path'],targets[i]['image_sha256'])
            assert ref(record['generated_path'])['sha256']==record['generated_sha256'];empty+=not bool(E.detected_categories(generated['ObjectF1'][i],.05))
            rows.append([eff[i],cap[i],f1[i],(eff[i]+cap[i]+f1[i])/3,Low.pixcorr(rgb,trgb),Low.ssim_y(y,ty,tm)])
        values[arm]=np.array(rows,np.float64);counts[arm]=dict(generated_detector_empty=int(empty),all_queries=400)
    result=statistics(values,q,ARMS);np.savez_compressed(root/'all2800_secondary_metrics.npz',**values,concept_bootstrap_draws=result.pop('draws'))
    with (root/'all2800_secondary_metrics.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['arm','query_index','query_id','concept_id','stimulus_id']+list(METRICS));writer.writeheader()
        for arm in IMAGE_ARMS:
            for i,row in enumerate(q):writer.writerow(dict(arm=arm,query_index=i,**{k:row[k] for k in ('query_id','concept_id','stimulus_id')},**dict(zip(METRICS,values[arm][i]))))
    final=dict(status='COMPLETE_ALL7_FULL400_SEED_AND_LOWLEVEL_DEVELOPMENT_READOUT',protocol=ref(path),upstream_audit=ref(image_root/'independent_images400_CPU_audit.json'),all_queries=2800,**result,
        next_action=next_action(audit['decision'],result['contrasts']),target_detector_empty=sum(not E.detected_categories(v,.05) for v in target['ObjectF1']),generated_counts=counts,
        public_ObjectF1_empty_both_is_one=True,all_metrics_and_queries_retained=True,per_query=ref(root/'all2800_secondary_metrics.csv'),arrays=ref(root/'all2800_secondary_metrics.npz'),
        fixed_predeclared_qualitative=panels(root,image_root,q,e['fixed_qualitative400'],Image),seconds=time.monotonic()-start,official_R80=False,no_new_models_or_images=True,SOTA=False)
    assert not torch.cuda.is_initialized();atomic(root/'secondary_readout.json',final);print(json.dumps(dict(status=final['status'],queries=2800,next_action=final['next_action'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['prepare','features','score'],required=True);p.add_argument('--arm',choices=['target']+list(IMAGE_ARMS));p.add_argument('--stage',choices=['EffNet','CapSim','ObjectF1']);p.add_argument('--slot',type=int,default=0);a=p.parse_args();prepare(a.protocol) if a.mode=='prepare' else features(a) if a.mode=='features' else score(a.protocol)
