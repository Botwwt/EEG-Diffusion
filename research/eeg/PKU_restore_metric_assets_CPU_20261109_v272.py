"""Restore exactly hashed historical metric weights and strict-load on a CPU node."""
import concurrent.futures,hashlib,importlib.util,json,os,socket,sys,time,traceback,urllib.request
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower() and not os.environ.get('CUDA_VISIBLE_DEVICES')
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'assets/evaluation_v272';out.mkdir(exist_ok=True);start=time.monotonic()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,r):
    b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)

try:
    spec=json.loads((R/'code/metric_restoration_v272.json').read_text());sources=R/'code/metric_sources_v272'
    for name,digest in spec['sources_sha256'].items():assert sha(sources/name)==digest
    def download(row):
        p=out/row['filename'];tmp=p.with_name(p.name+'.partial');h=hashlib.sha256();offset=0
        if p.exists():assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];return dict(row,path=str(p),verified_sha256=sha(p),reused=True)
        if tmp.exists():
            with tmp.open('rb') as f:
                for b in iter(lambda:f.read(4<<20),b''):offset+=len(b);h.update(b)
        assert offset<=row['bytes']
        if offset<row['bytes']:
            headers={'Accept-Encoding':'identity','User-Agent':'EEG-metric-restoration'}
            if offset:headers['Range']='bytes='+str(offset)+'-'
            request=urllib.request.Request(row['url'],headers=headers)
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request,timeout=45) as response,tmp.open('ab') as f:
                assert response.status==(206 if offset else 200)
                if offset:assert response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-')
                while True:
                    assert time.monotonic()-start<1500
                    b=response.read(4<<20)
                    if not b:break
                    offset+=len(b);assert offset<=row['bytes'];f.write(b);h.update(b)
                f.flush();os.fsync(f.fileno())
        assert offset==row['bytes'] and h.hexdigest()==row['sha256'],(row['name'],offset,h.hexdigest())
        tmp.replace(p);r=dict(row,path=str(p),verified_sha256=sha(p),reused=False);write(out/(row['name']+'_asset.json'),r);return r
    results=[];failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(download,row):row['name'] for row in spec['assets']}
        for future in concurrent.futures.as_completed(jobs):
            try:results.append(future.result())
            except Exception as e:failures.append(dict(name=jobs[future],error=repr(e)))
    write(out/'download_status.json',dict(status='ALL_BYTES_RESTORED' if not failures else 'PARTIAL_PRESERVED',assets=results,failures=failures,seconds=time.monotonic()-start))
    assert not failures,failures
    import torch
    import torchvision
    torch.set_num_threads(4);torch.manual_seed(1729)
    from torchvision.models import alexnet,inception_v3,efficientnet_b1,resnet50
    lookup={v['name']:v for v in results};checks={}
    for name,model in [('AlexNet',alexnet(weights=None)),('Inception',inception_v3(weights=None,init_weights=False)),('EffNet',efficientnet_b1(weights=None))]:
        state=torch.load(lookup[name]['path'],map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
        assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());checks[name]=dict(strict_load=True,parameters=sum(p.numel() for p in model.parameters()))
        del model,state
    model=resnet50(weights=None);state=torch.load(lookup['SwAV']['path'],map_location='cpu',weights_only=True);state={k.replace('module.',''):v for k,v in state.items()};diff=model.load_state_dict(state,strict=False)
    assert set(diff.missing_keys)=={'fc.weight','fc.bias'} and all(k.startswith(('projection_head.','prototypes.')) for k in diff.unexpected_keys)
    checks['SwAV']=dict(strict_backbone=True,unused_head_missing=diff.missing_keys,unused_pretrain_heads=diff.unexpected_keys);del model,state
    ms=importlib.util.spec_from_file_location('pinned_clip_model_v272',sources/'openai_clip_model.py');module=importlib.util.module_from_spec(ms);ms.loader.exec_module(module)
    scripted=torch.jit.load(lookup['CLIP_L14']['path'],map_location='cpu').eval();model=module.build_model(scripted.state_dict()).eval();del scripted
    assert model.visual.input_resolution==224 and model.visual.conv1.kernel_size==(14,14) and len(model.visual.transformer.resblocks)==24
    checks['CLIP_L14']=dict(strict_build=True,parameters=sum(p.numel() for p in model.parameters()))
    del model
    sys.path.insert(0,str(sources));from dinov3.hub.backbones import dinov3_vits16
    model=dinov3_vits16(pretrained=False).eval().requires_grad_(False);model.load_state_dict(torch.load(lookup['DINOv3']['path'],map_location='cpu',weights_only=True),strict=True)
    with torch.inference_mode():patch,cls=model.get_intermediate_layers(torch.zeros(1,3,256,256),n=[11],return_class_token=True,norm=True)[0]
    assert patch.shape==(1,256,384) and cls.shape==(1,384) and torch.isfinite(patch).all() and torch.isfinite(cls).all()
    checks['DINOv3']=dict(strict_load=True,layer11_patch_shape=list(patch.shape),layer11_cls_shape=list(cls.shape),CPU_forward_finite=True,official_source_commit=spec['DINO_official_source_commit'],historical_source_bitexact_claim=False)
    assert not torch.cuda.is_initialized()
    write(out/'completion.json',dict(status='PASS_ALL6_EXACT_HISTORICAL_WEIGHT_SHA_AND_CPU_STRICT_MODELS',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),seconds=time.monotonic()-start,assets=results,checks=checks,
        GPU_initialized=False,images_scored=0,SEED_caption_and_detector_assets_still_missing=True,new_scoring_queue_ready=False,source_sha256=sha(__file__)))
    print('PASS_ALL6_METRIC_WEIGHT_BYTES_AND_CPU_MODELS',flush=True)
except BaseException as e:
    write(out/('failure_'+os.environ['SLURM_JOB_ID']+'.json'),dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
