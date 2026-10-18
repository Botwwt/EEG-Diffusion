"""SERVER ONLY: independent image metrics for source-heldout conditioning arms."""
import datetime
import fcntl
import os
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile

assert sys.platform=='linux'
R=Path('/root/autodl-tmp/mced_research_bundle');B=Path('/root/autodl-tmp/mced_bootstrap_20261011')
OUT=Path('/root/mced_eeg_norm_stats_content_metrics_v2');SAMPLES=Path('/root/mced_eeg_norm_stats_content_samples_v2')
FEAT=R/'studies/eeg_clip_visual_feature_cache_v2';DATA=R/'external/things_eeg2_direct_v1'
PROOF=R/'studies/raev2_reference_source_v2';SOURCE=Path('/root/mced_representation_sources_v2')
ASSET=Path('/root/mced_brain_dinov3_s_v1')
assert not OUT.exists();OUT.mkdir();START=time.monotonic();SAVED=None;FDS=[]
COUNTS=dict(encoder_calls=0,encoder_image_states=0,optimizer_updates=0,UNet_calls=0,VAE_calls=0)

def identity(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())

def write(n,v):
    with (OUT/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def validate(d):
    c=json.loads((d/'final_commit.json').read_text());assert c['complete']
    for n,v in c['files'].items():assert identity(d/n)==v

try:
    write('plan.json',dict(script=identity(Path(__file__)),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),
        sample_study=str(SAMPLES),selection='Exactly96distinct source heldout concepts outside current32 statistics trial; selected deterministically before new inference, reused in earlier research; no independent confirmation',
        encoder='Strict official frozen DINOv3-S/16; finalCLS+meanpatch768D,l2normalized; not the CLIP conditioning encoder',
        primary='Co-primary DINO2AFC comparisons in gallery96: trained real minus no_stats; SAME real weights with actual minus unrelated stats. Two fixed seeds averaged per image',
        secondary='PixCorr; same real prior zero-stat intervention; separately trained shuffled prior; fit mean; no checkpoint/best-of-N selection',
        oracle='True image original finalCLIP is unavailable at deployment, headroom only',
        preprocessing='Author square256 bilinear antialias,/255,ImageNetnorm; pixelcorrelation resizedRGB',
        bootstrap='20k paired sourceconcept-cluster draws, average2fixedEEGseeds per image first; source96only, no population subject or seed inference',
        expected_encoder_calls=79,expected_encoder_image_states=1264,
        official_test_read=False,exploratory=True,full_paper_protocol=False,SOTA=False))
    while not (SAMPLES/'final_commit.json').exists():
        if (SAMPLES/'failure.json').exists():raise RuntimeError('Matched source sampler failed')
        if (SAMPLES/'plan.json').exists():os.kill(json.loads((SAMPLES/'plan.json').read_text())['pid'],0)
        assert time.monotonic()-START<10800;time.sleep(10)
    validate(SAMPLES);validate(ASSET);validate(PROOF)
    sampler=json.loads((SAMPLES/'results.json').read_text());os.kill(sampler['pid'],0)
    assert str(B/'eeg_norm_stats_content_samples_v2.py').encode() in Path('/proc/'+str(sampler['pid'])+'/cmdline').read_bytes().split(b'\0')
    for row in json.loads((PROOF/'dinov3_source_manifest.json').read_text())['files']:
        assert identity(SOURCE/'dinov3'/row['path'])=={k:row[k] for k in ['bytes','sha256']}
    weight=ASSET/'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    assert identity(weight)==dict(bytes=86531063,sha256='08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d')
    fc=json.loads((FEAT/'final_commit.json').read_text());assert fc['complete']
    assert identity(FEAT/'image_receipts.json')==fc['files']['image_receipts.json']
    archive=DATA/'training_images.zip';receipt=json.loads(archive.with_name(archive.name+'.receipt.json').read_text())
    assert receipt['whole_file_verified'] and receipt['zip_crc_verified'] and identity(archive)=={k:receipt[k] for k in ['bytes','sha256']}
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import pil_to_tensor
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>2<<30
    SAVED=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),torch.get_num_threads(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

    def afc(scores):
        n=len(scores);diagonal=scores.diag()
        assert scores.shape==(n,n) and n>=2 and torch.isfinite(scores).all()
        return ((scores<diagonal[:,None]).sum(1)+.5*((scores==diagonal[:,None]).sum(1)-1))/(n-1)

    assert torch.equal(afc(torch.tensor([[.9,.95,.2],[.8,.5,.4],[.7,.6,.3]])),torch.tensor([.5,.5,0.]))
    assert torch.equal(afc(torch.ones(3,3)),torch.full((3,),.5))
    write('formula_preflight.json',dict(asymmetric_direction_passed=True,ties_half_credit_passed=True))
    sys.path.insert(0,str(SOURCE/'dinov3'))
    from dinov3.hub.backbones import dinov3_vits16
    model=dinov3_vits16(pretrained=False)
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    model=model.cuda().eval().requires_grad_(False)
    fixed=list(model.parameters())+list(model.buffers());versions=[p._version for p in fixed]
    transform=transforms.Compose([transforms.Lambda(lambda x:x/255.),transforms.Resize((256,256),antialias=True),
                                  transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    def encode(x):
        patch,cls=model.get_intermediate_layers(x,n=[11],return_class_token=True,norm=True)[0]
        y=F.normalize(torch.cat([cls,patch.mean(1)],dim=-1),dim=-1)
        assert y.shape==(len(x),768) and torch.isfinite(y).all()
        COUNTS['encoder_calls']+=1;COUNTS['encoder_image_states']+=len(x)
        return y

    def pixels(x):
        mean=torch.tensor([.485,.456,.406],device=x.device)[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device=x.device)[None,:,None,None]
        return (x*std+mean).clamp(0,1)

    def pearson(a,b):
        a=a.flatten(1);b=b.flatten(1);a=a-a.mean(1,keepdim=True);b=b-b.mean(1,keepdim=True)
        return (a*b).sum(1)/(a.square().sum(1)*b.square().sum(1)).sqrt().clamp_min(1e-12)

    result=json.loads((SAMPLES/'results.json').read_text());ids=result['image_indices'];assert len(ids)==len(set(ids))==96
    protocol=json.loads(Path('/root/mced_eeg_norm_stats_content_conditions_v1/image_plan.json').read_text())
    assert ids==protocol['source_image_indices'] and len({i//10 for i in ids})==96
    traces=json.loads((SAMPLES/'trace.json').read_text());assert len(traces)==4608
    byrun={}
    for row in traces:byrun.setdefault(row['run'],[]).append(row)
    assert len(byrun)==1152
    for cohort in result['cohorts']:
        for index in ids:
            events=byrun[cohort['arm']+'/'+str(index)];assert len(events)==4
            assert events[0]['latent_sha256']==result['paired_initial_latent_hashes'][str(index)]
            assert len({r['condition_sha256'] for r in events})==1
    write('independent_pairing_trace_audit.json',dict(complete=True,actual_UNet_trace_rows=4608,image_runs=1152,
        all_twelve_arms_actual_first_latents_matched_by_image=True,no_scientific_model_calls=True))
    receipts={x['index']:x for x in json.loads((FEAT/'image_receipts.json').read_text()) if x['split']=='train'}
    truths=[];selected=[];arrays={};evaluations=[]
    with torch.inference_mode(),zipfile.ZipFile(archive) as z:
        for i in ids:
            rec=receipts[i];raw=z.read(rec['member'])
            assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())=={k:rec[k] for k in ['bytes','sha256']}
            selected.append(rec)
            with Image.open(io.BytesIO(raw)) as im:truths.append(transform(pil_to_tensor(im.convert('RGB')).float()))
        truths=torch.stack(truths).cuda();tf=torch.cat([encode(x) for x in truths.split(16)])
        assert torch.equal(tf[:16],encode(truths[:16]))
        true_pixels=pixels(truths)
        for row in result['cohorts']:
            fd=os.open(row['memfd_path'],os.O_RDONLY);FDS.append(fd)
            p=Path('/proc/'+str(os.getpid())+'/fd/'+str(fd));assert fcntl.fcntl(fd,1034)==15 and identity(p)==row['identity']
            with np.load(p) as f:
                a=f['arr_0'];assert a.shape==(96,512,512,3) and a.dtype==np.uint8 and f['image_indices'].tolist()==ids
                generated=torch.stack([transform(torch.from_numpy(x.copy()).permute(2,0,1).float()) for x in a]).cuda()
            gf=torch.cat([encode(x) for x in generated.split(16)]);scores=gf@tf.T;diag=scores.diag()
            values=dict(DINO_2AFC=afc(scores),DINO_cosine=diag,DINO_Top1=((scores>=diag[:,None]).sum(1)==1).float(),
                        PixCorr=pearson(pixels(generated),true_pixels))
            arrays[row['arm']]={k:v.cpu().double().numpy() for k,v in values.items()}
            evaluated=dict(arm=row['arm'],oracle_reference=row['oracle_reference'],**{k:float(v.mean()) for k,v in arrays[row['arm']].items()},
                           per_image={k:v.tolist() for k,v in arrays[row['arm']].items()})
            evaluations.append(evaluated)
            print(json.dumps({k:v for k,v in evaluated.items() if k!='per_image'}),flush=True)
    assert COUNTS==dict(encoder_calls=79,encoder_image_states=1264,optimizer_updates=0,UNet_calls=0,VAE_calls=0),COUNTS
    assert versions==[p._version for p in fixed] and all(p.grad is None for p in fixed)
    for mode in ['point_no_stats','point_real_stats','point_shuffled_stats','point_real_zero_stats','point_real_unrelated_stats']:
        arrays[mode+'_seedmean']={k:np.mean([arrays[mode+'_'+str(seed)][k] for seed in [2026109501,2026109502]],axis=0) for k in arrays[mode+'_2026109501']}
    rng=np.random.default_rng(2026112001)
    bootstrap=rng.integers(0,96,(20000,96))
    concepts=np.asarray(ids,dtype=np.int64)//10;unique=np.unique(concepts)
    membership=(unique[:,None]==concepts[None,:]).astype(np.int32)
    draws=rng.integers(0,len(unique),(20000,len(unique)))
    cluster_weights=membership[draws].sum(1);denominator=cluster_weights.sum(1)
    assert np.all(denominator>0)
    comparisons=[]
    for arm,control in [('point_real_stats_seedmean','point_no_stats_seedmean'),
                        ('point_real_stats_seedmean','point_real_unrelated_stats_seedmean'),
                        ('point_real_stats_seedmean','point_real_zero_stats_seedmean'),
                        ('point_real_stats_seedmean','point_shuffled_stats_seedmean'),
                        ('point_no_stats_seedmean','point_shuffled_stats_seedmean'),
                        ('point_real_stats_seedmean','constant_fit_mean')]:
        metrics={}
        for k in arrays[control]:
            d=arrays[arm][k]-arrays[control][k]
            metrics[k]=dict(mean_difference=float(d.mean()),
                paired_concept_cluster_bootstrap95=np.quantile((cluster_weights@d)/denominator,[.025,.975]).tolist(),
                secondary_paired_image_bootstrap95=np.quantile(d[bootstrap].mean(1),[.025,.975]).tolist())
        comparisons.append(dict(arm=arm,control=control,metrics=metrics,source_images=96,source_concepts=len(unique)))
    # Two predeclared co-primary DINO comparisons. This sign-flip inference
    # assumes symmetric paired differences across independent concept clusters.
    sign_rng=np.random.default_rng(2026112002)
    signs=sign_rng.choice(np.asarray([-1.,1.]),size=(100000,96))
    primary_tests=[]
    for comp in comparisons[:2]:
        d=arrays[comp['arm']]['DINO_2AFC']-arrays[comp['control']]['DINO_2AFC']
        assert len(d)==96 and len(unique)==96
        simulated=(signs@d)/96
        p=float((1+np.count_nonzero(np.abs(simulated)>=abs(float(d.mean()))))/100001)
        primary_tests.append(dict(arm=comp['arm'],control=comp['control'],two_sided_paired_sign_flip_p=p))
    ordered=sorted(range(2),key=lambda i:primary_tests[i]['two_sided_paired_sign_flip_p'])
    previous_adjusted=0.
    for rank,index in enumerate(ordered):
        adjusted=min(1.,max(previous_adjusted,(2-rank)*primary_tests[index]['two_sided_paired_sign_flip_p']))
        primary_tests[index]['Holm2_adjusted_p']=adjusted;previous_adjusted=adjusted
    write('co_primary_sign_flip_tests.json',dict(tests=primary_tests,draws=100000,
        alternative='two-sided',multiplicity='Holm2',assumption='Symmetric paired differences across independent concept clusters',
        source_concepts_reused_in_earlier_research=True,independent_confirmatory_data=False))
    write('image_receipts.json',selected);write('evaluations.json',evaluations)
    write('bindings.json',dict(samples_commit=identity(SAMPLES/'final_commit.json'),DINO_weights=identity(weight),
                               source_manifest=identity(PROOF/'dinov3_source_manifest.json'),source_training_zip=identity(archive),
                               DINO_is_not_CLIP_conditioning_encoder=True,oracle_unavailable_at_inference=True))
    write('results.json',dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-START,counts=COUNTS,
        evaluations=[{k:v for k,v in row.items() if k!='per_image'} for row in evaluations],paired_comparisons=comparisons,co_primary_tests=primary_tests,
        primary_metric='DINO_2AFC',primary_direction='generated_query_true_gallery96',primary_uncertainty_unit='paired_source_concept_cluster',image_indices=ids,
        source_heldout_only=True,one_source=True,two_fixed_EEG_training_seeds=True,official_test_read=False,exploratory=True,
        full_paper_protocol=False,novelty_established=False,SOTA=False))
    write('final_commit.json',dict(complete=True,files={p.name:identity(p) for p in OUT.iterdir() if p.is_file()}))
except BaseException:
    write('failure.json',dict(traceback=traceback.format_exc(),counts=COUNTS,seconds=time.monotonic()-START));raise
finally:
    if SAVED is not None:
        cpu,cuda,threads,tf32,cudnn=SAVED;torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda);torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=cudnn

    for fd in FDS:os.close(fd)
