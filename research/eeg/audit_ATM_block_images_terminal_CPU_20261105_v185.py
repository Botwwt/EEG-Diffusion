"""Independent complete-image audit; OLD CPU, only after the sealed lifecycle."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
os.environ['MPLBACKEND']='Agg'
import argparse,datetime,hashlib,io,json,resource,sys,time,traceback,zipfile
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--plan',required=True);a=ap.parse_args()
root=Path(a.policy).parent.parent;out=root/'image_output';dst=root/'terminal_review'
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def process(pid,ticks=None):
 p=Path('/proc',str(pid),'stat')
 if not p.exists():return dict(PID=pid,live=False)
 s=p.read_text().rsplit(')',1)[1].split()
 return dict(PID=pid,start_ticks=int(s[19]),live=s[0] not in ('Z','X') and (ticks is None or int(s[19])==ticks))
def guard():
 assert time.monotonic()-started<850
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<3<<30
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
try:
 launch=read(Path(a.policy).parent/'actual_launch.json')
 while process(launch['PID'],launch['start_ticks'])['live']:
  assert time.monotonic()-started<30
  time.sleep(.25)
 done=read(out/'completion.json');assert done['status']=='COMPLETE_BLOCK_FACTORIAL_IMAGE_EVALUATION'
 assert not (out/'failure.json').exists()
 assert sha(a.policy)==done['policy']['sha256'] and len(done['events'])==16
 for e in done['events']:
  if e['GPU']:
   p=e['receipt'];assert sha(p['path'])==p['sha256'];j=read(p['path'])
   assert j['status']=='COMPLETED' and j['exit_code']==0
   assert not process(j['pid'])['live'] and not process(j['wrapper_pid'])['live']
 import numpy as np
 from PIL import Image
 sys.path.insert(0,str(Path(a.policy).parent))
 import atm_block_image_contract_20261105_v185 as C
 c=C.setup(a.policy);K=c['K'];q=c['q'];before=C.resources(c)
 import score_low_level as low
 assert K.ref(low.__file__)==q['runtime_sources']['low_level']
 assert K.ref(__file__)==q['sources']['review'] and K.ref(a.plan)==q['review_plan']
 plan=read(a.plan);indices=plan['local_visual_query_indices']
 assert indices=={'R4':[0,100,200,300],'R80':[0,50,100,150]}
 assert plan['chosen_before_new_effect_values_read'] and plan['chosen_before_generation_started']
 assert not (dst/'completion.json').exists()
 assert read(out/'image_audit.json')['status']=='PASS_ALL2400_IMAGE_PROVENANCE'
 result=read(out/'readout.json');assert result['status']=='COMPLETE_BLOCK_FACTORIAL_R4_R80_ALL_METRICS'
 for p in [done['readout'],done['image_audit'],result['arrays'],result['matrices'],result['paired_differences']]:K.checked(p)
 values=np.load(out/'conditions.npy');tokens=np.load(out/'tokens.npy')
 assert values.shape==(600,4,1024) and tokens.shape==(600,4,4,2048)
 assert np.array_equal(values,c['values'])
 training=np.load(K.checked(q['diagnostic_conditions']))
 assert np.array_equal(values[:,2:],training[:,[1,2]])
 records=read(out/'ordered_images.json');assert len(records)==2400
 index={(r['domain'],r['arm'],r['query_index']):r for r in records};assert len(index)==2400
 arrays=np.load(out/'metrics.npz');matrices=np.load(out/'cosine_matrices.npz');diffs=np.load(out/'paired_differences.npz')
 identities=('query_id','subject_id','stimulus_id','concept_id','array_row','image_seed')
 metadata=K.read(K.checked(c['prepared']['target_metadata_ref']))['Dcheck']
 paths=set();inodes=set();zips={};sample=[];pixel_count=0;low_count=0;maxlow=0.;semantic_count=0;paircount=0
 feature_checks=[];rank_decomposition={}
 for domain in ['R4','R80']:
  n=400 if domain=='R4' else 200;offset=0 if domain=='R4' else 400
  queries=c['queries'][domain];assert len(queries)==n
  groups={}
  for i,row in enumerate(queries):groups.setdefault(row['concept_id'],[]).append(i)
  assert len(groups)==(40 if domain=='R4' else 200)
  features=np.load(out/f'features_{domain}.npy');target=np.load(out/f'targets_{domain}.npy')
  assert features.shape==(n,4,768) and target.shape==(n,768)
  assert np.isfinite(features).all() and np.isfinite(target).all()
  for j,arm in enumerate(C.ARMS):
   guard();m=matrices[domain+'__'+arm];assert m.shape==(n,n) and np.isfinite(m).all()
   diag=np.diag(m);other=~np.eye(n,dtype=bool)
   win=(m<diag[:,None]).astype(float)+.5*(m==diag[:,None])
   afc=np.sum(np.where(other,win,0),axis=1)/(n-1)
   checks=dict(target_cosine=diag,DINO2AFC=afc,gallery_top1=(m.argmax(1)==np.arange(n)).astype(float))
   if domain=='R4':
    top=np.zeros(n);within=np.zeros(n)
    for ids in groups.values():
     assert len(ids)==10
     for i in ids:
      neg=[k for k in ids if k!=i];top[i]=ids[int(np.argmax(m[i,ids]))]==i
      within[i]=float(win[i,neg].sum()/9)
    checks.update(within_concept10_top1=top,within_concept10_two_afc=within)
   else:checks['gallery_top5']=np.array([i in sorted(range(n),key=lambda k:(-float(m[i,k]),k))[:5] for i in range(n)],float)
   for key,v in checks.items():
    np.testing.assert_allclose(v,arrays[domain+'__'+arm+'__'+key],atol=1e-12,rtol=0);semantic_count+=n
   fn=features[:,j]/np.linalg.norm(features[:,j],axis=1,keepdims=True);tn=target/np.linalg.norm(target,axis=1,keepdims=True)
   recalc=fn@tn.T;err=float(abs(recalc-m).max());assert err<3e-6
   feature_checks.append(dict(domain=domain,arm=arm,max_cosine_error=err,retained_matrix_is_primary=True))
   metrics=C.R4_METRICS if domain=='R4' else C.R80_METRICS
   for key in metrics:
    v=arrays[domain+'__'+arm+'__'+key];assert v.shape==(n,)
    assert result['domains'][domain]['defined_counts'][arm][key]==int(np.isfinite(v).sum())
    mean=result['domains'][domain]['means'][arm][key]
    assert (mean is None) if not np.isfinite(v).all() else abs(float(v.mean())-mean)<1e-12
   if domain=='R80':
    paper=read(out/('paper_'+arm+'_completion.json'));per=np.load(K.checked(paper['values']))
    for k,key in enumerate(C.PAPER):np.testing.assert_equal(per[:,k],arrays[domain+'__'+arm+'__'+key])
  groupids=[groups[k] for k in sorted(groups)]
  rng=np.random.default_rng(1729);draw=rng.integers(0,len(groupids),(1000,len(groupids)))
  rank_decomposition[domain]={}
  for aa,bb in C.CONTRASTS:
   contrast=aa+'-'+bb
   for key in metrics:
    d=arrays[domain+'__'+aa+'__'+key]-arrays[domain+'__'+bb+'__'+key]
    np.testing.assert_equal(d,diffs[domain+'__'+contrast+'__'+key])
    saved=result['domains'][domain]['fixed_comparisons'][contrast][key]
    assert saved['total_queries']==n and saved['defined_queries']==int(np.isfinite(d).sum())
    if np.isfinite(d).all():
     gm=np.array([sum(float(d[i]) for i in g)/len(g) for g in groupids])
     boot=gm[draw].mean(1);ci=np.quantile(boot,[.025,.975])
     assert abs(float(d.mean())-saved['difference'])<1e-12
     np.testing.assert_allclose(ci,saved['descriptive_concept_bootstrap95'],atol=1e-12,rtol=0)
    else:assert saved['difference'] is None and saved['descriptive_concept_bootstrap95'] is None
    paircount+=1
   da=arrays[domain+'__'+aa+'__DINO2AFC']-arrays[domain+'__'+bb+'__DINO2AFC']
   mass=da*(n-1);assert abs(mass.sum()/(n*(n-1))-da.mean())<1e-12
   decomp=dict(positive_queries=int((da>0).sum()),negative_queries=int((da<0).sum()),unchanged_queries=int((da==0).sum()),
     rank_mass_gained=float(mass[mass>0].sum()),rank_mass_lost=float(-mass[mass<0].sum()),mean_difference=float(da.mean()),
     posthoc_descriptive_not_unique_cause=True)
   if domain=='R4':
    dw=arrays[domain+'__'+aa+'__within_concept10_two_afc']-arrays[domain+'__'+bb+'__within_concept10_two_afc']
    outside=(da*399-dw*9)/390
    np.testing.assert_allclose(da,9/399*dw+390/399*outside,atol=1e-12,rtol=0)
    decomp.update(within_mean=float(dw.mean()),other_concept_mean=float(outside.mean()),within_weight=9/399,other_concept_weight=390/399)
   rank_decomposition[domain][contrast]=decomp
  for i in range(n):
   guard();expected=queries[i]
   if domain=='R4':
    m=metadata[i];assert all(m[k]==expected[k] for k in identities)
    if m['image_zip'] not in zips:zips[m['image_zip']]=zipfile.ZipFile(m['image_zip'])
    raw=zips[m['image_zip']].read(m['image_member']);assert hashlib.sha256(raw).hexdigest()==m['image_sha256']
    target_source=dict(image_zip=m['image_zip'],image_member=m['image_member'],sha256=m['image_sha256'])
   else:target_source=expected['target'];raw=K.checked(target_source).read_bytes()
   trgb,ty=low.pixels(io.BytesIO(raw));tm=low.moments(ty)
   if i in indices[domain]:
    targetpath=dst/'samples'/domain/f'q{i:03d}_target.image';targetpath.parent.mkdir(parents=True,exist_ok=True);targetpath.write_bytes(raw)
   noise=c['positive'][i]['noise'] if domain=='R4' else c['r80noise'][i]['noise']
   for j,arm in enumerate(C.ARMS):
    row=index[domain,arm,i];assert all(row[k]==expected[k] for k in identities)
    p=K.checked(row['image']);st=p.stat();inode=(st.st_dev,st.st_ino)
    assert str(p) not in paths and inode not in inodes;paths.add(str(p));inodes.add(inode)
    with Image.open(p) as im:rgb=np.asarray(im.convert('RGB'))
    assert rgb.shape==(512,512,3) and hashlib.sha256(rgb.tobytes()).hexdigest()==row['pixels_sha256']
    v=values[offset+i,j][None];t=tokens[offset+i,j][None]
    assert hashlib.sha256(v.tobytes()).hexdigest()==row['condition_FP32_sha256']
    assert hashlib.sha256(v.astype(np.float16).tobytes()).hexdigest()==row['condition_FP16_sha256']
    assert hashlib.sha256(t.tobytes()).hexdigest()==row['tokens_FP16_sha256'] and row['noise']==noise
    if j<2:assert row['image']==c['reference_records'][domain][arm][i]['image']
    else:assert row['UNet_NFE']==4 and row['EEG_deployment_forward_calls']==1 and row['normal_inference_only_EEG'] and not row['query_target_or_concept_network_input']
    if domain=='R80':assert row['target']==target_source
    rgb,y=low.pixels(p)
    actual=[low.pixcorr(rgb,trgb),low.ssim_y(y,ty,tm),float(abs(rgb.mean((0,1))-trgb.mean((0,1))).mean())]
    for key,x in zip(['PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error'],actual):
     saved=arrays[domain+'__'+arm+'__'+key][i]
     if np.isfinite(x) and np.isfinite(saved):err=abs(x-saved);assert err<1e-10;maxlow=max(maxlow,err)
     else:assert np.isnan(x) and np.isnan(saved)
     low_count+=1
    pixel_count+=1
    if i in indices[domain]:
     samplepath=dst/'samples'/domain/f'q{i:03d}_{arm}.webp';samplepath.write_bytes(p.read_bytes())
     sample.append(dict(domain=domain,query_index=i,arm=arm,image=ref(samplepath),target=ref(targetpath),target_source=target_source,record=row))
 for z in zips.values():z.close()
 assert pixel_count==2400 and low_count==7200 and paircount==138 and semantic_count==11200 and len(sample)==32
 write(dst/'rank_accounting.json',dict(status='EXACT_DESCRIPTIVE_RANK_ACCOUNTING',domains=rank_decomposition,not_a_new_optimized_selection_rule=True))
 write(dst/'fixed_samples.json',sample)
 write(dst/'identity_queries.json',c['queries'])
 import matplotlib.pyplot as plt
 names=['NN: frozen NCE','PP: stochastic P','NP: new head','PN: new trunk'];colors=['#555555','#36759f','#bc8c34','#b04468']
 figures=[]
 for domain in ['R4','R80']:
  keys=list(C.R4_METRICS if domain=='R4' else C.R80_METRICS);cols=3;rows=(len(keys)+2)//3
  fig,axes=plt.subplots(rows,cols,figsize=(15,4*rows),squeeze=False)
  for ax,key in zip(axes.flat,keys):
   means=[result['domains'][domain]['means'][arm][key] for arm in C.ARMS]
   ax.bar(range(4),[np.nan if x is None else x for x in means],color=colors)
   ax.set_xticks(range(4),names,rotation=16,ha='right',fontsize=8);ax.set_title(key+(' (lower better)' if key in result['distance_metrics_lower_better'] else ' (higher better)'),fontsize=10)
   for j,x in enumerate(means):
    if x is not None:ax.text(j,x,f'{x:.4f}',ha='center',va='bottom',fontsize=8)
   ax.margins(y=.18);ax.grid(axis='y',alpha=.2)
  for ax in axes.flat[len(keys):]:ax.axis('off')
  fig.suptitle(f'{domain}: all registered measures | sub-01 | frozen block factorial | exposed development',fontsize=14)
  fig.tight_layout(rect=[0,0,1,.97]);p=dst/f'means_{domain}.png';fig.savefig(p,dpi=130);plt.close(fig);figures.append(ref(p))
  fig,axes=plt.subplots(rows,cols,figsize=(15,4*rows),squeeze=False)
  short=dict(zip(C.ARMS,['NN','PP','NP','PN']));labels=[short[aa]+' - '+short[bb] for aa,bb in C.CONTRASTS]
  for ax,key in zip(axes.flat,keys):
   for j,(aa,bb) in enumerate(C.CONTRASTS):
    v=result['domains'][domain]['fixed_comparisons'][aa+'-'+bb][key]
    if v['difference'] is not None:
     lo,hi=v['descriptive_concept_bootstrap95'];ax.plot([lo,hi],[j,j],color=colors[3] if j<3 else colors[2]);ax.scatter([v['difference']],[j],s=20,color=colors[3] if j<3 else colors[2])
   ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(len(C.CONTRASTS)),labels,fontsize=8);ax.invert_yaxis();ax.set_title(key,fontsize=10);ax.grid(axis='x',alpha=.2)
  for ax in axes.flat[len(keys):]:ax.axis('off')
  fig.suptitle(f'{domain}: every fixed contrast | descriptive concept bootstrap 95% | no training-seed uncertainty',fontsize=12)
  fig.tight_layout(rect=[0,0,1,.97]);p=dst/f'contrasts_{domain}.png';fig.savefig(p,dpi=130);plt.close(fig);figures.append(ref(p))
  fig,axes=plt.subplots(4,5,figsize=(15,12))
  for ri,i in enumerate(indices[domain]):
   selected=[s for s in sample if s['domain']==domain and s['query_index']==i]
   paths1=[Path(selected[0]['target']['path'])]+[Path(next(s for s in selected if s['arm']==arm)['image']['path']) for arm in C.ARMS]
   for col,p in enumerate(paths1):
    with Image.open(p) as im:axes[ri,col].imshow(im.convert('RGB'))
    axes[ri,col].axis('off')
    if ri==0:axes[ri,col].set_title((['Target']+names)[col],fontsize=11)
    if col==0:axes[ri,col].text(0,.98,f'q{i}',transform=axes[ri,col].transAxes,color='white',backgroundcolor='black')
  fig.suptitle(f'{domain}: fixed indices, all arms | chosen before image generation and effect read',fontsize=12)
  fig.tight_layout(rect=[0,0,1,.97]);p=dst/f'samples_{domain}.png';fig.savefig(p,dpi=130);plt.close(fig);figures.append(ref(p))
 guard();after=C.resources(c)
 proof=dict(status='PASS_ALL2400_IDENTITIES_PIXELS_NOISES_11200_RANK_VALUES_7200_LOWLEVEL_138_CIS',
 UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),image_completion=ref(out/'completion.json'),policy=ref(a.policy),review_plan=ref(a.plan),source=ref(__file__),
 pixel_images=2400,semantic_values=semantic_count,lowlevel_values=low_count,maximum_lowlevel_error=maxlow,paired_intervals=paircount,features=feature_checks,
 fixed_sample_images=32,figures=figures,neural_evaluation_and_training_NOT_repeated=True,raw_event_pairing_NOT_replayed=True,
 actual_plot_view_pending=True,NEW_contacted=False,no_SOTA_or_second_originality_claim=True,resources_before=before,resources_after=after,
 seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)
 write(dst/'completion.json',proof);print(json.dumps(dict(status=proof['status'],seconds=proof['seconds'],completion=ref(dst/'completion.json'))),flush=True)
except BaseException as exc:
 dst.mkdir(exist_ok=True);write(dst/('failure_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json'),dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
