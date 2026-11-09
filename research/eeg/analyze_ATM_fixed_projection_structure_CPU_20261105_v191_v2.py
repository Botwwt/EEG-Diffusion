"""Terminal-only SSIM factor accounting; no models, fits, or image changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,hashlib,importlib.util,io,itertools,json,resource,signal,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
from PIL import Image

ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args()
start=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
plan=json.loads(Path(args.plan).read_bytes());out=Path(plan['output_root'])
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def checked(r):
 p=Path(r['path']);assert sha(p)==r['sha256'];assert 'bytes' not in r or p.stat().st_size==r['bytes'];return p
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def write(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.pending');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def proc(pid,ticks=None):
 p=Path('/proc',str(pid),'stat')
 if not p.exists():return False
 s=p.read_text().rsplit(')',1)[1].split();return s[0] not in ['Z','X'] and (ticks is None or int(s[19])==ticks)
try:
 assert ref(__file__)==plan['worker'];assert not (out/'completion.json').exists();signal.alarm(870)
 parent=Path(plan['parent_root']);la=read(parent/'code/actual_launch.json');ra=read(parent/'terminal_review/actual_launch.json')
 assert not proc(la['PID'],la['start_ticks']) and not proc(ra['PID'],ra['start_ticks'])
 proof=read(parent/'terminal_review/completion.json')
 assert proof['status']=='PASS_ALL2400_IDENTITIES_PIXELS_NOISES_11200_RANK_VALUES_7200_LOWLEVEL_138_CIS'
 assert sha(parent/'image_output/completion.json')==proof['image_completion']['sha256']
 sys.path.insert(0,str(parent/'code'))
 import atm_fixed_projection_contract_20261105_v190 as C
 c=C.setup(checked(plan['parent_policy']));K=c['K'];before=C.resources(c,starting=True)
 spec=importlib.util.spec_from_file_location('prior_structure',checked(plan['factor_source']));F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
 import score_low_level as low
 assert ref(low.__file__)==c['q']['runtime_sources']['low_level']
 original=read(c['out']/'readout.json');stored=np.load(checked(original['arrays']))
 def guard():
  assert time.monotonic()-start<850
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<plan['maximum_extra_CPU_bytes']
  assert K.allocated(Path(plan.get('budget_root',str(out))))<plan['maximum_output_bytes'];return C.resources(c)
 fixtures=F.checks();allarrays={};summaries={};identities=[];max_ssim=max_product=max_alloc=max_enum=max_independent=0.;inodes=set();enum_cells=0
 # Independent explicit 11x11 convolution uses only valid pixels, rather than gaussian_filter.
 from scipy.signal import convolve2d
 kernel=np.exp(-np.arange(-5,6,dtype=np.float64)**2/(2*1.5**2));kernel/=kernel.sum();kernel=kernel[:,None]*kernel[None,:]
 def explicit(x,y):
  conv=lambda z:convolve2d(z,kernel,mode='valid')
  mx,my=conv(x),conv(y);vx=np.maximum(conv(x*x)-mx*mx,0);vy=np.maximum(conv(y*y)-my*my,0);xy=conv(x*y)-mx*my;sxsy=np.sqrt(vx*vy)
  return np.array([(2*mx*my+.01**2)/(mx*mx+my*my+.01**2),(2*sxsy+.03**2)/(vx+vy+.03**2),(xy+.03**2/2)/(sxsy+.03**2/2)])
 def enumerate_allocation(a,b):
  result=np.zeros(3)
  for order in itertools.permutations(range(3)):
   old=b.copy();v=old.prod(0)
   for k in order:
    old[k]=a[k];u=old.prod(0);result[k]+=(u-v).mean()/6;v=u
  return result
 for domain,n in [('R4',400),('R80',200)]:
  rows=c['queries'][domain];records=C.records(c,domain);values=np.full((n,4,len(F.METRICS)),np.nan);alloc=np.full((n,6,3),np.nan)
  metadata=read(checked(c['prepared']['target_metadata_ref']))['Dcheck'] if domain=='R4' else None;zips={};flat=[]
  try:
   for i,row in enumerate(rows):
    if domain=='R4':
     m=metadata[i]
     if m['image_zip'] not in zips:zips[m['image_zip']]=zipfile.ZipFile(m['image_zip'])
     raw=zips[m['image_zip']].read(m['image_member']);assert hashlib.sha256(raw).hexdigest()==m['image_sha256'];target_path=io.BytesIO(raw)
    else:target_path=checked(row['target'])
    with Image.open(target_path) as im:tp=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC)).copy()
    target=F.target_data(tp);trgb,ty,_,_,ts,thp=target;mask=ts[5:-5,5:-5]>.02;facts=[];flat.append(not bool(mask.any()))
    for j,arm in enumerate(C.ARMS):
     r=records[arm][i];assert all(r[k]==row[k] for k in ['query_id','subject_id','stimulus_id','concept_id','array_row','image_seed'])
     p=checked(r['image']);st=p.stat();inode=(st.st_dev,st.st_ino);assert inode not in inodes;inodes.add(inode)
     pixels=F.decoded(p,r['image']['sha256'],pixels_sha=r['pixels_sha256'],generated=True)
     rgb=pixels.astype(np.float64)/255.;y=np.asarray(Image.fromarray(pixels).convert('L'),dtype=np.float64)/255.
     fac,error=F.factors(y,target);max_product=max(max_product,error);facts.append(fac);sm=fac.prod(0)
     values[i,j]=(sm.mean(),*(a.mean() for a in fac),fac[2][mask].mean() if mask.any() else np.nan,sm[mask].mean() if mask.any() else np.nan,
      F.corr(rgb,trgb),F.corr(rgb-rgb.mean((0,1),keepdims=True),trgb-trgb.mean((0,1),keepdims=True)),F.corr(y-F.gaussian(y,2.),thp),float(abs(rgb.mean((0,1))-trgb.mean((0,1))).mean()))
     err=abs(values[i,j,0]-stored[domain+'__'+arm+'__SSIM_Y'][i]);max_ssim=max(max_ssim,float(err));assert err<1e-10
     if i==0:
      err=float(abs(explicit(y,ty)-fac).max());max_independent=max(max_independent,err);assert err<1e-10
    for k,(a,b) in enumerate(C.CONTRASTS):
     ia,ib=C.ARMS.index(a),C.ARMS.index(b);alloc[i,k]=F.allocation(facts[ia],facts[ib]);err=abs(alloc[i,k].sum()-(values[i,ia,0]-values[i,ib,0]));max_alloc=max(max_alloc,float(err))
     if i in [0,n//4,n//2,3*n//4]:
      err=float(abs(enumerate_allocation(facts[ia],facts[ib])-alloc[i,k]).max());max_enum=max(max_enum,err);assert err<1e-12;enum_cells+=1
    identities.append(dict(domain=domain,index=i,**{k:row[k] for k in ['query_id','subject_id','stimulus_id','concept_id','array_row','image_seed']}))
    if (i+1)%25==0:write(out/'progress.json',dict(domain=domain,queries=i+1,status='READ_ONLY_ALL_IMAGES_FACTOR_ACCOUNTING'));guard()
  finally:
   for z in zips.values():z.close()
  allarrays[domain+'__metrics']=values;allarrays[domain+'__allocations']=alloc;assert np.isfinite(alloc).all()
  groups=[[i for i,r in enumerate(rows) if r['concept_id']==g] for g in sorted({r['concept_id'] for r in rows})];assert len(groups)==(40 if domain=='R4' else 200)
  draws=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)))
  def summary(v):
   valid=np.isfinite(v);gm=np.array([v[g].mean() for g in groups]);complete=bool(valid.all())
   return dict(difference=float(v.mean()) if complete else None,descriptive_concept_bootstrap95=np.quantile(gm[draws].mean(1),[.025,.975]).tolist() if complete else None,defined_queries=int(valid.sum()),total_queries=n)
  pairs={}
  for k,(a,b) in enumerate(C.CONTRASTS):
   ia,ib=C.ARMS.index(a),C.ARMS.index(b);pairs[a+'-'+b]={key:summary(values[:,ia,j]-values[:,ib,j]) for j,key in enumerate(F.METRICS)}
   pairs[a+'-'+b].update({key:summary(alloc[:,k,j]) for j,key in enumerate(['SSIM_luminance_allocation','SSIM_contrast_allocation','SSIM_structure_allocation'])})
  summaries[domain]=dict(queries=n,concepts=len(groups),target_flat_queries=int(sum(flat)),means={a:{m:float(values[:,j,k].mean()) if np.isfinite(values[:,j,k]).all() else None for k,m in enumerate(F.METRICS)} for j,a in enumerate(C.ARMS)},comparisons=pairs)
 assert len(inodes)==2400 and len(identities)==600 and enum_cells==48
 with (out/'per_query.npz').open('xb') as f:np.savez_compressed(f,**allarrays)
 write(out/'identities.json',identities)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plots=[]
 for domain in ['R4','R80']:
  fig,ax=plt.subplots(figsize=(12,5));names=[];x=np.arange(6);colors=['#3568a9','#d7892d','#3c9466']
  for j,key in enumerate(['SSIM_luminance_allocation','SSIM_contrast_allocation','SSIM_structure_allocation']):
   v=[summaries[domain]['comparisons'][a+'-'+b][key] for a,b in C.CONTRASTS];y=np.array([r['difference'] for r in v]);ci=np.array([r['descriptive_concept_bootstrap95'] for r in v]).T
   ax.errorbar(x+(j-1)*.22,y,yerr=np.maximum(0,np.stack([y-ci[0],ci[1]-y])),fmt='o',capsize=3,color=colors[j],label=['Luminance','Contrast','Structure'][j])
  short=dict(zip(C.ARMS,['NN','CC','CN','FH']));ax.axhline(0,c='grey',lw=.8);ax.set_xticks(x,[short[a]+' - '+short[b] for a,b in C.CONTRASTS]);ax.set_ylabel('Allocation to paired SSIM difference');ax.set_title(domain+' full panel: arithmetic factor allocation, descriptive intervals');ax.legend(ncol=3);fig.tight_layout()
  p=out/(domain+'_SSIM_factor_allocations.png');fig.savefig(p,dpi=150);plt.close(fig);plots.append(ref(p))
 after=guard();assert 'torch' not in sys.modules or not sys.modules['torch'].cuda.is_initialized();signal.alarm(0)
 write(out/'completion.json',dict(status='COMPLETE_ALL2400_SSIM_FACTORS_600_QUERY_156_INTERVALS',plan=ref(args.plan),parent_review=ref(parent/'terminal_review/completion.json'),parent_readout=ref(c['out']/'readout.json'),domains=summaries,arms=list(C.ARMS),metrics=list(F.METRICS),arrays=ref(out/'per_query.npz'),identities=ref(out/'identities.json'),plots=plots,
  checks=dict(fixtures=fixtures,all2400_ssim_vs_locked_max_error=max_ssim,all2400_product_max_error=max_product,all3600_allocation_sum_max_error=max_alloc,independent_explicit_convolution_images=8,independent_convolution_max_error=max_independent,independent_six_order_enumerations=48,enumeration_max_error=max_enum),
  seconds=time.monotonic()-start,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources_before=before,resources_after=after,CUDA_initialized=False,new_images=0,new_fits=0,
  interpretation='Secondary arithmetic attribution, not causal mechanism or neural information. Existing SSIM_Y retained; published eval_ssim separate. Flat target masks retained as undefined, no silent exclusion. Existing exposed controls known before registration; FH effects unread then.'))
except BaseException as exc:
 write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
