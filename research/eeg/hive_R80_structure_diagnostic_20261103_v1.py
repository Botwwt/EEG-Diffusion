"""Post-hoc CPU factor diagnosis on existing R80 pictures, with no method updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import hashlib,io,json,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROLES=('parent','ordinary_full1654','bounded_F','mean_ordinary_full1654')
CONTRASTS=(('ordinary_full1654','parent'),('ordinary_full1654','bounded_F'),('ordinary_full1654','mean_ordinary_full1654'))
METRICS=('SSIM256_Y','luminance_factor','contrast_factor','structure_factor','textured_structure','textured_SSIM',
         'PixCorr_RGB','PixCorr_RGB_channel_centered','PixCorr_Y_highpass_sigma2','mean_RGB_absolute_error')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def checked(pin):
    p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256'];return p
def ref(p):return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def write(p,data):
    p=Path(p);temp=p.with_name(p.name+'.pending');temp.write_bytes((json.dumps(data,indent=2,allow_nan=False)+'\n').encode());temp.replace(p)
def corr(a,b):
    a=a.ravel()-a.mean();b=b.ravel()-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>0 else 0.
def gaussian(a,sigma=1.5):return gaussian_filter(a,sigma=sigma,truncate=3.5,mode='reflect')
def decoded(path,expected,*,pixels_sha=None,generated=False):
    with Path(path).open('rb') as stream:
        raw=stream.read();assert hashlib.sha256(raw).hexdigest()==expected
        if hasattr(os,'posix_fadvise'):os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    image=Image.open(io.BytesIO(raw)).convert('RGB')
    if generated:assert image.size==(512,512)
    if pixels_sha is not None:assert hashlib.sha256(np.asarray(image).tobytes()).hexdigest()==pixels_sha
    image=image.resize((256,256),Image.Resampling.BICUBIC)
    return np.asarray(image).copy()
def target_data(pixels):
    rgb=pixels.astype(np.float64)/255.;y=np.asarray(Image.fromarray(pixels).convert('L'),dtype=np.float64)/255.
    mean=gaussian(y);var=np.maximum(gaussian(y*y)-mean*mean,0);std=np.sqrt(var)
    return rgb,y,mean,var,std,y-gaussian(y,2.)
def factors(y,target):
    _,ty,my,vy,sy,_=target
    mx=gaussian(y);vx=np.maximum(gaussian(y*y)-mx*mx,0);sx=np.sqrt(vx)
    cov=gaussian(y*ty)-mx*my
    light=(2*mx*my+.01**2)/(mx*mx+my*my+.01**2)
    contrast=(2*sx*sy+.03**2)/(vx+vy+.03**2)
    structure=(cov+.03**2/2)/(sx*sy+.03**2/2)
    joined=(2*mx*my+.01**2)*(2*cov+.03**2)/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    fac=np.stack((light[5:-5,5:-5],contrast[5:-5,5:-5],structure[5:-5,5:-5]))
    error=float(abs(fac.prod(0)-joined[5:-5,5:-5]).max());assert error<1e-10
    return fac,error
def allocation(a,b):
    result=[]
    for i in range(3):
        j,k=[v for v in range(3) if v!=i]
        weight=(b[j]*b[k]+a[j]*a[k])/3+(a[j]*b[k]+b[j]*a[k])/6
        result.append(float(((a[i]-b[i])*weight).mean()))
    assert abs(sum(result)-(a.prod(0).mean()-b.prod(0).mean()))<1e-12
    return np.asarray(result)
def checks():
    rng=np.random.default_rng(1729)
    for _ in range(10):
        a=rng.uniform(.1,1.,(3,8,8));b=rng.uniform(.1,1.,(3,8,8));allocation(a,b)
    a=np.stack([np.full((8,8),.7),np.ones((8,8)),np.ones((8,8))])
    b=np.stack([np.full((8,8),.3),np.ones((8,8)),np.ones((8,8))])
    np.testing.assert_allclose(allocation(a,b),[.4,0,0],rtol=0,atol=1e-14)
    pixels=np.full((256,256,3),127,np.uint8);target=target_data(pixels);fac,error=factors(target[1],target)
    np.testing.assert_allclose(fac,1,rtol=0,atol=1e-12)
    return {'product_allocation_identity':True,'brightness_only_allocation':True,'identity_factors':True,'factorization_error':error}

def run(policy_path):
    import resource,signal
    started=time.monotonic();signal.alarm(880);q=read(policy_path);out=Path(q['output_root']);assert sha(__file__)==q['worker']['sha256']
    assert not (out/'completion.json').exists();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    sys.path.insert(0,str(Path(q['native_policy']['path']).parent))
    import atm_native_postfit_contract_20261103_v1 as K
    native=read(checked(q['native_policy']));parent,train,e,C=K.setup(native)
    def guard(starting=False):
        for attempt in range(3):
            try:state=K.resources(native,parent,train,e,C,starting=starting);break
            except FileNotFoundError:
                if attempt==2:raise
        used=sum(p.stat().st_blocks*512 for p in out.rglob('*') if p.is_file())
        assert used<q['maximum_output_bytes']
        assert state['free_bytes']>=state['required_remaining_bytes']+parent['maximum_output_bytes']-state['native_project_used_bytes']+q['maximum_output_bytes']-used
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<q['maximum_extra_CPU_bytes']
        assert time.monotonic()-started<850
        return state
    before=guard(True);math_checks=checks();checked(q['previous_full_audit'])
    manifests={role:[json.loads(line) for line in checked(q['manifests'][role]).read_text().splitlines()] for role in ROLES}
    assert all(len(r)==2000 for r in manifests.values())
    gallery={r['stimulus_id']:r for r in map(json.loads,checked(q['gallery']).read_text().splitlines())};assert len(gallery)==200
    target_cache={};target_pixels={}
    for sid,row in gallery.items():
        p=decoded(row['image_path'],row['image_sha256']);target_pixels[sid]=p;target_cache[sid]=target_data(p)
    values=np.lib.format.open_memmap(out/'metrics.npy',mode='w+',dtype=np.float64,shape=(2000,4,len(METRICS)));values[:]=np.nan
    allocations=np.lib.format.open_memmap(out/'SSIM_factor_allocations.npy',mode='w+',dtype=np.float64,shape=(2000,3,3));allocations[:]=np.nan
    selected=np.empty((8,5,256,256,3),np.uint8);chosen={n:i for i,n in enumerate(q['fixed_pixel_indices'])}
    identities=[];max_error=0.;flat=0;inodes=set()
    identity=('query_id','subject','stimulus_id','concept_id','array_row','image_seed','target_image_sha256')
    try:
        for i,base in enumerate(manifests['parent']):
            target=target_cache[base['stimulus_id']];trgb,ty,_,_,ts,thp=target;mask=ts[5:-5,5:-5]>.02
            assert base['target_image_sha256']==gallery[base['stimulus_id']]['image_sha256']
            flat+=int(not mask.any());facts=[]
            if i in chosen:selected[chosen[i],0]=target_pixels[base['stimulus_id']]
            for j,role in enumerate(ROLES):
                row=manifests[role][i];assert all(row[k]==base[k] for k in identity)
                p=Path(row['generated_path']);st=p.stat();key=(st.st_dev,st.st_ino);assert key not in inodes;inodes.add(key)
                pixels=decoded(p,row['generated_sha256'],pixels_sha=row.get('generated_pixel_sha256'),generated=True)
                if i in chosen:selected[chosen[i],j+1]=pixels
                rgb=pixels.astype(np.float64)/255.;y=np.asarray(Image.fromarray(pixels).convert('L'),dtype=np.float64)/255.
                fac,error=factors(y,target);max_error=max(max_error,error);facts.append(fac)
                ssim=fac.prod(0)
                centered=rgb-rgb.mean((0,1),keepdims=True);tcenter=trgb-trgb.mean((0,1),keepdims=True)
                values[i,j]=(ssim.mean(),*(v.mean() for v in fac),fac[2][mask].mean() if mask.any() else 0.,
                    ssim[mask].mean() if mask.any() else 0.,corr(rgb,trgb),corr(centered,tcenter),corr(y-gaussian(y,2.),thp),
                    float(abs(rgb.mean((0,1))-trgb.mean((0,1))).mean()))
            for j,(_,other) in enumerate(CONTRASTS):allocations[i,j]=allocation(facts[1],facts[ROLES.index(other)])
            identities.append({k:base[k] for k in identity})
            if (i+1)%50==0:
                values.flush();allocations.flush();write(out/'progress.json',{'status':'RUNNING_CPU_ONLY_EXISTING_IMAGES','queries':i+1,'generated_files_checked':4*(i+1)})
                guard()
        assert np.isfinite(values).all() and np.isfinite(allocations).all() and len(inodes)==8000
        values.flush();allocations.flush();write(out/'identities.json',identities)
        np.savez_compressed(out/'fixed_metric_input_pixels.npz',indices=np.asarray(q['fixed_pixel_indices']),pixels=selected)
        subjects=sorted({r['subject'] for r in identities});concepts=sorted({r['concept_id'] for r in identities})
        assert len(subjects)==10 and len(concepts)==200
        lookup={(r['subject'],r['concept_id']):i for i,r in enumerate(identities)};assert len(lookup)==2000
        grid=np.array([[lookup[(s,c)] for c in concepts] for s in subjects])
        rng=np.random.default_rng(1729);draws=[]
        for _ in range(1000):draws.append((rng.integers(0,10,10),rng.integers(0,200,200)))
        def summary(v):
            matrix=v[grid];samples=np.asarray([matrix[np.ix_(s,c)].mean() for s,c in draws])
            return {'difference':float(v.mean()),'descriptive_subject_concept_bootstrap95':np.quantile(samples,[.025,.975]).tolist(),'per_subject':matrix.mean(1).tolist()}
        comparisons={}
        for j,(a,b) in enumerate(CONTRASTS):
            ia,ib=ROLES.index(a),ROLES.index(b);differences=values[:,ia]-values[:,ib]
            result={key:summary(differences[:,k]) for k,key in enumerate(METRICS)}
            for k,key in enumerate(('SSIM_luminance_allocation','SSIM_contrast_allocation','SSIM_structure_allocation')):result[key]=summary(allocations[:,j,k])
            np.testing.assert_allclose(allocations[:,j].sum(1),differences[:,0],rtol=0,atol=1e-12)
            comparisons[a+'-'+b]=result
        after=guard();import torch;assert not torch.cuda.is_initialized();signal.alarm(0)
        result={'status':'COMPLETE_FULL_R80_EXISTING_IMAGE_STRUCTURE_DIAGNOSIS','queries':2000,'generated_images_verified':8000,'target_images_verified':200,
            'roles':list(ROLES),'metrics':list(METRICS),'means':{r:{m:float(values[:,j,k].mean()) for k,m in enumerate(METRICS)} for j,r in enumerate(ROLES)},
            'comparisons':comparisons,'factorization_max_error':max_error,'factor_allocation_sum_exact':True,'target_flat_ROI_queries':flat,
            'protocol':ref(policy_path),'arrays':ref(out/'metrics.npy'),'allocations':ref(out/'SSIM_factor_allocations.npy'),'identities':ref(out/'identities.json'),
            'fixed_pixels':ref(out/'fixed_metric_input_pixels.npz'),'math_checks':math_checks,
            'scope':'Post-hoc descriptive factor allocation; arithmetic attribution is not a neural or causal mechanism. Supplementary256px metrics are separate from published evaluator metrics.',
            'new_GPU_jobs':0,'new_fits':0,'new_research_images':0,'method_or_weights_changed':False,'CUDA_initialized':False,
            'seconds':time.monotonic()-started,'extra_CPU_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,'resources_before':before,'resources_after':after}
        write(out/'completion.json',result);write(out/'progress.json',{'status':'COMPLETE_CPU_ONLY_STRUCTURE_DIAGNOSIS'})
    except BaseException as ex:
        write(out/'failure.json',{'status':'CPU_STRUCTURE_DIAGNOSTIC_FAILED_PREFIX_PRESERVED','error':repr(ex),'traceback':traceback.format_exc()});raise

if __name__=='__main__':run(sys.argv[1])
