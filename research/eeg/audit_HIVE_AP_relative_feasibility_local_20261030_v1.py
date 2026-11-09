"""Independent gradient geometry, packed update and Gram spectrum checks."""
import argparse,datetime,hashlib,json,os
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();E=a.directory;read=lambda p:json.loads(Path(p).read_bytes())
manifest=read(E/'received_manifest.json')['manifest']
for name,item in manifest.items():
    raw=(E/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
s=read(E/'completion.json');v=np.load(E/'actual_T80_readouts.npz',allow_pickle=False);packed=np.load(E/'actual_parameter_gradients.npz',allow_pickle=False);grams=np.load(E/'actual_four_tangent_grams.npz',allow_pickle=False);es=np.load(E/'actual_four_eigenvalues.npz',allow_pickle=False)
u=(v['ordinary']-v['beta'])/v['gamma'];one=np.ones(1024)/32;tangent_errors=[];maximum=0.
def check(x,y,tol=1e-12):
    global maximum
    err=float(np.max(abs(np.asarray(x)-np.asarray(y))));assert err<tol;maximum=max(maximum,err);return err
for field,grad in [('relative_gradient_tangent','relative_gradient_condition'),('AP_gradient_tangent','AP_gradient_condition')]:
    z=v['gamma']*v[grad];z-=np.outer(z@one,one);uc=u-u.mean(1,keepdims=True);z-=uc*np.sum(z*uc,1,keepdims=True)/np.sum(uc*uc,1,keepdims=True);tangent_errors.append(check(z,v[field]))
gr,ga=v['relative_gradient_tangent'],v['AP_gradient_tangent'];nr=np.linalg.norm(gr,axis=1);na=np.linalg.norm(ga,axis=1);cos=np.sum(gr*ga,1)/(nr*na);check(cos,v['gradient_cosine'])
d=-(gr/nr[:,None]+ga/na[:,None]);d/=np.linalg.norm(d,axis=1,keepdims=True);check(d,v['common_direction_tangent']);margin=np.minimum(-np.sum(d*gr,1)/nr,-np.sum(d*ga,1)/na)
check(margin,np.sqrt((1+cos)/2));check(cos.mean(),s['condition']['gradient_cosine_mean']);check(margin.min(),s['condition']['normalized_common_descent_margin_min']);assert np.count_nonzero(cos<0)==s['condition']['negative_cosines']
gr,ga=packed['relative'],packed['AP'];assert len(gr)==len(ga)==525568
dc=-(gr/np.linalg.norm(gr)+ga/np.linalg.norm(ga));dc/=np.linalg.norm(dc);check(dc,packed['direction']);derivatives=np.asarray([gr@dc,ga@dc]);check(derivatives,s['parameter']['analytic_directional_derivatives']);check(gr@ga/(np.linalg.norm(gr)*np.linalg.norm(ga)),s['parameter']['gradient_cosine'])
for r in s['parameter']['central_difference']:assert np.max(abs(np.asarray(r['central_difference'])-derivatives)/(1+abs(derivatives)))<1e-7
rank_reports={}
for key in grams.files:
    matrix=grams[key];assert matrix.shape==(1022,1022);eigen=np.linalg.eigvalsh(matrix);check(eigen,es[key],tol=1e-6);sv=np.sqrt(np.maximum(eigen,0));tol=8192*np.finfo(float).eps*sv[-1];rank=int(np.count_nonzero(sv>tol));r=s['rank_witnesses'][key[4:]];assert rank==r['numerical_rank'];check(sv[0],r['smallest_singular'],tol=1e-7);check(sv[-1],r['largest_singular'],tol=1e-7)
    rank_reports[key]=dict(numerical_rank=rank,tangent_dimension=1022,smallest_singular=float(sv[0]),largest_singular=float(sv[-1]))
assert s['new_GPU_calls']==0 and s['new_images']==0 and s['scientific_updates']==0 and s['parameter']['old_parameters_restored_exactly']
r=dict(status='PASS_INDEPENDENT_LOCAL_ACTUAL80_TANGENT_AND_525568_PARAMETER_GEOMETRY_FOUR1022_GRAM_SPECTRA',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=len(manifest),parameter_components=525568,queries=80,roles=20,maximum_error=maximum,rank_reports=rank_reports,
    actual_AP_Jacobians_JVPs_and_head_forwards_checked_on_server_only=True,no_global_or_image_feasibility_claim=True,no_new_fitted_model=True)
(E/'independent_local_audit.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8');print(json.dumps(r))
