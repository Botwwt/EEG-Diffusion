"""Independent source-vs-batched-rank fixtures before real candidate image scoring."""
import json,os,socket,sys,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');sys.path.insert(0,str(R/'package_cache/evaluation_runtime_v275'));start=time.monotonic()
import numpy as np
import torch
from PKU_EEG_evaluation_core_20261109_v279 import ranked,paper_source_scope,sha
torch.set_num_threads(4);torch.manual_seed(1729);scope=paper_source_scope();concepts=['same']*10;eye=torch.eye(10);image=eye.reshape(10,1,2,5);identity=lambda x:x.flatten(1);checks=[]
for name,q in [('identity',eye),('deranged',torch.roll(eye,1,0)),('mixed',.25*eye+.75*torch.roll(eye,1,0)),('duplicates',torch.cat([eye[:9],eye[:1]])),('random',torch.randn(10,10))]:
    actual=ranked(q,eye,list(range(10)),concepts,'paper');expected,den=scope['two_way_identification'](q.reshape(10,1,2,5),image,identity,lambda x:x,return_avg=False);error=float(np.max(np.abs(actual['two_afc'].numpy()-np.asarray(expected)/den)));assert error<1e-7
    checks.append(dict(fixture=name,primary_rank_max_abs=error))
# Preserve the old independent-normalization behavior as a diagnostic, not a new metric.
old_differences=[]
for fixture,z in [('identity',eye),('deranged',torch.roll(eye,1,0)),('mixed',.25*eye+.75*torch.roll(eye,1,0)),('duplicates',torch.cat([eye[:9],eye[:1]]))]:
    a=z.double();a-=a.mean(1,keepdim=True);a/=a.norm(dim=1,keepdim=True);b=eye.double();b-=b.mean(1,keepdim=True);b/=b.norm(dim=1,keepdim=True);s=(a@b.T).clamp(-1,1);c=s.diag()[:,None];mask=~torch.eye(10,dtype=torch.bool);old=((s<c)&mask).sum(1).numpy()/9
    expected,den=scope['two_way_identification'](z.reshape(10,1,2,5),image,identity,lambda x:x,return_avg=False)
    old_differences.append(dict(fixture=fixture,old_two_afc=old.tolist(),author_two_afc=(np.asarray(expected)/den).tolist(),mismatched_queries=int(np.count_nonzero(old!=np.asarray(expected)/den))))
checks.append(dict(old_implementation_diagnosis=old_differences,no_epsilon_tie_rule_introduced=True))
g=torch.randn(830,32);q=torch.randn(4,32);concepts=[str(i//10) for i in range(830)];actual,scores=ranked(q,g,[0,10,20,30],concepts,'paper',return_scores=True)
reference=np.corrcoef(g.numpy(),q.numpy())[:830,830:].T;error=float(np.abs(reference-scores.numpy()).max());assert error<1e-12
correct=reference[np.arange(4),[0,10,20,30]][:,None];mask=np.ones_like(reference,dtype=bool);mask[np.arange(4),[0,10,20,30]]=False;expected=((reference<correct)&mask).sum(1)/829
assert np.allclose(expected,actual['two_afc'].numpy(),atol=1e-7,rtol=0)
checks.append(dict(fixture='all830_real_gallery_numpy_Pearson',similarity_max_abs=error,explicit_positive_indices=True))
dino=ranked(g[:4],g,[0,1,2,3],concepts,'DINO');assert torch.allclose(dino['correct_similarity'],torch.ones(4),atol=1e-6,rtol=0) and (dino['two_afc']==1).all()
x=torch.full((1,3,32,32),.2,dtype=torch.float64);y=torch.full_like(x,.7);actual=scope['ssim'](x,y);expected=(2*.2*.7+.01**2)/(.2**2+.7**2+.01**2);assert abs(actual-expected)<1e-10
checks.append(dict(fixture='original_SSIM_analytic_constant',measured=float(actual),analytic=float(expected)))
assert not torch.cuda.is_initialized();out=R/'runs/EEG_metric_core_v279';out.mkdir();r=dict(status='PASS_SOURCE_DIRECTION_TIES_EXPLICIT_GALLERY_AND_SSIM_FIXTURES',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,checks=checks,real_candidate_images_read=0,GPU_initialized=False,source_sha256=sha(__file__))
p=out/'completion.json';b=(json.dumps(r,indent=2)+'\n').encode()
with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
print(json.dumps(r),flush=True)
