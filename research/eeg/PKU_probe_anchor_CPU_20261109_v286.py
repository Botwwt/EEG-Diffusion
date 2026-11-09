"""Real EEG native identity plus independent angular-cap and meanbound checks, no optimizer."""
import json,math,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_anchor_engineering_v286';out.mkdir();start=time.monotonic()
def write(n,r):
    p=out/n;b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode()
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
try:
    import torch
    from PKU_anchored_encoder_core_20261109_v286 import make_model,constrain,posterior_indices,sha
    torch.set_num_threads(4);torch.manual_seed(1729)
    split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());rows=split['image_rows']['T'][:32]
    raw=torch.load(R/'data/recovery_20261108_v236/sub-03_train_mean4.pt',map_location='cpu',weights_only=True,mmap=True);post,_=posterior_indices();x=raw[rows][:,post,:].clone();del raw
    m=make_model('C_pair',device='cpu');anchor=m.anchor;versions=[p._version for p in anchor.parameters()]
    with torch.no_grad():old=anchor(x);new=m(x)
    assert torch.equal(old,new),'Actual zero difference must retain the native FP32 function exactly'
    loss=(m(x)*torch.randn_like(new)).mean();loss.backward();gradient=float(m.candidate.proj_in.weight.grad.norm());assert gradient>0 and all(p.grad is None for p in anchor.parameters())
    assert [p._version for p in anchor.parameters()]==versions
    rho=.25;eta=math.sqrt(2-2/math.sqrt(1+rho*rho));checks=[]
    for magnitude in [0.,.01,1.,100.,1e6]:
        torch.manual_seed(1729);u0=torch.randn(257,1024,dtype=torch.float64);u0-=u0.mean(1,keepdim=True);u0=u0/u0.norm(dim=1,keepdim=True)*32
        # Production uses FP32 deliberately; independent reference is FP64 and tests finite numeric tolerance.
        w=torch.linspace(.8,1.2,1024);b=torch.linspace(-.1,.1,1024);old=u0.float()*w+b;q=old+magnitude*torch.randn_like(old)
        changed=constrain(q,old,w,b);a=(old-b)/w;c=(changed-b)/w;yn=a/a.norm(dim=1,keepdim=True);yp=c/c.norm(dim=1,keepdim=True)
        cosine=(yn*yp).sum(1);chord=(yn-yp).norm(dim=1);normerr=(c.norm(dim=1)-a.norm(dim=1)).abs().max();meanbound=float(yn.mean(0).norm())+eta
        assert float(cosine.min())>=1/math.sqrt(1+rho*rho)-2e-6 and float(chord.max())<=eta+2e-6 and float(normerr)<2e-5
        assert float(yp.mean(0).norm())<=meanbound+2e-6
        pair0=(yn[:-1]-yn[1:]).norm(dim=1);pair1=(yp[:-1]-yp[1:]).norm(dim=1);assert float((pair1-pair0).abs().max())<=2*eta+2e-6
        u=u0;du=(q.double()-b.double())/w.double()-u;du-=du.mean(1,keepdim=True);du-=(du*u).sum(1,keepdim=True)/u.square().sum(1,keepdim=True)*u
        delta=rho*u.norm(dim=1,keepdim=True)*du/(u.square().sum(1,keepdim=True)+du.square().sum(1,keepdim=True)).sqrt();ref=(u+delta)*u.norm(dim=1,keepdim=True)/(u+delta).norm(dim=1,keepdim=True)
        error=float((c.double()-ref).abs().max());assert error<2e-5
        checks.append(dict(candidate_perturbation_scale=magnitude,min_cosine=float(cosine.min()),max_unit_chord=float(chord.max()),max_norm_error=float(normerr),independent_FP64_coordinate_max_abs=error,unit_mean_norm=float(yp.mean(0).norm()),native_plus_eta=meanbound))
    assert not torch.cuda.is_initialized()
    write('completion.json',dict(status='PASS_REAL_EEG_NATIVE_IDENTITY_GRADIENT_AND_GLOBAL_CAP_ALGEBRA',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,raw_EEG_queries=32,FP32_bitexact_native=True,firstlayer_gradient_norm=gradient,
        rho=rho,eta=eta,angle_cap_radians=math.atan(rho),checks=checks,new_optimizer_updates=0,new_images=0,source_sha256=sha(__file__),native_model_parameters=sum(p.numel() for p in m.anchor.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),no_image_quality_or_novelty_claim=True))
    print('PASS_ANCHORED_EEG_ENGINEERING_NO_FIT',flush=True)
except BaseException as e:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
