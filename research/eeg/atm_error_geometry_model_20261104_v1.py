"""Same author 9,675,648-parameter prior; affine analytic paths are fixed."""
import importlib.util
import torch
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

def original_module(path):
    spec=importlib.util.spec_from_file_location('error_geometry_original_prior',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def tensors(stats,device):
    return {k:torch.tensor(stats[k],device=device,dtype=torch.float32) for k in ('basis','mean_error','mean_error_rotated','eigenvalues','isotropic_eigenvalues','wrong_eigenvalues')}

def parameters(mode,z,t,c,stats,alphas):
    a=alphas[t].sqrt()[:,None];b=(1-alphas[t]).sqrt()[:,None]
    if mode=='native_v':return dict(a=a,b=b,network_input=z)
    key={'iso_error':'isotropic_eigenvalues','paired_error':'eigenvalues','wrong_error':'wrong_eigenvalues'}[mode]
    lam=stats[key][None];s=(a*a*lam+b*b).sqrt();A=a*lam.sqrt()/s;B=b/s;m=c+stats['mean_error_rotated']
    return dict(a=a,b=b,m=m,lam=lam,s=s,A=A,B=B,network_input=(z-a*m)/s)

def target(mode,f,c,noise,t,stats,alphas):
    a=alphas[t].sqrt()[:,None];b=(1-alphas[t]).sqrt()[:,None];z=a*f+b*noise;p=parameters(mode,z,t,c,stats,alphas)
    y=a*noise-b*f if mode=='native_v' else p['A']*noise-p['B']*(f-p['m'])/p['lam'].sqrt()
    return z,p,y

def clean(mode,net,z,t,c,stats,alphas):
    p=parameters(mode,z,t,c,stats,alphas);v=net(p['network_input'],t.float(),c)
    if mode=='native_v':d=p['a']*z-p['b']*v
    else:d=p['m']+p['lam'].sqrt()*(p['A']*p['network_input']-p['B']*v)
    return d,v,p

@torch.no_grad()
def sample(mode,net,c_original,stats,seeds,*,trace=False):
    device=next(net.parameters()).device;assert not net.training and len(seeds)==len(c_original)
    c=c_original@stats['basis'];generators=[torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    scheduler=DDPMScheduler(prediction_type='sample',clip_sample=False);scheduler.set_timesteps(50,device=device);alphas=scheduler.alphas_cumprod.to(device)
    z=randn_tensor(c.shape,generator=generators,device=device,dtype=torch.float32);states=[]
    for tt in scheduler.timesteps:
        t=torch.full((len(c),),int(tt),device=device,dtype=torch.long);d,v,p=clean(mode,net,z,t,c,stats,alphas)
        if trace:states.append(dict(t=int(tt),state=z[:2].cpu().numpy(),prediction=v[:2].cpu().numpy(),clean=d[:2].cpu().numpy()))
        z=scheduler.step(d,int(tt),z,generator=generators).prev_sample
        assert torch.isfinite(z).all()
    return z@stats['basis'].T,states
