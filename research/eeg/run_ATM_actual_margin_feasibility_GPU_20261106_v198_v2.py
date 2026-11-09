"""Two fixed actual parameter-step feasibility probes; restore every intervention.

Classical constrained optimization is a diagnostic tool here, not a novel method.
Protection is checked on actual per-query cosine margins, then on all seen T.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import datetime,fcntl,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
plan=json.loads(Path(sys.argv[1]).read_bytes());sys.path.insert(0,str(Path(plan['parent_policy']).parent))
import atm_response_repair_contract_20261105_v196 as C
c=C.setup(plan['parent_policy']);q,K,tc=c['q'],c['K'],c['train'];out=Path(plan['output_root']);started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==plan['worker'];before=C.resources(c,True)
job=K.read(c['repo']/'audit'/(plan['job_name']+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==tc['physical']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sys.path[:0]=tc['physical']['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
import torch.nn.functional as F
from models.atms import ATMS
from safetensors import safe_open
import atm_response_contrast_repair_core_20261105_v195 as H
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory)
def guard():
    assert time.monotonic()-started<850 and time.time()<plan['absolute_deadline_epoch']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<10<<30 and torch.cuda.max_memory_allocated()<20<<30
    assert K.allocated(out.parent)<plan['maximum_bytes'];used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
    fs=os.statvfs(out);assert fs.f_bavail*fs.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
def flat(vs):return torch.cat([v.reshape(-1) for v in vs])
def batch_margins(pred,target):
    score=F.normalize(pred,dim=1)@F.normalize(target,dim=1).T;n=len(pred);ii=torch.arange(n,device=pred.device)
    own=score.diag();same=ii[:,None]//10==ii[None,:]//10;other=~torch.eye(n,device=pred.device,dtype=torch.bool)
    within=score.masked_fill(~(same&other),-torch.inf);whole=score.masked_fill(~other,-torch.inf)
    wi=within.argmax(1);bi=whole.argmax(1)
    return torch.cat([own-score[ii,wi],own-score[ii,bi]]),wi,bi
try:
    src={k:torch.load(K.checked(v),map_location='cpu',weights_only=True,mmap=True)['model'] for k,v in q['models'].items()}
    initial={n:src['NN'][n] if n.startswith('proj_eeg.') else v for n,v in src['CC'].items()}
    xall=torch.load(K.checked(tc['physical']['EEG']),map_location='cpu',weights_only=True,mmap=True)
    vall=torch.load(K.checked(tc['physical']['visual_target']),map_location='cpu',weights_only=True,mmap=True);T=np.asarray(tc['T']);assert len(T)==13230
    visual_all=F.normalize(vall[T].cuda(),dim=1)
    anchor=ATMS().cuda().eval().requires_grad_(False);anchor.load_state_dict(src['NN'],strict=True)
    net=ATMS().cuda();net.load_state_dict(initial,strict=True);allparams=H.prepare_student(net)
    def all_T():
        guard();values=[]
        with torch.inference_mode():
            for lo in range(0,13230,160):
                ids=T[lo:lo+160];values.append(F.normalize(net(xall[ids].cuda(),torch.ones(len(ids),device='cuda',dtype=torch.long)),dim=1))
            pred=torch.cat(values);result=[]
            for lo in range(0,13230,160):
                hi=min(lo+160,13230);ix=torch.arange(lo,hi,device='cuda');score=pred[lo:hi]@visual_all.T;own=score[torch.arange(hi-lo,device='cuda'),ix].clone()
                score[torch.arange(hi-lo,device='cuda'),ix]=-torch.inf
                whole=own-score.max(1).values;same=ix[:,None]//10==torch.arange(13230,device='cuda')[None,:]//10
                within=own-score.masked_fill(~same,-torch.inf).max(1).values
                result.append(torch.stack([own,within,whole],dim=1).cpu().numpy())
        return np.concatenate(result)
    baseline_T=all_T();assert np.array_equal(baseline_T,np.load(out.parent/'baseline_T.npz')['metrics']);np.savez_compressed(out/'baseline_T.npz',T_rows=T,metrics=baseline_T)
    Qdone=K.read(Path(q['project_root'])/'Qbank/completion.json');Qbank=np.load(K.checked(Qdone['bank']));w={};kv=[]
    with safe_open(K.checked(c['cv']['bq']['h94']),framework='pt',device='cpu') as f:
        for key,name in [('P_weight','image_proj.proj.weight'),('P_bias','image_proj.proj.bias'),('P_gamma','image_proj.norm.weight'),('P_beta','image_proj.norm.bias')]:w[key]=f.get_tensor(name).half().float().cuda()
        for lay in Qdone['layouts']:
            ix=lay['ip_index'];kv.append({k:f.get_tensor(f'ip_adapter.{ix}.{n}.weight').half().float().cuda() for k,n in [('K','to_k_ip'),('V','to_v_ip')]})
    rng=np.random.default_rng(1729);schedule=np.concatenate([rng.permutation(280) for _ in range(3)])[:830]
    order=np.random.default_rng(1739).permutation(1323);results=[];temporary_adam=0;temporary_weight_interventions=0
    for batch_index in [0,82]:
        guard();net.load_state_dict(initial,strict=True);allparams=H.prepare_student(net);lo=16*batch_index;pos=(order[lo:lo+16,None]*10+np.arange(10)).reshape(-1);ids=T[pos];n=len(ids)
        x=xall[ids].cuda();visual=vall[ids].cuda();sid=torch.ones(n,device='cuda',dtype=torch.long);unit=int(schedule[batch_index]);ti,li=divmod(unit,70)
        bank=Qbank[f'l{li:02d}'][:,ti];heads,dim=bank.shape[1],bank.shape[-1];Q=torch.tensor(bank.transpose(1,0,2,3).reshape(1,heads,8,dim).copy(),device='cuda').float();weights={**w,**kv[li]}
        with torch.no_grad():old=H.response(anchor(x,sid),Q,weights);vresp=H.response(visual,Q,weights);scale=(vresp-vresp.mean(0)).square().mean()
        pred=net(x,sid);response=H.response(pred,Q,weights);loss=(response-old).square().mean()/scale;mv,wi,bi=batch_margins(pred,visual)
        gs=torch.autograd.grad(loss,allparams,allow_unused=True,retain_graph=True);connected=[i for i,g in enumerate(gs) if g is not None];params=[allparams[i] for i in connected];gR=flat([gs[i] for i in connected]).detach().double()
        theta=flat(params).detach().clone();assert theta.numel()==545930;G=torch.empty((2*n,len(theta)),device='cuda',dtype=torch.float64)
        for i in range(2*n):
            guard();gg=torch.autograd.grad(mv[i],params,retain_graph=True);G[i]=flat(gg).detach().double()
        cpu=ATMS().double();cpu.load_state_dict(initial,strict=True);cp_all=H.prepare_student(cpu);cp=[cp_all[i] for i in connected];cy=cpu(x.cpu().double(),sid.cpu());cv=visual.cpu().double();cw={k:v.cpu().double() for k,v in weights.items()};cq=Q.cpu().double()
        cmatrix=F.normalize(cy,dim=1)@F.normalize(cv,dim=1).T;ci=torch.arange(n);cvectors=[cmatrix.diag()-cmatrix[ci,wi.cpu()],cmatrix.diag()-cmatrix[ci,bi.cpu()]];cm=torch.cat(cvectors)
        CPU_proof=[]
        for i in [0,2*n-1]:
            cg=flat(torch.autograd.grad(cm[i],cp,retain_graph=True));error=float((cg-G[i].cpu()).norm()/cg.norm().clamp_min(1e-20));assert error<1e-4;CPU_proof.append(dict(constraint=i,gradient_relative_L2=error))
        cr=H.response(cy,cq,cw);cl=(cr-old.cpu().double()).square().mean()/scale.cpu().double();cg=flat(torch.autograd.grad(cl,cp));errR=float((cg-gR.cpu()).norm()/cg.norm().clamp_min(1e-20));assert errR<1e-4
        del cpu,cp,cp_all,cy,cmatrix,cm,cvectors,cr,cl,cg,gg,gs
        opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);offset=0
        for p in params:p.grad=gR[offset:offset+p.numel()].reshape(p.shape).float().clone();offset+=p.numel()
        torch.nn.utils.clip_grad_norm_(params,1.);opt.step();temporary_adam+=1;adam=(flat(params).detach()-theta).double();del opt
        def assign(direction,alpha=1.):
            vec=theta+direction.float()*alpha;offset=0
            with torch.no_grad():
                for p in params:p.copy_(vec[offset:offset+p.numel()].reshape(p.shape));offset+=p.numel()
        assign(torch.zeros_like(adam));assert torch.equal(flat(params),theta)
        rownorm=G.norm(dim=1);assert bool((rownorm>1e-12).all());G/=rownorm[:,None];A=(G@G.T).cpu().numpy();b=(G@adam).cpu().numpy();an=float(adam.norm())
        sol=minimize(lambda lam:(.5*lam@(A@lam)+b@lam,A@lam+b),np.zeros(2*n),jac=True,bounds=[(0,None)]*(2*n),method='L-BFGS-B',options={'maxiter':10000,'ftol':2e-15,'gtol':1e-12,'maxls':50})
        initial_solver_iterations=sol.nit
        sol=minimize(lambda lam:(.5*lam@(A@lam)+b@lam,A@lam+b),sol.x,jac=True,bounds=[(0,None)]*(2*n),method='L-BFGS-B',options={'maxiter':10000,'ftol':0.,'gtol':1e-13,'maxls':100})
        lam=sol.x;direction=adam+G.T@torch.tensor(lam,device='cuda');slopes=(G@direction).cpu().numpy();dual=A@lam+b
        primal_error=float(np.max(abs(slopes-dual)));violation=float(max(0,-slopes.min()));complement=float(abs(lam*dual).max());dn=float(direction.norm())
        valid=bool(primal_error<1e-7*an and violation<1e-7*an and complement<1e-7*an*an and dn<=an*(1+1e-6))
        proof=dict(initial_solver_iterations=initial_solver_iterations,solver_success=bool(sol.success),solver_message=str(sol.message),solver_iterations=sol.nit,KKT_valid=valid,minimum_normalized_slope=float(slopes.min()),primal_dual_max_error=primal_error,complementarity_max=complement,adam_norm=an,projected_norm=dn,norm_retained=dn/an,response_gradient_dot_adam=float(gR@adam),response_gradient_dot_projected=float(gR@direction),CPU_gradient=CPU_proof,CPU_response_gradient_relative_L2=errR)
        np.savez_compressed(out/f'batch{batch_index}_QP.npz',A=A,b=b,lam=lam,actual_Gd=slopes,row_norm=rownorm.cpu().numpy(),adam_norm=np.array(an),direction_norm=np.array(dn),gR_dot_adam=np.array(float(gR@adam)),gR_dot_direction=np.array(float(gR@direction)))
        base_margin=mv.detach().cpu().numpy();base_loss=float(loss.detach());del mv,pred,response,loss
        def evaluate(d,alpha):
            global temporary_weight_interventions
            guard();assign(d,alpha);temporary_weight_interventions+=1
            with torch.inference_mode():
                y=net(x,sid);m,_,_=batch_margins(y,visual);vl=float((H.response(y,Q,weights)-old).square().mean()/scale);ma=m.cpu().numpy();diff=ma-base_margin
            return dict(alpha=alpha,loss=vl,relative_loss_improvement=(base_loss-vl)/base_loss,min_margin_change=float(diff.min()),mean_margin_change=float(diff.mean()),decreased_over_tolerance=int((diff<-1e-6).sum()),negative_flips=int(((base_margin>1e-6)&(ma<0)).sum()),effective_step_norm=float(d.norm())*alpha,actual_FP32_parameter_step_norm=float((flat(params).detach()-theta).double().norm())),ma
        line=[];accepted=None
        if valid:
            for power in range(11):
                alpha=2.**(-power);event,ma=evaluate(direction,alpha);event['power']=power;line.append(event)
                if event['min_margin_change']>=-1e-6 and event['relative_loss_improvement']>1e-7:accepted=alpha;break
        cases={};finite={};all_values={}
        choices=[('raw_Adam',adam,1.),('projected_full',direction,1.)] if valid else [('raw_Adam',adam,1.)]
        if accepted is not None:choices += [('accepted_projected',direction,accepted),('same_norm_raw',adam,dn/an*accepted)]
        for name,d,alpha in choices:
            event,ma=evaluate(d,alpha);cases[name]=event;finite[name]=ma;actual=all_T();all_values[name]=actual
            if name=='raw_Adam':assert np.array_equal(actual,np.load(out.parent/f'batch{batch_index}_fullT.npz')['raw_Adam'])
            diff=actual-baseline_T;cases[name]['full_T']=dict(queries=13230,mean_changes=diff.mean(0).tolist(),within_margin_decreased_over_tolerance=int((diff[:,1]<-1e-6).sum()),global_margin_decreased_over_tolerance=int((diff[:,2]<-1e-6).sum()),within_negative_flips=int(((baseline_T[:,1]>1e-6)&(actual[:,1]<0)).sum()),global_negative_flips=int(((baseline_T[:,2]>1e-6)&(actual[:,2]<0)).sum()),minimum_changes=diff.min(0).tolist())
        assign(torch.zeros_like(adam));assert all(torch.equal(v.cpu(),initial[k]) for k,v in net.state_dict().items());assert np.array_equal(all_T(),baseline_T)
        np.savez_compressed(out/f'batch{batch_index}_actual.npz',T_rows=ids,T_positions=pos,baseline_margin=base_margin,**{k:v for k,v in finite.items()});np.savez_compressed(out/f'batch{batch_index}_fullT.npz',**all_values)
        result=dict(batch=batch_index,queries=n,unit=unit,connected_parameters=len(theta),constraints=2*n,base_response_loss=base_loss,QP=proof,line_search=line,accepted_alpha=accepted,cases=cases,all_weights_and_fullT_outputs_exactly_restored=True)
        results.append(result);K.write(out/'progress.json',dict(completed_batches=[x['batch'] for x in results],new_training_updates=0));guard()
        del G,A,direction,adam,params,allparams;allparams=H.prepare_student(net)
    guard();after=C.resources(c)
    for pin in q['models'].values():K.checked(pin)
    K.write(out/'completion.json',dict(status='COMPLETE_TWO_ACTUAL_PARAMETER_CONE_FEASIBILITY_PROBES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(sys.argv[1]),results=results,all_T=13230,full_T_metric_names=['paired_visual_cosine','within10_minimum_margin','full_T_gallery_minimum_margin'],new_training_updates=0,temporary_Adam_steps_discarded=temporary_adam,temporary_weight_interventions_discarded=temporary_weight_interventions,new_models=0,new_images=0,all_restored=True,not_new_method_or_generalization=True,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources=after))
except BaseException as exc:
    restored=False
    if 'net' in globals() and 'initial' in globals():
        net.load_state_dict(initial,strict=True);restored=all(torch.equal(v.cpu(),initial[k]) for k,v in net.state_dict().items())
    K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),weights_restored=restored,seconds=time.monotonic()-started));raise
