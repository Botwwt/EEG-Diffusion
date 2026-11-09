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
    assert K.allocated(out)<plan['maximum_bytes'];used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
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
    baseline_T=all_T();oldroot=Path(plan['original_root']);assert np.array_equal(baseline_T,np.load(oldroot/'baseline_T.npz')['metrics']);np.savez_compressed(out/'baseline_T.npz',T_rows=T,metrics=baseline_T)
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
        base_margin=mv.detach().cpu().numpy();base_loss=float(loss.detach());del mv,pred,response,loss
        G0=G.clone();rownorm=G.norm(dim=1);Gn=G/rownorm[:,None];rhs=torch.zeros(2*n,device='cuda',dtype=torch.float64)
        labels=[(i,0,int(wi[i])) for i in range(n)]+[(i,1,int(bi[i])) for i in range(n)];known=set(labels)
        oq=np.load(oldroot/'recovery_v2'/f'batch{batch_index}_QP.npz');A0=(Gn@Gn.T).cpu().numpy();b0=(Gn@adam).cpu().numpy()
        assert np.array_equal(A0,oq['A']) and np.array_equal(b0,oq['b'])
        d0=adam+Gn.T@torch.tensor(oq['lam'],device='cuda');assert abs(float(d0.norm())-float(oq['direction_norm']))<1e-12
        def evaluate(d,alpha):
            global temporary_weight_interventions
            guard();assign(d,alpha);temporary_weight_interventions+=1
            with torch.inference_mode():
                y=net(x,sid);score=F.normalize(y,dim=1)@F.normalize(visual,dim=1).T;ii=torch.arange(n,device='cuda')
                m,wn,bn=batch_margins(y,visual);fixed=torch.cat([score.diag()-score[ii,wi],score.diag()-score[ii,bi]])
                actualstep=(flat(params).detach()-theta).double();lin=G0@actualstep
                delta=m.double()-torch.tensor(base_margin,device='cuda').double();nonlinear=fixed.double()-torch.tensor(base_margin,device='cuda').double()-lin;switch=fixed.double()-m.double()
                assert float(switch.min())>-1e-7 and float((delta-lin-nonlinear+switch).abs().max())<1e-12
                vl=float((H.response(y,Q,weights)-old).square().mean()/scale);ma=m.cpu().numpy();df=ma-base_margin
                e=dict(alpha=alpha,loss=vl,relative_loss_improvement=(base_loss-vl)/base_loss,min_margin_change=float(df.min()),mean_margin_change=float(df.mean()),decreased_over_tolerance=int((df<-1e-6).sum()),negative_flips=int(((base_margin>1e-6)&(ma<0)).sum()),effective_step_norm=float(d.norm())*alpha,actual_FP32_parameter_step_norm=float(actualstep.norm()),old_pair_violations=int((fixed.cpu().numpy()-base_margin<-1e-6).sum()),changed_hard_negative=int((torch.cat([wn,bn])!=torch.cat([wi,bi])).sum()),switch_loss_mean=float(switch.mean()),nonlinear_old_pair_mean=float(nonlinear.mean()),linear_margin_mean=float(lin.mean()))
                detail=torch.stack([delta,lin,nonlinear,switch,fixed.double()-torch.tensor(base_margin,device='cuda').double()],dim=1).cpu().numpy()
            return e,ma,wn.cpu().tolist(),bn.cpu().tolist(),detail
        original,original_m,_,_,original_detail=evaluate(d0,1.)
        assert np.array_equal(original_m,np.load(oldroot/'recovery_v2'/f'batch{batch_index}_actual.npz')['projected_full'])
        histories=[];direction=d0;sol_lam=oq['lam'];last_valid=True
        for cut in range(6):
            guard();event,ma,wn,bn,detail=evaluate(direction,1.);new=[]
            if cut<5:
                for kind,ns in enumerate([wn,bn]):
                    for i,j in enumerate(ns):
                        if ma[kind*n+i]-base_margin[kind*n+i]<-1e-6 and (i,kind,j) not in known:new.append((i,kind,j))
            histories.append(dict(round=cut,constraints=len(labels),full_step=event,new_pairs=[list(v) for v in new]))
            np.savez_compressed(out/f'batch{batch_index}_cut{cut}_accounting.npz',margin=ma,components=detail)
            if not new:break
            assign(torch.zeros_like(adam));y=F.normalize(net(x,sid),dim=1);vv=F.normalize(visual,dim=1);newg=[];newrhs=[]
            for i,kind,j in new:
                guard();pair=(y[i]*(vv[i]-vv[j])).sum();gr=flat(torch.autograd.grad(pair,params,retain_graph=True)).detach().double();rn=gr.norm();assert float(rn)>1e-12
                slack=float(pair.detach())-float(base_margin[kind*n+i]);assert slack>-1e-6
                # FP32 dot versus matrix reduction can differ slightly; preserve actual slack, do not clamp it to fabricate feasibility.
                newg.append(gr/rn);newrhs.append(-slack/float(rn));labels.append((i,kind,j));known.add((i,kind,j))
            del y,gr,pair;Gn=torch.cat([Gn,torch.stack(newg)]);rhs=torch.cat([rhs,torch.tensor(newrhs,device='cuda',dtype=torch.float64)]);del newg
            AA=(Gn@Gn.T).cpu().numpy();bb=(Gn@adam-rhs).cpu().numpy();warm=np.concatenate([sol_lam,np.zeros(len(new))])
            sol=minimize(lambda lam:(.5*lam@(AA@lam)+bb@lam,AA@lam+bb),warm,jac=True,bounds=[(0,None)]*len(warm),method='L-BFGS-B',options={'maxiter':10000,'ftol':0.,'gtol':1e-13,'maxls':100})
            sol_lam=sol.x;direction=adam+Gn.T@torch.tensor(sol_lam,device='cuda');slope=(Gn@direction-rhs).cpu().numpy();dual=AA@sol_lam+bb;an=float(adam.norm())
            err=float(abs(slope-dual).max());vio=float(max(0,-slope.min()));comp=float(abs(sol_lam*dual).max());last_valid=bool(err<1e-7*an and vio<1e-7*an and comp<1e-7*an*an)
            histories[-1]['next_QP']=dict(success=bool(sol.success),message=str(sol.message),iterations=sol.nit,KKT_valid=last_valid,violation=vio,complementarity=comp,primal_dual_error=err)
            np.savez_compressed(out/f'batch{batch_index}_cut{cut}_QP.npz',A=AA,b=bb,lam=sol_lam,rhs=rhs.cpu().numpy(),actual_slack=slope,adam_norm=np.array(an),direction_norm=np.array(float(direction.norm())),labels=np.array(labels,dtype=np.int32))
            if not last_valid:break
        line=[];accepted=None;dn=float(direction.norm());an=float(adam.norm())
        if last_valid:
            for power in range(11):
                event,ma,_,_,detail=evaluate(direction,2.**(-power));event['power']=power;line.append(event)
                if event['min_margin_change']>=-1e-6 and event['relative_loss_improvement']>1e-7:accepted=2.**(-power);break
        choices=[('original_cone_full',d0,1.)]
        if last_valid:choices.append(('active_set_full',direction,1.))
        if accepted is not None:choices += [('active_set_accepted',direction,accepted),('same_norm_original_cone',d0,dn/float(d0.norm())*accepted),('same_norm_raw',adam,dn/an*accepted)]
        cases={};finite={};all_values={};details={}
        for name,direction_case,alpha in choices:
            event,ma,_,_,detail=evaluate(direction_case,alpha);cases[name]=event;finite[name]=ma;details[name]=detail;actual=all_T();all_values[name]=actual
            if name=='original_cone_full':assert np.array_equal(actual,np.load(oldroot/'recovery_v2'/f'batch{batch_index}_fullT.npz')['projected_full'])
            diff=actual-baseline_T;cases[name]['full_T']=dict(queries=13230,mean_changes=diff.mean(0).tolist(),within_margin_decreased_over_tolerance=int((diff[:,1]<-1e-6).sum()),global_margin_decreased_over_tolerance=int((diff[:,2]<-1e-6).sum()),within_negative_flips=int(((baseline_T[:,1]>1e-6)&(actual[:,1]<0)).sum()),global_negative_flips=int(((baseline_T[:,2]>1e-6)&(actual[:,2]<0)).sum()),minimum_changes=diff.min(0).tolist())
        assign(torch.zeros_like(adam));assert all(torch.equal(v.cpu(),initial[k]) for k,v in net.state_dict().items());assert np.array_equal(all_T(),baseline_T)
        np.savez_compressed(out/f'batch{batch_index}_actual.npz',T_rows=ids,T_positions=pos,baseline_margin=base_margin,**finite)
        np.savez_compressed(out/f'batch{batch_index}_fullT.npz',**all_values);np.savez_compressed(out/f'batch{batch_index}_components.npz',**details)
        results.append(dict(batch=batch_index,queries=n,unit=unit,connected_parameters=len(theta),CPU_gradients=CPU_proof,CPU_response_gradient_relative_L2=errR,base_response_loss=base_loss,original_full=original,cut_history=histories,last_KKT_valid=last_valid,line_search=line,accepted_alpha=accepted,cases=cases,all_weights_and_fullT_outputs_exactly_restored=True))
        K.write(out/'progress.json',dict(completed_batches=[x['batch'] for x in results],new_training_updates=0));guard()
        del G,G0,Gn,params,allparams;allparams=H.prepare_student(net)
    guard();after=C.resources(c)
    for pin in q['models'].values():K.checked(pin)
    K.write(out/'completion.json',dict(status='COMPLETE_TWO_ACTIVE_NEGATIVE_ACCOUNTING_PROBES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(sys.argv[1]),results=results,all_T=13230,full_T_metric_names=['paired_visual_cosine','within10_minimum_margin','full_T_gallery_minimum_margin'],component_names=['actual_minimum_margin_change','linear_old_pair','nonlinear_old_pair','hard_negative_switch_penalty','actual_old_pair_margin_change'],new_training_updates=0,temporary_Adam_steps_discarded=temporary_adam,temporary_weight_interventions_discarded=temporary_weight_interventions,new_models=0,new_images=0,all_restored=True,not_new_method_or_generalization=True,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources=after))
except BaseException as exc:
    restored=False
    if 'net' in globals() and 'initial' in globals():
        net.load_state_dict(initial,strict=True);restored=all(torch.equal(v.cpu(),initial[k]) for k,v in net.state_dict().items())
    K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),weights_restored=restored,seconds=time.monotonic()-started));raise
