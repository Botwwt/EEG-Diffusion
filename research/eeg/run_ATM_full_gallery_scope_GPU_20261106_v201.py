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
    assert time.monotonic()-started<1100 and time.time()<plan['absolute_deadline_epoch']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<10<<30 and torch.cuda.max_memory_allocated()<20<<30
    assert K.allocated(out)<plan['maximum_bytes'];used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
    fs=os.statvfs(out);assert fs.f_bavail*fs.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
def flat(vs):return torch.cat([v.reshape(-1) for v in vs])
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
    def scope_pred(positions,model=None,device='cuda',dtype=torch.float32):
        model=net if model is None else model
        return torch.cat([model(xall[T[positions[j:j+160]]].to(device=device,dtype=dtype),torch.ones(len(positions[j:j+160]),device=device,dtype=torch.long)) for j in range(0,len(positions),160)])
    def margins(pred,positions):
        yy=F.normalize(pred,dim=1);score=yy@visual_all.T;ix=torch.tensor(positions,device='cuda');ii=torch.arange(len(ix),device='cuda');own=score[ii,ix]
        neg=score.clone();neg[ii,ix]=-torch.inf;same=ix[:,None]//10==torch.arange(13230,device='cuda')[None,:]//10
        within=neg.masked_fill(~same,-torch.inf);wi=within.argmax(1);gi=neg.argmax(1)
        return torch.cat([own-score[ii,wi],own-score[ii,gi]]),wi,gi
    def assign_delta(delta):
        global interventions
        vec=theta+delta.float();offset=0
        with torch.no_grad():
            for v in params:v.copy_(vec[offset:offset+v.numel()].reshape(v.shape));offset+=v.numel()
        interventions+=1
    def state_read(positions,floor):
        guard()
        with torch.no_grad():
            mm,wi,gi=margins(scope_pred(positions),positions);rv=H.response(net(x,sid),Q,weights);lossvalue=float((rv-old).square().mean()/scale)
        vals=mm.cpu().numpy();delta=vals.astype(np.float64)-floor.astype(np.float64)
        return dict(loss=lossvalue,relative_response_reduction=(base_loss-lossvalue)/base_loss,minimum_change=float(delta.min()),mean_change=float(delta.mean()),decreased_over_tolerance=int((delta<-1e-6).sum()),merit=float(np.square(np.minimum(delta,0)).sum())),vals
    def solve_step(positions,floor,proposal,label,proof):
        guard();pred=scope_pred(positions);mm,wi,gi=margins(pred,positions);count=len(positions);J=torch.empty((2*count,len(theta)),device='cuda',dtype=torch.float64)
        for row in range(2*count):
            guard();gr=flat(torch.autograd.grad(mm[row],params,retain_graph=True)).detach().double();J[row]=gr
        cpu_proofs=[]
        if proof:
            cpu=ATMS().double();cpu.load_state_dict({k:v.detach().cpu() for k,v in net.state_dict().items()},strict=True);ca=H.prepare_student(cpu);cp=[ca[i] for i in connected]
            cy=F.normalize(scope_pred(positions,cpu,'cpu',torch.float64),dim=1);vv=F.normalize(vall[T].double(),dim=1);ii=torch.tensor(positions);cw=wi.cpu();cg=gi.cpu()
            cm=torch.cat([(cy*(vv[ii]-vv[cw])).sum(1),(cy*(vv[ii]-vv[cg])).sum(1)])
            for row in [0,2*count-1]:
                z=flat(torch.autograd.grad(cm[row],cp,retain_graph=True));error=float((z-J[row].cpu()).norm()/z.norm().clamp_min(1e-20));assert error<1e-4;cpu_proofs.append(dict(row=row,relative_L2=error))
            del cpu,ca,cp,cy,vv,cm,z
        rownorm=J.norm(dim=1);assert bool((rownorm>1e-12).all());J/=rownorm[:,None]
        rhs=(torch.tensor(floor,device='cuda').double()-mm.detach().double())/rownorm
        AA=(J@J.T).cpu().numpy();bb=(J@proposal-rhs).cpu().numpy()
        sol=minimize(lambda lam:(.5*lam@(AA@lam)+bb@lam,AA@lam+bb),np.zeros(2*count),jac=True,bounds=[(0,None)]*(2*count),method='L-BFGS-B',options={'maxiter':10000,'ftol':0.,'gtol':1e-13,'maxls':100})
        lam=sol.x;step=proposal+J.T@torch.tensor(lam,device='cuda');slack=(J@step-rhs).detach().cpu().numpy();dual=AA@lam+bb;an=float(adam.norm());err=float(abs(slack-dual).max());violation=float(max(0,-slack.min()));comp=float(abs(lam*dual).max());valid=bool(err<1e-7*an and violation<1e-7*an and comp<1e-7*an*an)
        np.savez_compressed(out/(label+'_QP.npz'),A=AA,b=bb,lam=lam,rhs=rhs.detach().cpu().numpy(),actual_slack=slack,proposal_norm=np.array(float(proposal.norm())),step_norm=np.array(float(step.norm())),adam_norm=np.array(an),positions=positions,negative_positions=torch.cat([wi,gi]).cpu().numpy(),floor=floor,current_margin=mm.detach().cpu().numpy(),row_norm=rownorm.cpu().numpy())
        info=dict(label=label,constraints=2*count,success=bool(sol.success),iterations=int(sol.nit),KKT_valid=valid,violation=violation,complementarity=comp,primal_dual_error=err,step_norm=float(step.norm()),proposal_norm=float(proposal.norm()),CPU64_gradients=cpu_proofs)
        del pred,mm,J,gr;return step.detach(),info
    def full_stats(vals):
        df=vals.astype(np.float64)-baseline_T.astype(np.float64);ans=dict(mean_changes=df.mean(0).tolist(),minimum_changes=df.min(0).tolist(),decreased_over_tolerance=(df<-1e-6).sum(0).tolist(),increased_over_tolerance=(df>1e-6).sum(0).tolist())
        for j,key in [(1,'within'),(2,'global')]:
            oldv,newv=baseline_T[:,j],vals[:,j];ans[key]=dict(strict_top1_before=float((oldv>0).mean()),strict_top1_after=float((newv>0).mean()),strict_negative_flips=int(((oldv>0)&(newv<=0)).sum()),strict_positive_flips=int(((oldv<=0)&(newv>0)).sum()))
        return ans
    order=np.random.default_rng(1739).permutation(1323);results=[];temporary_adam=0;interventions=0
    for batch_index in [0,82]:
        guard();net.load_state_dict(initial,strict=True);allparams=H.prepare_student(net);lo=16*batch_index;pos=(order[lo:lo+16,None]*10+np.arange(10)).reshape(-1);ids=T[pos];n=len(ids)
        x=xall[ids].cuda();visual=vall[ids].cuda();sid=torch.ones(n,device='cuda',dtype=torch.long);unit=int(schedule[batch_index]);ti,li=divmod(unit,70);bank=Qbank[f'l{li:02d}'][:,ti];heads,dim=bank.shape[1],bank.shape[-1];Q=torch.tensor(bank.transpose(1,0,2,3).reshape(1,heads,8,dim).copy(),device='cuda').float();weights={**w,**kv[li]}
        with torch.no_grad():old=H.response(anchor(x,sid),Q,weights);vresp=H.response(visual,Q,weights);scale=(vresp-vresp.mean(0)).square().mean()
        pred=net(x,sid);response=H.response(pred,Q,weights);loss=(response-old).square().mean()/scale;base_loss=float(loss.detach());gs=torch.autograd.grad(loss,allparams,allow_unused=True);connected=[i for i,g in enumerate(gs) if g is not None];params=[allparams[i] for i in connected];gR=flat([gs[i] for i in connected]).detach().double();theta=flat(params).detach().clone();assert theta.numel()==545930
        opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);offset=0
        for p in params:p.grad=gR[offset:offset+p.numel()].reshape(p.shape).float().clone();offset+=p.numel()
        torch.nn.utils.clip_grad_norm_(params,1.);opt.step();temporary_adam+=1;adam=(flat(params).detach()-theta).double();del opt,gs,gR,pred,response,loss
        prior=Path(plan['prior_probe_root']);oldqp=np.load(oldroot/'recovery_v2'/f'batch{batch_index}_QP.npz');assert abs(float(adam.norm())-float(oldqp['adam_norm']))<1e-12
        assign_delta(adam);assert np.array_equal(all_T(),np.load(oldroot/'recovery_v2'/f'batch{batch_index}_fullT.npz')['raw_Adam']);assign_delta(torch.zeros_like(adam))
        previous=np.load(K.checked(plan['selection_sources'][str(batch_index)]))['nonlinear_accepted'];assert previous.shape==(13230,3)
        change=previous[:,1:]-baseline_T[:,1:]
        # Most harmed single margin per concept, then deterministic concept-index ties.
        damage=-(change.min(1).reshape(1323,10).min(1));excluded=set((pos//10).tolist());eligible=[i for i in range(1323) if i not in excluded]
        targeted=sorted(eligible,key=lambda i:(-float(damage[i]),i))[:16];random=np.random.default_rng(1729+batch_index).choice(np.array(eligible),16,replace=False).tolist()
        K.write(out/f'batch{batch_index}_selection.json',dict(targeted_concepts=targeted,random_concepts=random,excluded_concepts=sorted(excluded),selection_from_previous_seenT=True,damage=damage.tolist()))
        basecases={};fulltables={};finite={};mode_results=[]
        for mode,extras in [('full_gallery_only',[]),('targeted_extra_queries',targeted),('random_extra_queries',random)]:
            guard();assign_delta(torch.zeros_like(adam));positions=np.concatenate([pos,(np.asarray(extras,dtype=np.int64)[:,None]*10+np.arange(10)).reshape(-1)]);assert len(set(positions.tolist()))==len(positions)
            floor=np.concatenate([baseline_T[positions,1],baseline_T[positions,2]]);label=f'batch{batch_index}_{mode}'
            cn_check,cnm=state_read(positions,floor);assert cn_check['minimum_change']>=-1e-6
            direction,initial_qp=solve_step(positions,floor,adam,label+'_initial',True);history=[];feasible=False
            if initial_qp['KKT_valid']:
                for rr in range(6):
                    assign_delta(direction);event,ma=state_read(positions,floor);h=dict(round=rr,current=event);history.append(h)
                    if event['minimum_change']>=-1e-6:feasible=True;h['status']='FEASIBLE';break
                    if rr==5:h['status']='FIVE_CORRECTIONS_EXHAUSTED';break
                    current_delta=(flat(params).detach()-theta).double();correction,qi=solve_step(positions,floor,torch.zeros_like(adam),label+f'_restore{rr}',rr==0);h['QP']=qi
                    if not qi['KKT_valid']:h['status']='KKT_UNRESOLVED';break
                    rows=[];attempts=[];accepted=False
                    for power in range(11):
                        beta=2.**(-power);proposed=current_delta+beta*correction;assign_delta(proposed);e,v=state_read(positions,floor);passed=bool(e['minimum_change']>=-1e-6 or e['merit']<=event['merit']*(1-1e-4*beta));rows.append(v);attempts.append(dict(beta=beta,passed=passed,**e))
                        if passed:direction=proposed;accepted=True;break
                    h.update(status='CORRECTION_ACCEPTED' if accepted else 'FINITE_UNRESOLVED',search=attempts)
                    np.savez_compressed(out/(label+f'_restore{rr}_finite.npz'),floor=floor,before=ma,trials=np.stack(rows))
                    if not accepted:break
            # Report the full proposal and the first fixed feasible positive-response scale; no selection from image/test results.
            lines=[];accepted_alpha=None
            if initial_qp['KKT_valid']:
                for power in range(11):
                    alpha=2.**(-power);assign_delta(direction*alpha);e,v=state_read(positions,floor);lines.append(dict(power=power,alpha=alpha,**e))
                    if e['minimum_change']>=-1e-6 and e['relative_response_reduction']>1e-7:accepted_alpha=alpha;break
            choices=[('full',direction)]
            if accepted_alpha is not None:choices+=[('accepted',direction*accepted_alpha),('same_norm_raw',adam*(float(direction.norm())*accepted_alpha/float(adam.norm())))]
            cases={}
            for cname,delta in choices:
                assign_delta(delta);e,v=state_read(positions,floor);values=all_T();e.update(step_norm=float(delta.norm()),actual_FP32_step_norm=float((flat(params).detach()-theta).double().norm()),full_T=full_stats(values));cases[cname]=e;fulltables[mode+'__'+cname]=values;finite[mode+'__'+cname]=v
            np.savez_compressed(out/(label+'_scope.npz'),positions=positions,floor=floor,baseline_scope=cnm)
            mode_results.append(dict(mode=mode,queries=len(positions),original_queries=n,extra_concepts=extras,initial_QP=initial_qp,restoration=history,restoration_feasible=feasible,line_search=lines,accepted_alpha=accepted_alpha,cases=cases))
            assign_delta(torch.zeros_like(adam));assert all(torch.equal(v.cpu(),initial[k]) for k,v in net.state_dict().items()) and np.array_equal(all_T(),baseline_T)
            K.write(out/'progress.json',dict(batch=batch_index,completed_modes=[z['mode'] for z in mode_results],new_training_updates=0))
        np.savez_compressed(out/f'batch{batch_index}_fullT.npz',**fulltables);np.savez_compressed(out/f'batch{batch_index}_finite.npz',**finite)
        results.append(dict(batch=batch_index,original_queries=n,unit=unit,base_response_loss=base_loss,Adam_norm=float(adam.norm()),modes=mode_results,all_restored=True))
        del params,allparams;allparams=H.prepare_student(net)
    guard();after=C.resources(c)
    for pin in q['models'].values():K.checked(pin)
    K.write(out/'completion.json',dict(status='COMPLETE_FULL_GALLERY_AND_QUERY_SCOPE_PROBES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(sys.argv[1]),results=results,all_T=13230,full_T_metric_names=['paired_visual_cosine','within10_minimum_margin','full_T_gallery_minimum_margin'],new_training_updates=0,temporary_Adam_steps_discarded=temporary_adam,temporary_weight_interventions_discarded=interventions,new_models=0,new_images=0,all_restored=True,seenT_diagnostic_not_generalization=True,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources=after))
except BaseException as exc:
    restored=False
    if 'net' in globals() and 'initial' in globals():
        net.load_state_dict(initial,strict=True);restored=all(torch.equal(v.cpu(),initial[k]) for k,v in net.state_dict().items())
    K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),weights_restored=restored,seconds=time.monotonic()-started));raise
