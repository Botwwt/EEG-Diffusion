"""Actual-parameter response repair with fixed CN margins and persistent Adam.

Training-only visual targets and NN responses never enter deployment. The QP
is a local subproblem; finite feasibility is checked on the sampled query set.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
import atm_response_contrast_repair_core_20261105_v195 as H
from atm_margin_repair_schedule_20261106_v202 import epoch_batches

def flat(xs):return torch.cat([x.reshape(-1) for x in xs])

def cpu_tree(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return copy.deepcopy(x)

def assert_tree_equal(a,b):
    if torch.is_tensor(a):assert torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:assert_tree_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):assert_tree_equal(x,y)
    else:assert a==b,(a,b)

class Trainer:
    def __init__(self,net,anchor,initial,xall,vall,T,Qbank,layouts,weights,kv,guard):
        self.net,self.anchor,self.initial=net,anchor,initial
        self.xall,self.vall,self.T=xall,vall,np.asarray(T)
        self.visual=F.normalize(vall[self.T].cuda(),dim=1)
        self.bank,self.layouts,self.weights,self.kv=Qbank,layouts,weights,kv
        self.guard=guard;self.attempts=self.accepted=self.exposures=0
        rng=np.random.default_rng(1729);self.units=np.concatenate([rng.permutation(280) for _ in range(3)])[:830]
        self.allparams=H.prepare_student(net)
        context=self.context(0);loss=self.response_loss(context)
        gs=torch.autograd.grad(loss,self.allparams,allow_unused=True)
        self.params=[p for p,g in zip(self.allparams,gs) if g is not None]
        active={id(p) for p in self.params};self.names=[n for n,p in net.named_parameters() if id(p) in active]
        assert len(self.names)==31 and sum(p.numel() for p in self.params)==545930
        assert [id(p) for p in self.params]==[id(dict(net.named_parameters())[n]) for n in self.names]
        self.opt=torch.optim.AdamW(self.params,lr=1e-4,weight_decay=.01)
        self.baseline=self.all_T()
        self.last_event=None

    def context(self,index):
        epoch,bi=divmod(index,83);order=np.random.default_rng(1739+epoch).permutation(1323)
        b=list(epoch_batches(order.tolist()))[bi];positions=np.asarray(b['response_positions'],np.int64)
        scope=np.asarray(b['constraint_positions'],np.int64);rows=self.T[positions]
        x=self.xall[rows].cuda();sid=torch.ones(len(rows),device='cuda',dtype=torch.long)
        unit=int(self.units[index]);ti,li=divmod(unit,70);bank=self.bank[f'l{li:02d}'][:,ti]
        heads,dim=bank.shape[1],bank.shape[-1]
        Q=torch.tensor(bank.transpose(1,0,2,3).reshape(1,heads,8,dim).copy(),device='cuda').float()
        weights={**self.weights,**self.kv[li]}
        with torch.no_grad():
            old=H.response(self.anchor(x,sid),Q,weights)
            target_response=H.response(self.vall[rows].cuda(),Q,weights)
            scale=(target_response-target_response.mean(0)).square().mean()
        assert float(scale)>1e-12
        return dict(epoch=epoch,batch=bi,positions=positions,scope=scope,x=x,sid=sid,Q=Q,weights=weights,old=old,scale=scale,unit=unit)

    def response_loss(self,c):return (H.response(self.net(c['x'],c['sid']),c['Q'],c['weights'])-c['old']).square().mean()/c['scale']

    def predict(self,positions):
        return torch.cat([self.net(self.xall[self.T[positions[j:j+160]]].cuda(),torch.ones(len(positions[j:j+160]),device='cuda',dtype=torch.long)) for j in range(0,len(positions),160)])

    def margins(self,positions):
        yy=F.normalize(self.predict(positions),dim=1);score=yy@self.visual.T
        ii=torch.arange(len(positions),device='cuda');ix=torch.tensor(positions,device='cuda');own=score[ii,ix]
        neg=score.clone();neg[ii,ix]=-torch.inf
        same=ix[:,None]//10==torch.arange(13230,device='cuda')[None,:]//10
        wi=neg.masked_fill(~same,-torch.inf).argmax(1);gi=neg.argmax(1)
        return torch.cat([own-score[ii,wi],own-score[ii,gi]])

    @torch.no_grad()
    def all_T(self):
        self.guard();pred=[]
        for lo in range(0,13230,160):pred.append(F.normalize(self.predict(np.arange(lo,min(lo+160,13230))),dim=1))
        pred=torch.cat(pred);ans=[]
        for lo in range(0,13230,160):
            hi=min(lo+160,13230);ix=torch.arange(lo,hi,device='cuda');ii=torch.arange(hi-lo,device='cuda')
            score=pred[lo:hi]@self.visual.T;own=score[ii,ix].clone();score[ii,ix]=-torch.inf
            same=ix[:,None]//10==torch.arange(13230,device='cuda')[None,:]//10
            ans.append(torch.stack([own,own-score.masked_fill(~same,-torch.inf).max(1).values,own-score.max(1).values],1).cpu().numpy())
        return np.concatenate(ans)

    def assign(self,theta,delta):
        vec=theta+delta.float();offset=0
        with torch.no_grad():
            for p in self.params:p.copy_(vec[offset:offset+p.numel()].reshape(p.shape));offset+=p.numel()
        assert offset==len(vec)

    @torch.no_grad()
    def finite(self,c,floor,base_loss):
        self.guard();m=self.margins(c['scope']).double();d=m-floor
        loss=float(self.response_loss(c))
        return dict(loss=loss,response_reduction=(base_loss-loss)/max(base_loss,1e-30),minimum_change=float(d.min()),decreased=int((d<-1e-6).sum()),merit=float(torch.minimum(d,torch.zeros_like(d)).square().sum()))

    def solve(self,c,floor,proposal,adam_norm):
        self.guard();m=self.margins(c['scope']);count=len(m);J=torch.empty((count,len(proposal)),device='cuda',dtype=torch.float64)
        for i in range(count):
            self.guard();J[i]=flat(torch.autograd.grad(m[i],self.params,retain_graph=True)).detach().double()
        rn=J.norm(dim=1);assert bool((rn>1e-12).all());J/=rn[:,None]
        rhs=(floor-m.detach().double())/rn
        A=(J@J.T).cpu().numpy();b=(J@proposal-rhs).cpu().numpy()
        sol=minimize(lambda v:(.5*v@(A@v)+b@v,A@v+b),np.zeros(count),jac=True,bounds=[(0,None)]*count,method='L-BFGS-B',callback=lambda _:self.guard(),options={'maxiter':10000,'ftol':0.,'gtol':1e-13,'maxls':100})
        lam=sol.x;delta=proposal+J.T@torch.tensor(lam,device='cuda');slack=(J@delta-rhs).detach().cpu().numpy();dual=A@lam+b
        err=float(abs(slack-dual).max());violation=float(max(0,-slack.min()));comp=float(abs(lam*dual).max())
        valid=bool(err<1e-7*adam_norm and violation<1e-7*adam_norm and comp<1e-7*adam_norm*adam_norm)
        info=dict(constraints=count,iterations=int(sol.nit),solver_success=bool(sol.success),KKT_valid=valid,violation=violation,complementarity=comp,primal_dual_error=err,proposal_norm=float(proposal.norm()),step_norm=float(delta.norm()))
        return delta.detach(),info

    def snapshot(self):
        state=self.net.state_dict();active=set(self.names)
        for n,v in state.items():
            if n not in active:assert torch.equal(v.cpu(),self.initial[n]),n
        return dict(trainable={n:state[n].detach().cpu().clone() for n in self.names},names=self.names,optimizer=cpu_tree(self.opt.state_dict()),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state(),attempts=self.attempts,accepted=self.accepted,exposures=self.exposures,last_event=self.last_event)

    def restore(self,state):
        assert state['names']==self.names and set(state['trainable'])==set(self.names)
        self.net.load_state_dict({**self.initial,**state['trainable']},strict=True);self.opt.load_state_dict(cpu_tree(state['optimizer']))
        self.attempts,self.accepted,self.exposures=[state[k] for k in ['attempts','accepted','exposures']];self.last_event=state['last_event']
        torch.set_rng_state(state['CPU_rng']);torch.cuda.set_rng_state(state['CUDA_rng'])

    def attempt(self,arm):
        assert arm in ('margin_constrained','response_only') and self.attempts<830
        self.guard();c=self.context(self.attempts);theta=flat(self.params).detach().clone()
        prior_optimizer=cpu_tree(self.opt.state_dict());self.opt.zero_grad(set_to_none=True)
        loss=self.response_loss(c);base_loss=float(loss.detach());loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(self.params,1.));assert np.isfinite(norm) and np.isfinite(base_loss)
        self.opt.step();adam=(flat(self.params).detach()-theta).double();an=float(adam.norm())
        assert torch.isfinite(adam).all();self.assign(theta,torch.zeros_like(adam))
        scope=c['scope'];floor=torch.tensor(np.concatenate([self.baseline[scope,1],self.baseline[scope,2]]),device='cuda',dtype=torch.float64)
        before=self.finite(c,floor,base_loss);qps=[];restoration=[];search=[];alpha=None
        if arm=='response_only':direction=adam;alpha=1.
        elif an>1e-20:
            direction,qi=self.solve(c,floor,adam,an);qps.append(qi)
            if qi['KKT_valid']:
                for iteration in range(6):
                    self.assign(theta,direction);event=self.finite(c,floor,base_loss)
                    if event['minimum_change']>=-1e-6:break
                    if iteration==5:break
                    current=(flat(self.params).detach()-theta).double()
                    correction,qi=self.solve(c,floor,torch.zeros_like(adam),an);qps.append(qi)
                    if not qi['KKT_valid']:break
                    accepted=False;trials=[]
                    for power in range(11):
                        beta=2.**(-power);candidate=current+beta*correction;self.assign(theta,candidate)
                        check=self.finite(c,floor,base_loss);passed=check['minimum_change']>=-1e-6 or check['merit']<=event['merit']*(1-1e-4*beta)
                        trials.append(dict(beta=beta,passed=passed,**check))
                        if passed:direction=candidate;accepted=True;break
                    restoration.append(dict(before=event,trials=trials,accepted=accepted))
                    if not accepted:break
                for power in range(11):
                    candidate_alpha=2.**(-power);self.assign(theta,direction*candidate_alpha);check=self.finite(c,floor,base_loss)
                    passed=check['minimum_change']>=-1e-6 and check['response_reduction']>1e-7
                    search.append(dict(alpha=candidate_alpha,passed=passed,**check))
                    if passed:alpha=candidate_alpha;break
        if alpha is None:
            self.assign(theta,torch.zeros_like(adam));self.opt.load_state_dict(prior_optimizer)
            assert_tree_equal(self.opt.state_dict(),prior_optimizer)
        else:self.assign(theta,direction*alpha);self.accepted+=1
        after=self.finite(c,floor,base_loss)
        if arm=='margin_constrained' and alpha is not None:assert after['minimum_change']>=-1e-6 and after['response_reduction']>1e-7
        self.attempts+=1;self.exposures+=len(c['positions'])
        event=dict(attempt=self.attempts,accepted_total=self.accepted,accepted=alpha is not None,alpha=alpha,epoch=c['epoch']+1,batch=c['batch']+1,exposures=self.exposures,unit=c['unit'],queries=len(c['positions']),constraint_queries=len(scope),response_loss_before=base_loss,preclip_norm=norm,Adam_norm=an,actual_parameter_step_norm=float((flat(self.params).detach()-theta).double().norm()),before=before,after=after,QP=qps,restoration=restoration,line_search=search)
        self.last_event=event;return event
