"""Native same-Q paired-effect integration; no scheduler or fit authorization.

All four modes build the same response graphs. O/A/D/U select zero/absolute/
paired-difference/shuffled-pair objectives. This file does not promote a DRAFT.
"""
import hashlib
import math
import torch
import torch.nn.functional as F
from brainhive_response_rank_v1 import NativeIPResponseCapture, ip_response

ARMS=('O','A','D','U')
EPSILON=1e-6


def stable_key(*parts):
    return hashlib.sha256('|'.join(map(str,(1729,)+parts)).encode()).hexdigest()


def build_pair_plan(contract, protocol, metadata):
    """A bijective next-concept pairing and commuting teacher-pair shuffle.

    Three-concept rotation shuffles the WHOLE F-feature pair, preserving its
    empirical joint distribution before Q-dependent attention. The response
    distribution under Q need not be identical. No scores are used.
    Returned global query indices follow the unchanged load_training order.
    """
    allowed=protocol['image_rows']['T'];n=len(allowed)
    assert n==13230 and len(set(allowed))==n
    assert {x['subject'] for x in contract['subjects']}=={'sub-%02d'%i for i in range(1,11)}
    assert set(allowed).isdisjoint(protocol['image_rows']['D']+protocol['image_rows']['V'])
    positions={value:i for i,value in enumerate(allowed)}
    partner=[-1]*(10*n);shuffled=[-1]*(10*n);affine=[-1]*(10*n);groups=[]
    aid=0
    for si,subject in enumerate(contract['subjects']):
        sources=[s for s in subject['sources'] if s['role'].startswith('oof')]
        assert {s['role'] for s in sources}=={'oof0','oof1'} and len(sources)==2
        for source in sources:
            ids=[int(r['array_row']) for r in source['groups']['component_train']]
            by_concept={}
            for value in ids:by_concept.setdefault(metadata[value]['concept_id'],[]).append(value)
            assert len(ids)==len(set(ids)) and all(len(v)==10 for v in by_concept.values())
            concepts=sorted(by_concept,key=lambda c:stable_key('paired_concept_order',source['role'],c))
            assert len(concepts)>5
            order=[i for c in concepts for i in sorted(by_concept[c],key=lambda v:stable_key('paired_stimulus_order',v))]
            for rank,value in enumerate(order):
                i=si*n+positions[value];j=si*n+positions[order[(rank+10)%len(order)]]
                p=si*n+positions[order[(rank+30)%len(order)]]
                assert affine[i]==-1
                partner[i]=j;shuffled[i]=p;affine[i]=aid
            groups.append(dict(subject=subject['subject'],role=source['role'],affine_id=aid,
                               queries=len(order),concepts=len(concepts)))
            aid+=1
    assert aid==20 and all(a>=0 for a in affine)
    assert sorted(partner)==sorted(shuffled)==list(range(10*n))
    shuffled_partner=[shuffled[j] for j in partner]
    assert shuffled_partner==[partner[p] for p in shuffled], 'Whole-pair shuffle must commute'
    for i,j in enumerate(partner):
        p,q=shuffled[i],shuffled_partner[i]
        assert i//n==j//n==p//n==q//n and affine[i]==affine[j]==affine[p]==affine[q]
        assert len({metadata[allowed[k%n]]['concept_id'] for k in (i,j,p,q)})==4
    return dict(partner=partner,shuffled=shuffled,shuffled_partner=shuffled_partner,
                expected_affine_ids=affine,groups=groups,queries=10*n,
                teacher_pair_joint_distribution_preserved=True,source_roles_preserved=True)


def response_objectives(si,sj,ti,tj,ui,uj):
    assert si.ndim==4 and si.shape==sj.shape==ti.shape==tj.shape==ui.shape==uj.shape
    assert all(torch.isfinite(t).all() for t in (si,sj,ti,tj,ui,uj))
    si,sj=si.float(),sj.float()
    ti,tj,ui,uj=[v.detach().float() for v in (ti,tj,ui,uj)]
    axes=(1,2,3)
    separation=(ti-tj).square().mean(axes)
    shuffled_separation=(ui-uj).square().mean(axes)
    valid=(separation>EPSILON)&(shuffled_separation>EPSILON)
    def normalized(error):
        # ALL objectives use one correct-pair scale and one common mask.
        # U changes only teacher assignment, never scale or effective queries.
        scale=torch.where(valid,separation+EPSILON,torch.ones_like(separation))
        values=error.square().mean(axes)/scale
        masked=torch.where(valid,values,torch.zeros_like(values))
        return masked.mean(),valid
    abs_i,valid=normalized(si-ti)
    abs_j,_=normalized(sj-tj)
    absolute=.5*(abs_i+abs_j)
    difference,_=normalized((si-sj)-(ti-tj))
    common,_=normalized(.5*((si-ti)+(sj-tj)))
    shuffled,valid_u=normalized((si-sj)-(ui-uj))
    assert all(torch.isfinite(v) for v in (absolute,difference,shuffled))
    # Exact squared-error decomposition: A=common + .25*raw_difference.
    # Removing common leaves the SAME coefficient on the contrast component.
    difference=.25*difference;shuffled=.25*shuffled
    assert torch.allclose(absolute,common+difference,atol=2e-5,rtol=2e-5)
    return dict(A=absolute,D=difference,U=shuffled),dict(
        valid_fraction=valid.float().mean().detach(),shuffled_valid_fraction=valid_u.float().mean().detach(),
        teacher_separation=(ti-tj).square().mean().detach(),
        shuffled_teacher_separation=shuffled_separation.mean().detach(),
        common_component=common.detach(),contrast_component=difference.detach(),
        student_difference_energy=(si-sj).square().mean().detach())


class PairedEffectCapture(NativeIPResponseCapture):
    def auxiliary(self,student_i,student_j,teacher_i,teacher_j,shuffled_i,shuffled_j,*,arm):
        assert arm in ARMS and not self.active and len(self.queries)==len(self.layers)>0
        assert student_i.requires_grad and student_j.requires_grad
        values={k:[] for k in ('A','D','U')};diagnostics={}
        try:
            for name,(attention,processor) in self.layers.items():
                q=self.queries[name]
                assert not q.requires_grad and all(not p.requires_grad for p in processor.parameters())
                si=ip_response(q,student_i,processor,attention.heads)
                sj=ip_response(q,student_j,processor,attention.heads)
                with torch.no_grad():
                    ti,tj,ui,uj=[ip_response(q,t,processor,attention.heads)
                                 for t in (teacher_i,teacher_j,shuffled_i,shuffled_j)]
                losses,info=response_objectives(si,sj,ti,tj,ui,uj)
                for k in values:values[k].append(losses[k])
                for k,v in info.items():diagnostics.setdefault(k,[]).append(v)
        finally:self.queries.clear()
        means={k:torch.stack(v).mean() for k,v in values.items()}
        # Explicit zero-weight graphs preserve both student branches in every
        # mode, including O, and prevent a cheaper computational baseline.
        selected=sum(v*(1. if k==arm else 0.) for k,v in means.items())
        return selected,means,{k:torch.stack(v).mean() for k,v in diagnostics.items()}


def paired_step(model,pipe,adapter,projector,prompts,scheduler,collector,inputs,latents,plan,indices,rng,arm):
    from brainhive_component_generator_v1 import denoise
    assert arm in ARMS and not collector.active and not collector.queries
    device=inputs['condition'].device
    i=indices.to(device);j=plan['partner'][i];p=plan['shuffled'][i];q=plan['shuffled_partner'][i]
    assert torch.equal(inputs['affine_ids'][i],inputs['affine_ids'][j])
    assert torch.equal(inputs['affine_ids'][i],inputs['affine_ids'][p])
    assert torch.equal(inputs['affine_ids'][i],inputs['affine_ids'][q])
    assert torch.equal(inputs['affine_ids'][i],plan['expected_affine_ids'][i])
    ci,cj=inputs['condition'][i],inputs['condition'][j]
    wi,bi=inputs['weight'][inputs['affine_ids'][i]],inputs['bias'][inputs['affine_ids'][i]]
    wj,bj=inputs['weight'][inputs['affine_ids'][j]],inputs['bias'][inputs['affine_ids'][j]]
    # Preserve actual FP32 host-affine/head arithmetic outside autocast.
    changed_i,info=model(ci,wi,bi,gate_mode='fixed')
    changed_j,_=model(cj,wj,bj,gate_mode='fixed')
    assert changed_i.dtype==changed_j.dtype==torch.float32 and not model.predict_risk and model.radius==.25
    latent=latents[i%13230]
    noise=torch.randn(latent.shape,device=device,generator=rng)
    timesteps=torch.randint(0,1000,(len(i),),device=device,generator=rng)
    noisy=scheduler.add_noise(latent,noise,timesteps)
    forwards=[0]
    def count(module,args,result):forwards[0]+=1
    handle=pipe.unet.register_forward_hook(count)
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            with torch.no_grad():
                base=projector(adapter(ci))
                ti,tj,ui,uj=[projector(adapter(inputs['target'][x%13230])) for x in (i,j,p,q)]
                with collector.capture():denoise(pipe,prompts,noisy,timesteps,base)
            frozen_q={k:(v,v.data_ptr(),v._version) for k,v in collector.queries.items()}
            si,sj=projector(adapter(changed_i)),projector(adapter(changed_j))
            prediction=denoise(pipe,prompts,noisy,timesteps,si)
            assert forwards[0]==2
            assert all(collector.queries[k] is v and v.data_ptr()==ptr and v._version==version
                       for k,(v,ptr,version) in frozen_q.items())
            auxiliary,terms,diagnostics=collector.auxiliary(si,sj,ti,tj,ui,uj,arm=arm)
            epsilon_loss=F.mse_loss(prediction.float(),noise)
            token_loss=(1-F.cosine_similarity(si.float(),ti.float(),dim=-1)).mean()
            loss=epsilon_loss+.1*token_loss+.1*auxiliary
        assert not collector.queries and torch.isfinite(loss)
        return dict(loss=loss,epsilon_loss=epsilon_loss,token_loss=token_loss,auxiliary=auxiliary,
                    terms=terms,diagnostics=diagnostics,changed_i=changed_i,changed_j=changed_j,
                    information=info,explicit_UNet_forwards=forwards[0],same_parent_Q_verified=True)
    finally:
        handle.remove();collector.queries.clear()
