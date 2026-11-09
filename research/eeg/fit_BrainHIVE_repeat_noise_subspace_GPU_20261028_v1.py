"""D23 complete-T native first-weight learning, three matched geometry arms."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_repeat_noise_subspace_20261027_v1 as V

def loadgeo(ref):
    return {k:v.cuda() if torch.is_tensor(v) else v for k,v in torch.load(U.checked(ref),weights_only=True,map_location='cpu').items()}

def metrics(p,y):
    a=F.normalize(p.float(),dim=-1);b=F.normalize(y.float(),dim=-1);score=torch.bmm(a.reshape(-1,10,1024),b.reshape(-1,10,1024).transpose(1,2))
    return dict(MSE=float((p-y).double().square().mean()),cosine=float((a*b).sum(1).mean()),within10_top1=float((score.argmax(2)==torch.arange(10,device=p.device)).float().mean()))

def main():
    p=argparse.ArgumentParser()
    for name in ('spec','profile','native-registration','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    assert a.max_seconds==540 and not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();stop=[False];stage='INPUTS'
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    try:
        s=U.read(a.spec);base=U.read(U.checked(s['scientific_spec']));profile=U.read(a.profile);assert profile['spec']==s['scientific_spec']
        for ref in s['source_pins']:U.checked(ref)
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(C.read(a.native_registration))
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;cfg=C.read(a.native_registration)
        doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=540,maximum_output_bytes=1024<<20,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():
            assert not stop[0] and time.monotonic()-start<510;budget.check()
            assert sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())<(1024<<20)-(1<<20)
        def log(v):
            v=v|dict(seconds=time.monotonic()-start)
            with (a.output/'training_log.jsonl').open('a') as f:f.write(json.dumps(v)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps(v),flush=True)
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        from brainhive_native_models_v1 import classes
        from brainhive_bounded_component_v1 import BoundedConditionAdapter
        BrainMLP=classes()['BrainMLP'];contract=U.read(U.checked(base['native_input_contract']));sub=contract['subjects'][0];assert sub['subject']=='sub-01'
        raw=torch.from_numpy(np.array(np.load(U.checked(base['T_mean4']),mmap_mode='r'),copy=True).reshape(13230,4250)).cuda();target=torch.from_numpy(np.load(U.checked(base['targets']),allow_pickle=False)).cuda()
        prepared=U.read(U.checked(s['prepared_EEG']));mapping=U.read(U.checked(prepared['mapping']));d09=U.read(U.checked(s['D09_completion']));panel=U.read(U.checked(U.read(U.checked(d09['inputs']))['panel']))
        dev=torch.from_numpy(np.array(np.load(U.checked(prepared['Dfit']),mmap_mode='r')[0],copy=True).reshape(820,4250)).cuda();dc=torch.from_numpy(np.array(np.load(U.checked(prepared['Dcheck']),mmap_mode='r'),copy=True).reshape(400,4250)).cuda()
        assert mapping['Dcheck_queries']==panel['Dcheck'];queries=mapping['Dfit_queries'][:820];source=next(x for x in sub['sources'] if x['role']=='full');dfrows=[x['array_row'] for x in source['groups']['calibration_only_Dfit']];assert dfrows==[x['array_row'] for x in queries]
        fbank=torch.load(U.checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True);truth=fbank[dfrows].cuda()
        assert torch.equal(fbank[U.read(Path(__file__).parent/'configs/split_manifest.json')['image_rows']['T']],target.cpu());del fbank
        oldcomponent=torch.load(U.checked(s['ordinary_checkpoint']),weights_only=False,map_location='cpu');assert oldcomponent['arm']=='ordinary' and oldcomponent['calibration']['selected_strength']==1.
        ordinary=BoundedConditionAdapter().cuda().eval().requires_grad_(False);ordinary.load_state_dict(oldcomponent['model'],strict=True)
        oof=np.full((13230,3,1024),np.nan,np.float32);oofnative=np.full_like(oof,np.nan);coverage=np.zeros((13230,3),np.int8);Dfit=np.full((820,3,1024),np.nan,np.float32);Dcheck=np.full((400,3,1024),np.nan,np.float32)
        diagnostic_kernel=torch.from_numpy(np.load(U.checked(s['old_shared_kernel_basis']),allow_pickle=False)).cuda()
        states={};qrefs={};proofs=[];dfmetrics={};fit_seconds=0.
        for ri,(role,item) in enumerate(base['roles'].items()):
            check();stage='COMPLETE_T_FITS_'+role;old=torch.load(U.checked(item['checkpoint']),weights_only=True,map_location='cpu');model=BrainMLP(4250,1024,num_layers=1,dropout=.1).cuda().requires_grad_(False);model.load_state_dict(old['model'],strict=True)
            gtrue=loadgeo(profile['geometries'][role]['repeat']);gperm=loadgeo(profile['geometries'][role]['permuted']);geos={V.ARMS[0]:gtrue,V.ARMS[1]:gperm,V.ARMS[2]:gtrue}
            white={key:V.whiten(raw,value) for key,value in [('repeat',gtrue),('permuted',gperm)]};qs={arm:g['Q0'].clone().requires_grad_(True) for arm,g in geos.items()};ix=np.asarray(item['fit_positions']);ixgpu=torch.as_tensor(ix,device='cuda');gen=torch.Generator().manual_seed(1729+ri)
            scale=old['logit_scale'].exp().clamp_max(100).cuda();tic=time.monotonic()
            for epoch in range(10):
                order=torch.randperm(len(ix),generator=gen);cover=np.zeros(len(ix),np.int8);losses={arm:0. for arm in V.ARMS};seen=0
                for bi,lo in enumerate(range(0,len(ix),1024)):
                    check();chosen=order[lo:lo+1024];cover[chosen.numpy()]+=1;ids=ixgpu[chosen.cuda()];labels=torch.arange(len(ids),device='cuda')
                    for arm in V.ARMS:
                        q=qs[arm];q.grad=None;g=geos[arm];key='permuted' if arm==V.ARMS[1] else 'repeat';model.train();torch.cuda.manual_seed(1729+ri*100000+epoch*1000+bi)
                        pred=model.block(V.residual_preactivation(raw[ids],white[key][ids],q,g,model.proj_in.bias));logits=scale*F.normalize(pred,dim=1)@F.normalize(target[ids],dim=1).T
                        loss=.5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels));assert torch.isfinite(loss);loss.backward()
                        with torch.no_grad():updated,step=V.fixed_step(q,q.grad,arm!=V.ARMS[2]);q.copy_(updated)
                        assert step['linearized_risk_change']<=1e-8;losses[arm]+=float(loss.detach())*len(ids)
                    seen+=len(ids)
                assert np.all(cover==1) and seen==len(ix)
                log(dict(stage='COMPLETE_ROLE_T_PASS',role=role,epoch=epoch+1,updates_per_model=(epoch+1)*item['updates_per_epoch'],exposures_per_model=(epoch+1)*len(ix),coverage_min=int(cover.min()),coverage_max=int(cover.max()),order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),mean_InfoNCE={arm:losses[arm]/seen for arm in V.ARMS},Dfit_used=False))
            fit_seconds+=time.monotonic()-tic;model.eval();states[role]={}
            held=item['held_positions'];w=model.block.layers[0].norm.weight;b=model.block.layers[0].norm.bias
            def infer(values,apply_ordinary):
                pieces=[]
                with torch.no_grad():
                    for e in values.split(256):
                        c=model(e)
                        if apply_ordinary:c=ordinary(c,w.expand(len(c),-1),b.expand(len(c),-1))[0]
                        pieces.append(c.cpu())
                return torch.cat(pieces)
            for ai,arm in enumerate(V.ARMS):
                check();q=qs[arm].detach();g=geos[arm];fold=V.folded_weight(q,g);error=V.metric_error(fold,g)
                if arm!=V.ARMS[2]:assert error<2e-5
                model.proj_in.weight.copy_(fold);state={key:val.detach().cpu().clone() for key,val in model.state_dict().items()}
                for key,value in old['model'].items():
                    if key!='proj_in.weight':assert torch.equal(state[key],value),key
                assert sum(v.numel() for v in state.values())==6452224;states[role][arm]=state
                qp=a.output/(role+'_'+arm+'_Q.pt');torch.save(q.detach().cpu(),qp);qrefs[role+'_'+arm]=U.ref(qp)
                with torch.no_grad():
                    sample=raw[ix[:64]];key='permuted' if ai==1 else 'repeat';expanded=model.block(V.residual_preactivation(sample,white[key][ix[:64]],q,g,model.proj_in.bias));folded=model(sample);rel=float(((folded-expanded).norm(dim=1)/expanded.norm(dim=1)).max());assert rel<3e-5
                old_kernel_response=float((g['W0'].double()@diagnostic_kernel).norm()/g['W0'].double().norm());new_kernel_response=float((fold.double()@diagnostic_kernel).norm()/fold.double().norm())
                proof=dict(role=role,arm=arm,updates=item['updates_per_epoch']*10,exposures=len(ix)*10,used_noise_metric_relative_error=error,true_regularized_repeat_noise_metric_error=V.metric_error(fold,gtrue),folded_condition_relative_max=rel,
                    old_shared_kernel_response_relative=old_kernel_response,new_shared_kernel_response_relative=new_kernel_response,old_kernel_used_for_postfit_probe_only=True,all_other_native_weights_bitwise_unchanged=True);proofs.append(proof)
                if role=='full':
                    pn=infer(dev,False);pc=infer(dev,True);Dfit[:,ai]=pc.numpy();Dcheck[:,ai]=infer(dc,True).numpy();dfmetrics[arm]=dict(native=metrics(pn.cuda(),truth),ordinary=metrics(pc.cuda(),truth));log(dict(stage='FINAL_DFIT820_LOG_ONLY',arm=arm,values=dfmetrics[arm]))
                else:
                    values=raw[held];oofnative[held,ai]=infer(values,False).numpy();oof[held,ai]=infer(values,True).numpy();coverage[held,ai]+=1
            del model,gtrue,gperm,geos,white,qs,old;torch.cuda.empty_cache();check()
        assert np.all(coverage==1) and np.isfinite(oof).all() and np.isfinite(Dfit).all() and np.isfinite(Dcheck).all()
        torch.save(dict(models=states,scientific_spec=s['scientific_spec'],trained_parameter='proj_in.weight',all_other_native_weights_frozen=True),a.output/'native_models.pt');modelref=U.ref(a.output/'native_models.pt')
        for name,value in [('OOF_T_native_conditions.npy',oofnative),('OOF_T_ordinary_conditions.npy',oof),('Dfit_ordinary_conditions.npy',Dfit),('Dcheck400_conditions.npy',Dcheck)]:U.save_array(a.output/name,value)
        U.atomic(a.output/'Dfit_final_metrics.json',dfmetrics);U.atomic(a.output/'fit_proofs.json',dict(proofs=proofs,all810_updates_complete=sum(x['updates'] for x in proofs)==810,all793800_exposures_complete=sum(x['exposures'] for x in proofs)==793800))
        lock=dict(status='D23_ALL400_CORRECT_EEG_CONDITIONS_LOCKED_BEFORE_IMAGES',query_ids=[x['query_id'] for x in panel['Dcheck']],arms=V.ARMS,conditions=U.ref(a.output/'Dcheck400_conditions.npy'),model=modelref,
            normal_inference_other_subject_EEG=False,Dcheck_target_F_or_images_used=False,Dcheck_scores_used_for_choices=False,ordinary_checkpoint=s['ordinary_checkpoint'],ordinary_strength=1.,source=U.ref(__file__))
        U.atomic(a.output/'condition_lock.json',lock);check()
        done=dict(status='D23_ALL9_NATIVE_FIRST_LAYER_FULL_T10_AND_DFIT820_COMPLETE',spec=U.ref(a.spec),scientific_spec=s['scientific_spec'],source=U.ref(__file__),wrapper=wrapper,profile=U.ref(a.profile),model=modelref,Q_states=qrefs,
            condition_lock=U.ref(a.output/'condition_lock.json'),OOF_T_native=U.ref(a.output/'OOF_T_native_conditions.npy'),OOF_T_ordinary=U.ref(a.output/'OOF_T_ordinary_conditions.npy'),Dfit_conditions=U.ref(a.output/'Dfit_ordinary_conditions.npy'),Dfit_metrics=U.ref(a.output/'Dfit_final_metrics.json'),fit_proofs=U.ref(a.output/'fit_proofs.json'),
            parameters_per_encoder=4352000,total_native_parameters_per_encoder=6452224,additional_inference_parameters=0,all_complete_T_per_role=True,models=9,updates=810,query_exposures=793800,
            no_Dfit_calibration_or_epoch_or_arm_selection=True,normal_inference_individual_trials=False,new_UNet_calls=0,new_images=0,formal_D_resumed=False,fit_body_seconds=fit_seconds,seconds=time.monotonic()-start,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes)
        U.atomic(a.output/'completion.json',done);print(json.dumps({k:v for k,v in done.items() if k!='Q_states'}))
    except BaseException as exc:
        partial=None
        if 'qs' in locals():
            try:
                path=a.output/'failure_partial_Q_and_RNG.pt';torch.save(dict(Q={k:v.detach().cpu() for k,v in qs.items()},role=role,epoch=epoch,CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state(),scientific_spec=s['scientific_spec']),path);partial=U.ref(path)
            except BaseException as save_error:partial=dict(error=repr(save_error))
        U.atomic(a.output/'failure.json',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc(),partial_state=partial,seconds=time.monotonic()-start,no_automatic_retry=True));raise

if __name__=='__main__':main()
