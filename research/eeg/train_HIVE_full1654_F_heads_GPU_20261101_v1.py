"""Fresh single-seed fullscope fit; no R80 tensors, calibration or epoch scan."""
import argparse,gc,hashlib,os,time,traceback
from pathlib import Path
import brainhive_full1654_F_study_contract_20261101_v1 as G

def run(path,arm):
    start=time.monotonic();s=G.setup(path);G.wrapper(s);before=G.resources(s,training_start=True);base=G.rss()
    import numpy as np,torch
    import torch.nn.functional as F
    import brainhive_exposed_fullfit_component_inputs_CPU_20261101_v1 as I
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs=I.load_verified(s['input_bank'],s['asset_protocol']);root=Path(s['output_root'])/arm;assert not root.exists();root.mkdir()
    torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    torch.manual_seed(1729);model=G.model_class(arm)().cuda();assert sum(v.numel() for v in model.parameters())==525568
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    orders=[torch.randperm(165400,generator=torch.Generator().manual_seed(G.seed('pass'+str(i)))) for i in range(3)]
    order_sha=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in orders]
    c=torch.tensor(np.asarray(inputs['condition']),device='cuda');aids=torch.tensor(inputs['affine_ids'].astype(np.int64),device='cuda')
    weight=torch.tensor(inputs['weight'],device='cuda');bias=torch.tensor(inputs['bias'],device='cuda');target=inputs['target'].to('cuda')
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    provenance=dict(study=G.ref(path),arm=arm,input_bank=s['input_bank'],orders_sha256=order_sha,source_affines=s['consumer_review'],parameters=525568,seed=1729,strength=1.,no_R80_payloads_opened=True)
    G.write(root/'fit_provenance.json',provenance);coverage=torch.zeros(3,165400,dtype=torch.bool)
    updates=seen=0;checkpoints=[];loss_total=0.;fit_started=time.monotonic();save_at={128,8192,10338,16384,20676,24576,31014}
    def budget():
        assert time.monotonic()-start<820 and G.rss()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
        assert G.allocated(s['output_root'])<12<<30 and G.ref(s['ordinary']['path'])==s['ordinary']
    def snapshot(epoch,offset):
        cp=root/'checkpoints'/f'step-{updates:06d}';cp.mkdir(parents=True)
        state=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),coverage=coverage.clone(),updates=updates,exposures=seen,completed_pass=epoch,offset=offset,study=G.ref(path),arm=arm,orders_sha256=order_sha)
        with (cp/'resume.pt').open('xb') as f:torch.save(state,f)
        commit=dict(checkpoint=G.ref(cp/'resume.pt'),updates=updates,exposures=seen,epoch=epoch,offset=offset)
        G.write(cp/'commit.json',commit);checkpoints.append(commit)
    try:
        with (root/'training.jsonl').open('x',buffering=1) as log:
            import json
            for epoch in range(3):
                for lo in range(0,165400,16):
                    indices=orders[epoch][lo:lo+16];idx=indices.cuda();aid=aids[idx];pos=idx%16540
                    optimizer.zero_grad(set_to_none=True);changed,_=model(c[idx],weight[aid],bias[aid])
                    if updates==0:assert torch.equal(changed,c[idx])
                    loss=.1*(1-F.cosine_similarity(changed.float(),target[pos].float(),dim=-1)).mean()
                    assert torch.isfinite(loss);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();assert not coverage[epoch,indices].any();coverage[epoch,indices]=True
                    n=len(indices);seen+=n;updates+=1;loss_total+=float(loss)*n
                    if updates%64==0 or updates in save_at:
                        event=dict(updates=updates,exposures=seen,pass_number=epoch+1,batch=n,last_loss=float(loss),mean_loss=loss_total/seen,gradient_norm=float(gradient),seconds=time.monotonic()-fit_started)
                        log.write(json.dumps(event)+'\n');G.write(root/'fit_progress.json',event,replace=True);budget()
                    if updates in save_at:snapshot(epoch,lo+n)
        assert updates==31014 and seen==496200 and coverage.all();torch.cuda.synchronize();fit_seconds=time.monotonic()-fit_started
        assert all(torch.isfinite(p).all() for p in model.parameters());assert any(not torch.equal(initial[k],v.cpu()) for k,v in model.state_dict().items())
        assert {int(v['step']) for v in optimizer.state.values()}=={31014}
        final=root/'component_final.pt'
        with final.open('xb') as f:torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},arm=arm,seed=1729,strength=1.,radius=.25,updates=updates,exposures=seen,passes=3,study=G.ref(path)),f)
        output=np.lib.format.open_memmap(root/'actual_train_outputs.npy',mode='w+',shape=(165400,1024),dtype=np.float32)
        with torch.inference_mode():
            for lo in range(0,165400,256):
                hi=min(lo+256,165400);aid=aids[lo:hi];output[lo:hi]=model(c[lo:hi],weight[aid],bias[aid])[0].cpu().numpy()
        output.flush();del output;budget()
        result=dict(status='COMPLETE_FULL1654_F_HEAD_THREE_PASSES',study=G.ref(path),arm=arm,updates=updates,exposures=seen,parameters=525568,component=G.ref(final),checkpoints=checkpoints,actual_train_outputs=G.ref(root/'actual_train_outputs.npy'),provenance=G.ref(root/'fit_provenance.json'),fit_seconds=fit_seconds,program_seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=G.rss()-base,resources_before=before,no_R80_payloads_opened=True,old_Dfit_V_are_train=True,official200_previously_exposed=True,independent_H=False,new_research_images=0)
        G.write(root/'fit_completion.json',result);print(__import__('json').dumps(result))
    except BaseException as error:
        G.write(root/'fit_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updates=updates,exposures=seen));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--arm',choices=G.ARMS,required=True);a=p.parse_args();run(a.study,a.arm)
