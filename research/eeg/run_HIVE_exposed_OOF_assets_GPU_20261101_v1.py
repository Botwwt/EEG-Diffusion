"""Real native train-only fold and held export. Profile weights are discarded."""
import argparse,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
import brainhive_exposed_OOF_assets_contract_CPU_20261101_v1 as G

def main(path,mode,task_id):
    context=G.preflight(path,mode,task_id);assert context['token'] is G.TOKEN
    s=context['s'];task=context['task'];root=Path(s['output_root'])
    folder=root/'engineering' if mode=='profile' else root/'tasks'/task_id
    folder.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();baseline_rss=int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmHWM:')))*1024
    try:
        import numpy as np,torch
        import brainhive_native_models_v1 as N
        import brainhive_official_encoder_training_v1 as T
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(min(1.,s['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory))
        T.seed_everything(torch);priors=N.prior_models('cuda');teacher=priors['fusion_encoder'].vision_model
        G.assert_input_aliases(s)
        assert sum(p.numel() for p in teacher.parameters())==11015168
        T.frozen_stochastic_teacher(torch,teacher);versions=[p._version for p in teacher.parameters()]
        del priors;gc.collect();torch.cuda.empty_cache()
        raw=torch.load(G.checked(s['subjects'][task['subject']]),weights_only=True,map_location='cpu',mmap=True)
        visual=torch.load(G.checked(s['visual_embeddings']),weights_only=True,map_location='cpu',mmap=True)
        assert raw.shape==(16540,63,250) and raw.dtype==torch.float32
        expected={'CLIP-ViT-B-32-laion2B-s34B-b79K':512,'CLIP-ViT-H-14-laion2B-s32B-b79K':1024,'vae':1024}
        assert set(visual)==set(expected) and all(v.shape==(16540,expected[k]) and v.dtype==torch.float32 for k,v in visual.items())
        binding=G.read(G.checked(s['bindings']));channels=binding['selected_indices'];assert len(channels)==len(set(channels))==17
        fit=torch.tensor(task['training_array_rows'],dtype=torch.long);held=torch.tensor(task['exported_OOF_array_rows'],dtype=torch.long)
        eeg=raw[fit[:,None],torch.tensor(channels)[None,:],:].contiguous().cuda()
        visuals={k:v.index_select(0,fit).contiguous().cuda() for k,v in visual.items()}
        assert torch.isfinite(eeg).all() and all(torch.isfinite(v).all() for v in visuals.values())
        del visual;gc.collect()
        T.seed_everything(torch);model=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda()
        assert sum(p.numel() for p in model.parameters())==6452224
        scale=torch.nn.Parameter(torch.tensor(2.6592,device='cuda'));optimizer,scheduler=T.make_optimizer(torch,model,scale,225)
        shuffle=torch.Generator(device='cpu').manual_seed(1730);updates=0;rows=[];committed=[];producer=G.ref(__file__)['sha256']
        registration=dict(task=task,protocol=context['protocol'],scope='Full1654 official-train-only fold; exposed benchmark acknowledged; no component or benchmark admission')
        G.atomic(folder/'registration.json',registration);reg=G.ref(folder/'registration.json')['sha256']
        count=1 if mode=='profile' else 25
        def budget(_stage=None):
            assert time.monotonic()-began<820
            assert torch.cuda.max_memory_allocated()<=s['maximum_GPU_bytes']
            extra=int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmHWM:')))*1024-baseline_rss
            assert extra<=s['maximum_extra_CPU_bytes'];G.resources(s)
        for epoch in range(1,count+1):
            row,updates=T.run_epoch(torch,model,teacher,scale,optimizer,scheduler,shuffle,eeg,visuals,epoch=epoch,updates=updates,total_updates=225,batch_size=1024,subject=task['subject'],role='train_only_official_oof'+str(task['held_fold']),device=torch.device('cuda'),deadline=began+820,autocast_bfloat16=True,observer=budget)
            rows.append(row)
            if mode=='fit':
                state=T.snapshot(torch,model,scale,optimizer,scheduler,shuffle,epoch,updates,reg,context['protocol']['sha256'],producer,torch.device('cuda'))
                T.publish_epoch(torch,folder,state,row);del state
            G.atomic(folder/'progress.json',dict(epoch=epoch,updates=updates,total_updates=225,UTC=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()),replace=True)
            budget()
        assert versions==[p._version for p in teacher.parameters()] and all(p.grad is None for p in teacher.parameters())
        if mode=='fit':
            committed,pending=T.scan_committed_epochs(folder,reg,context['protocol']['sha256'],producer_source_sha256=producer,subject=task['subject'],role='train_only_official_oof'+str(task['held_fold']),updates_per_epoch=9,total_epochs=25,full_epoch_rows=8270)
            assert not pending and updates==225
            final=T.publish_final(torch,folder,model,scale,registration_sha256=reg,registration_core_sha256=context['protocol']['sha256'],producer_source_sha256=producer,committed=committed,total_epochs=25,total_updates=225,subject=task['subject'],role='train_only_official_oof'+str(task['held_fold']))
        # No held-row visual features are evaluated or scored. Only EEG is exported.
        model.eval();held_eeg=raw[held[:,None],torch.tensor(channels)[None,:],:].contiguous();del raw,eeg,visuals;gc.collect()
        output=[]
        with torch.no_grad():
            for batch in held_eeg.split(512):output.append(model(batch.cuda().reshape(len(batch),4250)).cpu())
        observed=torch.cat(output);assert observed.shape==(8270,1024) and torch.isfinite(observed).all()
        cpu=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval();cpu.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()},strict=True)
        expected_output=[]
        with torch.no_grad():
            for batch in held_eeg.split(256):expected_output.append(cpu(batch.reshape(len(batch),4250)))
        expected_output=torch.cat(expected_output);maximum=float((observed-expected_output).abs().max())
        assert torch.allclose(observed,expected_output,atol=1e-5,rtol=5e-6)
        affine={k:v.detach().cpu() for k,v in model.state_dict().items() if k.endswith('norm.weight') or k.endswith('norm.bias')}
        assert len(affine)==2 and min(float(v.abs().min()) for k,v in affine.items() if k.endswith('weight'))>1e-8
        budget();output_path=folder/'engineering_held_conditions.npy' if mode=='profile' else folder/'OOF_conditions.npy'
        with output_path.open('xb') as f:np.save(f,observed.numpy(),allow_pickle=False)
        with (folder/'export_array_rows.npy').open('xb') as f:np.save(f,held.numpy(),allow_pickle=False)
        if mode=='profile':
            # Exercise actual serialization/restore without publishing a learned model.
            state=T.snapshot(torch,model,scale,optimizer,scheduler,shuffle,1,9,reg,context['protocol']['sha256'],producer,torch.device('cuda'))
            restored=T.restore(torch,state,model,scale,optimizer,scheduler,shuffle,reg,context['protocol']['sha256'],producer,torch.device('cuda'),9,225)
            assert restored==(1,9)
        G.atomic(folder/'completion.json',dict(status='PASS_ACTUAL_TRAIN_ONLY_OOF_ASSET_ENGINEERING' if mode=='profile' else 'COMPLETE_NATIVE_FULL1654_TRAIN_ONLY_FOLD_AND_HELD_EXPORT',protocol=context['protocol'],task_id=task_id,temporary_updates=9 if mode=='profile' else 0,learned_weights_saved=mode=='fit',updates=updates,epochs=count,fit_EEG_exposures=8270*count,held_queries=8270,raw_conditions=G.ref(output_path),export_array_rows=G.ref(folder/'export_array_rows.npy'),final_model=G.ref(folder/'brain_final.pt') if mode=='fit' else None,all8270_CPU_FP32_export_checked=True,CPU_GPU_maximum_absolute_error=maximum,teacher_dropout=.1,teacher_weights_frozen=True,own_affine_keys=list(affine),official_test_payloads_loaded=False,no_held_visual_target_supervision=True,final_component_or_benchmark_admitted=False,not_independent_H=True,future_GPU_bit_exactness_not_claimed=True,seconds=time.monotonic()-began,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmHWM:')))*1024-baseline_rss))
    except BaseException as ex:
        G.atomic(folder/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc(),seconds=time.monotonic()-began,no_automatic_retry=True));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['profile','fit'],required=True);p.add_argument('--task',required=True);a=p.parse_args();main(a.protocol,a.mode,a.task)
