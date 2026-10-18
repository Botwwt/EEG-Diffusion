"""Native pretrained d16, smaller TRAIN task and legal coarse-scale objective.

Keep all original model tensors and all ten token scales for sampling. Train
only the original first seven scales (155 tokens) using native progressive
forward, with three matched-budget arms. This cannot establish generalization.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    pending=path.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');pending.replace(path)


def metric_vectors(logits,truth):
    """Per-image token NLL, predictive entropy and token accuracy."""
    import torch.nn.functional as F
    lp=logits.float().log_softmax(-1)
    nll=F.cross_entropy(logits.float().transpose(1,2),truth,reduction='none').mean(1)
    entropy=(-(lp.exp()*lp).sum(-1)).mean(1)
    accuracy=(logits.argmax(-1)==truth).float().mean(1)
    return nll,entropy,accuracy


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config_path=Path(args.config);cfg=json.loads(config_path.read_text())
    assert cfg['schema']=='avde-fast-train-learnability-v1'
    assert (cfg['training_images'],cfg['R'],cfg['training_tokens'],cfg['steps_per_arm'])==(64,4,155,200)
    assert cfg['arms']==['paired_eeg','null_domain','image_oracle'] and cfg['precision']=='fp32'
    switch=json.loads(Path(cfg['switch_receipt']).read_text());assert switch['status']=='completed'
    preflight=json.loads(Path(cfg['progressive_check']).read_text())
    assert preflight['status']=='completed' and preflight['target_future_nonleakage']=='real-data-pass'
    metric_check=json.loads(Path(cfg['metric_check']).read_text())
    assert metric_check['status']=='completed' and metric_check['actual_metric_function_sha256']==digest(__file__)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
    out=Path(cfg['out']);out.mkdir(exist_ok=False);begin=time.monotonic()
    report=dict(status='running',stage='loading_verified_training_assets',seed=0,official_test_read=False,
        scope=cfg['scope'],configuration_sha256=digest(config_path),arms={},optimizer_updates=0,
        full_paper_reproduction=False,generalization_evaluated=False,SOTA_claim=False,
        probability_risk_guarantee_claimed=False)
    write(out/'config.json',cfg);write(out/'progress.json',report)
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms.functional import to_pil_image
        source=Path(cfg['source']);sys.path.insert(0,str(source))
        from native_generation import build_native,validate_native_state,source_identity,generate_one,query_seed
        from models.labram import get_input_chans
        from utils.lr_control import filter_params
        from utils.misc import _change_builtin_print
        from utils.amp_sc import AmpOptimizer
        torch.set_num_threads(2);torch.manual_seed(0);random.seed(0);np.random.seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;_change_builtin_print(True)
        torch.cuda.reset_peak_memory_stats();report['model_source']=source_identity()
        checkpoint=Path(switch['preserved_checkpoint']['path'])
        assert digest(checkpoint)==switch['preserved_checkpoint']['sha256']
        state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        assert state['epoch']==6
        report['warm_start']=switch['preserved_checkpoint']
        assert shutil.disk_usage(out).free>2*1300*1024**2+5*1024**3
        null_root=Path(cfg['null_weight_root']);assert not null_root.exists()
        assert shutil.disk_usage(null_root.parent).free>1300*1024**2+5*1024**3
        vae,encoder,var=build_native(device='cuda',shared_aln=False,attn_l2_norm=True)
        for name,model in [('vae_local',vae),('labram_local',encoder),('var_wo_ddp',var)]:
            validate_native_state(state['trainer'][name]);model.load_state_dict(state['trainer'][name],strict=True)
        vae.eval().requires_grad_(False);encoder.eval().requires_grad_(False)
        initial={name:value.clone() for name,value in state['trainer']['var_wo_ddp'].items()}
        del state
        raw_root=Path(cfg['raw']);raw_manifest=json.loads((raw_root/'manifest.json').read_text())
        assert raw_manifest['status']=='completed' and not raw_manifest['official_test_array_decoded']
        raw_path=Path(raw_manifest['cache']['path']);assert digest(raw_path)==raw_manifest['cache']['sha256']
        assert digest(cfg['features'])=='557fc952ba74a351fd17fee2ab11bf67d7b351d0d571fb32b25728fdad5e5831'
        manifest_path=Path(cfg['images'])/'training_image_manifest.json'
        assert digest(manifest_path)=='db9786fba452da9cf40b18be45fe3ae6871447deb7eddc8cdf39dc70b5f51421'
        rows=json.loads(manifest_path.read_text())
        concepts=np.rint(np.linspace(0,1653,64)).astype(int);ids=torch.tensor(concepts*10)
        assert len(torch.unique(ids//10))==64
        raw=np.load(raw_path,allow_pickle=True)['preprocessed_eeg_data']
        eeg=torch.from_numpy(raw[ids.numpy()].copy()).float();del raw
        channels=get_input_chans([ch.upper() for ch in raw_manifest['channels']])
        target_features=torch.load(cfg['features'],map_location='cpu',weights_only=True)['img_features'][ids].float()
        transform=transforms.Compose([transforms.Resize(256,interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),transforms.ToTensor()])
        pixels=[];image_assets=[]
        for index in ids.tolist():
            row=rows[index];assert row['image_id']==index
            path=Path(cfg['images'])/'image_set'/row['member'];assert digest(path)==row['sha256']
            pixels.append(transform(Image.open(path).convert('RGB')).mul(2).sub(1))
            image_assets.append(dict(image_id=index,path=str(path),sha256=row['sha256']))
        pixels=torch.stack(pixels);inputs=[];targets=[];conditions=[]
        with torch.no_grad():
            for start in range(0,64,4):
                z,_,_=encoder(eeg[start:start+4].mean(1).div(100).unsqueeze(2).cuda(),input_chans=channels)
                conditions.append(z.cpu())
                tokens=vae.img_to_idxBl(pixels[start:start+4].cuda())
                targets.append(torch.cat(tokens,1).cpu())
                inputs.append(vae.quantize.idxBl_to_var_input(tokens).cpu())
        cond=torch.cat(conditions);tokens=torch.cat(targets);contexts=torch.cat(inputs)
        assert tokens.shape==(64,680) and contexts.shape==(64,679,32)
        assert cond.shape==(64,1024) and torch.isfinite(cond).all()
        cache=out/'training_packet.pt'
        torch.save(dict(schema=cfg['schema'],split='training_diagnostic',image_ids=ids,
            EEG_R4=cond,oracle=target_features,token_targets=tokens,legal_context=contexts,
            EEG_source_sha256=raw_manifest['cache']['sha256'],official_test_read=False),cache)
        write(out/'data_contract.json',dict(status='completed',split='training_diagnostic',R=4,
            all_queries_seen_by_warm_start_training=True,concept_ids=concepts.tolist(),image_ids=ids.tolist(),
            images=image_assets,cache=dict(path=str(cache),sha256=digest(cache)),
            transform='Resize(shorter_edge=256,BILINEAR),CenterCrop256,[-1,1]; fixed diagnostic crop',
            source_full_training_used_fresh_crops=True,oracle_excluded_from_EEG_quality_metrics=True,
            official_test_read=False,training_target_tokens_never_passed_to_free_generation=True))
        del eeg,encoder,conditions,targets,inputs;gc.collect();torch.cuda.empty_cache()
        contexts=contexts.cuda();tokens=tokens.cuda();cond=cond.cuda();target_features=target_features.cuda()
        report.update(stage='training',training_images=64,native_tokens=680,loss_tokens=155,
            model_parameters=sum(p.numel() for p in var.parameters()),precision='fp32',model_size_changed=False,
            computational_task_reduced_by_native_progressive_forward=True)
        def budget():assert time.monotonic()-begin<cfg['max_gpu_seconds'],'Predeclared 30-minute GPU budget reached'
        def evaluate(condition):
            previous=var.training;rate=var.cond_drop_rate;cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
            var.eval();var.cond_drop_rate=0.;var.prog_si=6;result={}
            with torch.no_grad():
                for label,values in [('paired',condition),('permuted',condition.roll(1,0)),('missing',torch.zeros_like(condition))]:
                    losses=[];entropies=[];accuracies=[]
                    for start in range(0,64,4):
                        logits=var(contexts[start:start+4,:154],values[start:start+4]).float()
                        assert logits.shape==(min(4,64-start),155,4096) and torch.isfinite(logits).all()
                        nll,entropy,accuracy=metric_vectors(logits,tokens[start:start+4,:155])
                        losses.extend(nll.cpu().tolist());entropies.extend(entropy.cpu().tolist())
                        accuracies.extend(accuracy.cpu().tolist())
                    result[label]=dict(NLL=float(np.mean(losses)),entropy=float(np.mean(entropies)),
                        token_accuracy=float(np.mean(accuracies)),per_image_NLL=losses)
            var.cond_drop_rate=rate;var.train(previous);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
            return result
        def sample_arm(folder,condition,step):
            previous=var.training;cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
            var.eval();var.prog_si=-1;sample_rows=[]
            with torch.no_grad():
                for position in cfg['sample_positions']:
                    for label,donor in [('paired',position),('permuted',(position-1)%64),('missing',None)]:
                        value=condition[donor:donor+1] if donor is not None else torch.zeros_like(condition[:1])
                        image,trace=generate_one(var,value,query_seed(cfg['generation_seed'],int(ids[position])),cfg=4.,top_k=900)
                        path=folder/f'q{int(ids[position])}_{label}_step{step}.png';to_pil_image(image[0].cpu()).save(path)
                        sample_rows.append(dict(image_id=int(ids[position]),condition=label,step=step,
                            path=str(path),sha256=digest(path),scale_trace=trace))
            var.prog_si=6;var.train(previous);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
            return sample_rows
        for arm in cfg['arms']:
            budget();torch.manual_seed(0);np.random.seed(0);random.seed(0)
            var.load_state_dict(initial,strict=True);var.requires_grad_(True);var.train();var.prog_si=6
            condition=cond if arm=='paired_eeg' else target_features if arm=='image_oracle' else torch.zeros_like(cond)
            folder=out/arm;folder.mkdir()
            names,parameters,groups=filter_params(var,nowd_keys={'cls_token','start_token','task_token','cfg_uncond',
                'pos_embed','pos_1LC','pos_start','start_pos','lvl_embed','gamma','beta','ada_gss','moe_bias','scale_mul'})
            for group in groups:group['weight_decay']=cfg['weight_decay']*group['wd_sc']
            optimizer=torch.optim.AdamW(groups,lr=cfg['lr'],betas=(.9,.95),fused=False)
            amp= AmpOptimizer(0,optimizer,names,parameters,cfg['grad_clip'],n_gradient_accumulation=8)
            entry=dict(status='running',step=0,metrics=[],samples=[],actual_Adam_steps=0)
            report['arms'][arm]=entry;report.update(arm=arm,step=0);write(out/'progress.json',report)
            entry['metrics'].append(dict(step=0,**evaluate(condition)))
            # Same initial weights, separate declared conditions; retain every sample.
            entry['samples']+=sample_arm(folder,condition,0)
            generator=torch.Generator().manual_seed(0);order=torch.empty(0,dtype=torch.long)
            for step in range(1,201):
                budget()
                if len(order)<32:order=torch.randperm(64,generator=generator)
                batch=order[:32].cuda();order=order[32:]
                total_loss=0.
                for index,start in enumerate(range(0,32,4)):
                    chosen=batch[start:start+4]
                    logits=var(contexts[chosen,:154],condition[chosen]);loss=F.cross_entropy(logits.transpose(1,2),tokens[chosen,:155])
                    assert torch.isfinite(loss)
                    norm,_=amp.backward_clip_step(stepping=index==7,loss=loss)
                    total_loss+=float(loss.detach())/8
                assert norm is not None and torch.isfinite(norm)
                entry['step']=step;entry['actual_Adam_steps']=step;report.update(step=step,optimizer_updates=report['optimizer_updates']+1)
                if step in cfg['evaluation_steps']:entry['metrics'].append(dict(step=step,**evaluate(condition)))
                if step%10==0:
                    report['seconds']=time.monotonic()-begin;write(out/'progress.json',report)
                    print(json.dumps(dict(arm=arm,step=step,loss=total_loss,grad_norm=float(norm),seconds=report['seconds'])),flush=True)
            counts=sorted({float(value['step']) for value in optimizer.state.values() if 'step' in value});assert counts==[200.]
            entry['samples']+=sample_arm(folder,condition,200)
            validate_native_state(var.state_dict())
            weight_folder=null_root if arm=='null_domain' else folder
            weight_folder.mkdir(exist_ok=True,parents=True);weight=weight_folder/'VAR_final.pth'
            torch.save({name:value.detach().cpu() for name,value in var.state_dict().items()},weight)
            entry.update(status='completed',weights=dict(path=str(weight),bytes=weight.stat().st_size,sha256=digest(weight)),
                optimizer_resume_saved=False,optimizer_state_not_needed_for_fixed_short_diagnostic=True)
            write(folder/'results.json',entry);write(out/'progress.json',report)
            del amp,optimizer,groups,parameters;var.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        paired=report['arms']['paired_eeg']['metrics'];null=report['arms']['null_domain']['metrics'];oracle=report['arms']['image_oracle']['metrics']
        tests=dict(oracle_reduction=oracle[0]['paired']['NLL']-oracle[-1]['paired']['NLL'],
            EEG_same_weight_permutation_gain=paired[-1]['permuted']['NLL']-paired[-1]['paired']['NLL'],
            EEG_gain_over_matched_null_training=null[-1]['paired']['NLL']-paired[-1]['paired']['NLL'])
        report.update(status='completed',stage='fast_training_diagnostic_completed',gates=dict(values=tests,
            oracle_interface=tests['oracle_reduction']>=.30,
            EEG_pairing=tests['EEG_same_weight_permutation_gain']>=.10,
            EEG_over_domain_adaptation=tests['EEG_gain_over_matched_null_training']>=.10),
            actual_Adam_updates=600,official_generation_quality_metric_computed=False,
            full_reproduction_or_low_repeat_generalization_established=False)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,message=str(error));raise
    finally:
        report['seconds']=time.monotonic()-begin;report['utc']=datetime.now(timezone.utc).isoformat()
        try:
            report['GPU_hours']=report['seconds']/3600
            report['peak_GPU_allocated_bytes']=torch.cuda.max_memory_allocated()
        except Exception:pass
        write(out/'progress.json',report);write(out/'results.json',report)
        print(json.dumps({key:value for key,value in report.items() if key!='arms'}),flush=True)


if __name__=='__main__':main()
