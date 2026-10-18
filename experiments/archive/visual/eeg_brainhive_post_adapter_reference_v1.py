"""SERVER ONLY: matched actual-post-adapter teacher EEG experiment, source TRAIN only.

Reviewed public BrainMLP class, no broken non-ATM wrapper or legacy ridge bridge.
Teacher-stage ablation using identical known components; novelty and quality unproven.
"""
import ast
import ctypes
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Dict,List,Tuple,Union

assert sys.platform=='linux' and os.uname().machine=='x86_64'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
R=Path('/root/autodl-tmp/mced_research_bundle');B=Path('/root/autodl-tmp/mced_bootstrap_20261011')
OUT=Path('/root/mced_eeg_brainhive_post_adapter_reference_v1');NATIVE=Path('/root/mced_eeg_brainhive_feature_contract_v1')
AUDIT=Path('/root/mced_eeg_brainhive_reference_audit_v1');DATA=Path('/root/mced_eeg_full_montage_cache_v3')
OLD=Path('/root/mced_eeg_native_target_level_reference_v1')
assert not OUT.exists();OUT.mkdir();START=time.monotonic();FDS=[];SAVED=None
SEEDS=[2026109501,2026109502];COUNTS=dict(train_calls=0,train_EEG_states=0,backward_calls=0,optimizer_updates=0,eval_calls=0,eval_EEG_states=0)

def identity(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())

def write(n,v,replace=False):
    p=OUT/n
    if replace:
        t=p.with_name(p.name+'.pending');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
    else:
        with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

try:
    write('plan.json',dict(script=identity(Path(__file__)),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),
        purpose='Test supervision of the actual A(F(image)) condition instead of F(image), changing only the frozen teacher stage',
        seeds=SEEDS,source_epochs=25,batch=1024,updates_per_seed=375,expected_optimizer_updates=750,
        images='Same source16540 training images, previous128heldout concepts1280images; fit15260; no target calibration or official EEG/image test access',
        EEG='Author native250ticks, mean4source01; exact17author O+P channels in published order, source-fit per-channel moments only',
        teacher='Whole-SHA-verified public TRAIN SynCLR/B32/VAE caches, frozen pretrained A(F(image)) after residual A; single intended teacher-stage change from the pre-adapter reference',
        architecture='Reviewed public BrainMLP4250->1024 plus one1024residualMLP, dropout.1,LayerNorm; direct public class avoids wrapper subject_ids signature conflict',
        objective='One symmetric cosine InfoNCE objective with learned logit scale initialized log(1/.07); no image head/ridge/MSE/extra loss',
        optimizer='AdamW5e-4,weight_decay.05 excluding bias/LayerNorm as HFTrainer,betas(.9,.995),cosine375steps,warmup0(current pinned YAML),gradient clipping1',
        initialization='Private seeded Torch native class initialization, not an author checkpoint; all source training complete before validation evaluation',
        inference='Export raw EEG1024vectors for direct P(query); A is already part of the teacher and must not be applied again. Same seeds, architecture, inputs, objective,25epochs and optimizer as pre-adapter reference.',
        evaluation='Concept-heldout source images, mean4 query;1280way and fixed32sourcegallery retrieval, not official200way mean80 results',
        full_paper_reproduction=False,novelty_established=False,SOTA=False))
    while not (NATIVE/'final_commit.json').exists():
        if (NATIVE/'failure.json').exists():raise RuntimeError('Native public prior feature contract failed')
        if (NATIVE/'plan.json').exists():os.kill(json.loads((NATIVE/'plan.json').read_text())['pid'],0)
        assert time.monotonic()-START<7200
        write('waiting_status.json',dict(pid=os.getpid(),stage='WAITING_FOR_NATIVE_PRIOR_FEATURE_CONTRACT',
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),optimizer_updates=0),True);time.sleep(15)
    for d in [NATIVE,AUDIT]:
        c=json.loads((d/'final_commit.json').read_text());assert c['complete']
        for n,v in c['files'].items():assert identity(d/n)==v
    contract=json.loads((NATIVE/'results.json').read_text());assert contract['complete'] and contract['TRAIN_only']
    a=next(x for x in contract['exports'] if x['stage']=='post_adapter')
    fd=os.open(a['memfd_path'],os.O_RDONLY);FDS.append(fd);fp=Path('/proc')/str(os.getpid())/'fd'/str(fd)
    assert fcntl.fcntl(fd,1034)==15 and identity(fp)=={k:a[k] for k in ['bytes','sha256']}
    dc=json.loads((DATA/'final_commit.json').read_text());assert dc['complete']
    for n in ['sub-01_train_mean4.pt','sub-01_train_layout.json']:assert identity(DATA/n)==dc['files'][n]
    old=json.loads((OLD/'final_commit.json').read_text());assert identity(OLD/'split.json')==old['files']['split.json']
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>4<<30
    SAVED=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),torch.get_num_threads(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    concepts=set(json.loads((OLD/'split.json').read_text())['heldout_concepts'])
    fit=torch.tensor([i for i in range(16540) if i//10 not in concepts]);val=torch.tensor([i for i in range(16540) if i//10 in concepts])
    assert len(fit)==15260 and len(val)==1280
    channels=json.loads((DATA/'sub-01_train_layout.json').read_text())['channels']
    selected=['P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2']
    indices=[channels.index(n) for n in selected];assert len(set(indices))==17
    x=torch.load(DATA/'sub-01_train_mean4.pt',weights_only=True,mmap=True)[:,indices,:].clone()
    fit_double=x[fit].double();mean=fit_double.mean((0,2))[None,:,None];std=fit_double.std((0,2),correction=0)[None,:,None].clamp_min(1e-6)
    del fit_double
    x=((x-mean.float())/std.float()).cuda();assert x.shape==(16540,17,250) and torch.isfinite(x).all()
    targets=torch.load(fp,weights_only=True).float().cuda();assert targets.shape==(16540,1024) and torch.isfinite(targets).all()
    unit=F.normalize(targets,dim=-1)
    scope=dict(torch=torch,nn=nn,F=F,Dict=Dict,List=List,Tuple=Tuple,Union=Union)
    for filename,sha,names in [('main/layers.py','560b038708652fac9baa98a1f8f642812eacb7bbb172a56e2c8199ebbdc5b559',['BaseMLP','ResNet1DLayer','ResNet1DBlock']),
        ('main/models_brain.py','fc06ed93c1f1f8659605ec561d87e85f91ca72856fe034f4cf8416c2b60b39b1',['BrainMLP'])]:
        p=AUDIT/'source'/filename;assert identity(p)['sha256']==sha
        nodes=[n for n in ast.parse(p.read_bytes()).body if isinstance(n,ast.ClassDef) and n.name in names];assert [n.name for n in nodes]==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope)
    write('input_bindings.json',dict(native_feature_commit=identity(NATIVE/'final_commit.json'),source_EEG=identity(DATA/'sub-01_train_mean4.pt'),
        source_split=identity(OLD/'split.json'),source_fit_images=15260,source_heldout_images=1280,selected_channel_names=selected,
        selected_channel_indices=indices,raw_native_features=a,prior_adapter_will_be_applied_only_at_sampling=False))
    rows=[]
    for seed in SEEDS:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=scope['BrainMLP'](in_channels=4250,hidden_size=1024,num_layers=1,dropout=.1).cuda()
        log_scale=nn.Parameter(torch.tensor(math.log(1/.07),device='cuda'))
        parameters=list(model.parameters())+[log_scale]
        assert sum(p.numel() for p in parameters)==6452225
        norm_ids={id(p) for m in model.modules() if isinstance(m,nn.LayerNorm) for p in m.parameters()}
        decay=[p for n,p in model.named_parameters() if not n.endswith('bias') and id(p) not in norm_ids]+[log_scale]
        no_decay=[p for n,p in model.named_parameters() if n.endswith('bias') or id(p) in norm_ids]
        optimizer=torch.optim.AdamW([dict(params=decay,weight_decay=.05),dict(params=no_decay,weight_decay=0.)],lr=5e-4,betas=(.9,.995))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:.5*(1+math.cos(math.pi*step/375)))
        generator=torch.Generator().manual_seed(seed+100);sequence=hashlib.sha256();loss_total=0.;updates=0
        model.train()
        for epoch in range(25):
            for ids in fit[torch.randperm(len(fit),generator=generator)].split(1024):
                sequence.update(ids.numpy().tobytes());out=model(x[ids]);assert out.shape==(len(ids),1024)
                logits=F.normalize(out,dim=-1)@unit[ids].T*log_scale.exp()
                labels=torch.arange(len(ids),device='cuda');loss=.5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();updates+=1;loss_total+=float(loss.detach())
                COUNTS['train_calls']+=1;COUNTS['train_EEG_states']+=len(ids);COUNTS['backward_calls']+=1;COUNTS['optimizer_updates']+=1
            if epoch==0 or (epoch+1)%5==0:print(json.dumps(dict(stage='NATIVE_FUSION_EEG_TRAIN',seed=seed,epoch=epoch+1,updates=updates,mean_loss=loss_total/updates,counts=COUNTS)),flush=True)
        assert updates==375
        name=f'{seed}_source_final.pt'
        torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},logit_scale=log_scale.detach().cpu(),seed=seed,
                        normalizer_mean=mean.float(),normalizer_std=std.float(),selected_channels=selected,
                        input_dim=4250,output_dim=1024,source_epochs=25,updates=375),OUT/name)
        row=dict(seed=seed,checkpoint=name,checkpoint_identity=identity(OUT/name),updates=375,mean_loss=loss_total/375,
                 input_sequence_sha256=sequence.hexdigest(),trainable_parameters=6452225)
        rows.append(row);write(f'{seed}_source_ready.json',dict(complete=True,**row,validation_not_read=True,official_test_not_read=True))
        del model,optimizer,scheduler,parameters,log_scale,decay,no_decay;torch.cuda.empty_cache()
    assert COUNTS['optimizer_updates']==750 and COUNTS['train_EEG_states']==763000
    baseline=Path('/root/mced_eeg_brainhive_native_reference_v1')
    baseline_result=json.loads((baseline/'results.json').read_text());assert baseline_result['complete']
    assert baseline_result['counts']['optimizer_updates']==750
    for row in rows:
        old=next(x for x in baseline_result['training'] if x['seed']==row['seed'])
        assert row['input_sequence_sha256']==old['input_sequence_sha256'] and row['updates']==old['updates'] and row['trainable_parameters']==old['trainable_parameters']
    write('matched_reference_contract.json',dict(complete=True,baseline_commit=identity(baseline/'final_commit.json'),
        same_two_seeds_same_inputs_and_optimizer_updates=True,same_public_BrainMLP_class_initialization_program=True,
        same_parameter_count_and_objective=True,only_teacher_stage_changed=True))
    write('training_complete_before_validation.json',dict(complete=True,counts=dict(COUNTS),official_test_or_target_read=False))
    bindings=json.loads((NATIVE/'input_bindings.json').read_text());case_ids=bindings['fit_indices']+bindings['heldout_indices'];assert len(case_ids)==64
    predictions={};evaluations=[]
    for row in rows:
        seed=row['seed'];state=torch.load(OUT/row['checkpoint'],weights_only=True)
        model=scope['BrainMLP'](in_channels=4250,hidden_size=1024,num_layers=1,dropout=.1).cuda().eval()
        model.load_state_dict(state['state'],strict=True)
        with torch.inference_mode():
            def infer(ids):
                result=[]
                for part in ids.split(256):
                    out=model(x[part]);COUNTS['eval_calls']+=1;COUNTS['eval_EEG_states']+=len(part);result.append(out)
                return torch.cat(result)
            prediction=infer(val);scores=F.normalize(prediction,dim=-1)@unit[val].T;correct=scores.diag();rank=(scores>=correct[:,None]).sum(1)-1
            case=infer(torch.tensor(case_ids));predictions[str(seed)]=case.cpu()
            q32=case[32:];gallery=unit[torch.tensor(case_ids[32:],device='cuda')];score32=F.normalize(q32,dim=-1)@gallery.T
            rank32=(score32>=score32.diag()[:,None]).sum(1)-1
            e=dict(seed=seed,source_heldout1280_top1=float((rank<1).float().mean()),source_heldout1280_top5=float((rank<5).float().mean()),
                   source_heldout1280_mean_cosine=float(correct.mean()),source_heldout1280_raw_MSE=float((prediction-targets[val]).square().mean()),
                   fixed32_source_heldout_top1=float((rank32<1).float().mean()),fixed32_predicted_raw_norm=float(q32.norm(dim=-1).mean()),
                   fixed32_true_raw_norm=float(targets[case_ids[32:]].norm(dim=-1).mean()),
                   mean4_query_not_official_mean80_test=True)
            evaluations.append(e);print(json.dumps(e),flush=True)
        del model,state;torch.cuda.empty_cache()
    assert COUNTS==dict(train_calls=750,train_EEG_states=763000,backward_calls=750,optimizer_updates=750,eval_calls=12,eval_EEG_states=2688),COUNTS
    torch.save(dict(predictions=predictions,source_indices=case_ids),OUT/'private_source_native_queries.pt')
    write('results.json',dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-START,
        counts=COUNTS,training=rows,evaluations=evaluations,case_source_indices=case_ids,
        all_training_complete_before_validation=True,official_test_or_target_read=False,native1024_no_ridge_or_shared512_projection=True,
        matched_teacher_stage_ablation=True,full_paper_reproduction=False,image_quality_established=False,novelty_established=False,SOTA=False))
    write('final_commit.json',dict(complete=True,files={p.name:identity(p) for p in OUT.iterdir() if p.is_file() and p.name!='waiting_status.json'}))
except BaseException:
    write('failure.json',dict(traceback=traceback.format_exc(),counts=COUNTS,seconds=time.monotonic()-START));raise
finally:
    if SAVED is not None:
        cpu,cuda,threads,tf32,cudnn=SAVED;torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda);torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=cudnn
    for fd in FDS:os.close(fd)
