"""SERVER ONLY: one contrastive objective pulled through the frozen native interface.

Matched to the pre-A reference's EEG model, seeds, source split, order and budget.
This is a mechanistic ablation, not an established novel method or paper reproduction.
"""
import ast
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
from typing import Dict, List, Tuple, Union

assert sys.platform == 'linux'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
R = Path('/root/autodl-tmp/mced_research_bundle')
B = Path('/root/autodl-tmp/mced_bootstrap_20261011')
OUT = Path('/root/mced_eeg_brainhive_condition_pullback_training_v1')
NATIVE = Path('/root/mced_eeg_brainhive_feature_contract_v1')
PRIOR = Path('/root/mced_eeg_brainhive_ram_assets_v1')
PRE = Path('/root/mced_eeg_brainhive_native_reference_v1')
AUDIT = Path('/root/mced_eeg_brainhive_reference_audit_v1')
DATA = Path('/root/mced_eeg_full_montage_cache_v3')
SPLIT = Path('/root/mced_eeg_native_target_level_reference_v1')
assert not OUT.exists()
OUT.mkdir()
START = time.monotonic()
FDS, SAVED = [], None
SEEDS = [2026109501, 2026109502]
COUNTS = dict(train_calls=0, train_EEG_states=0, optimizer_updates=0, backward_calls=0,
              train_interface_calls=0, train_interface_states=0,
              target_interface_calls=0, target_interface_states=0,
              eval_EEG_calls=0, eval_EEG_states=0, eval_interface_calls=0, eval_interface_states=0)

def identity(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8 << 20), b''):
            h.update(chunk)
    return dict(bytes=p.stat().st_size, sha256=h.hexdigest())

def write(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def validate(directory):
    c = json.loads((directory/'final_commit.json').read_text())
    assert c['complete']
    for name, row in c['files'].items():
        assert identity(directory/name) == row, (directory, name)

def adopt(path, expected):
    fd = os.open(path, os.O_RDONLY)
    FDS.append(fd)
    own = Path(f'/proc/{os.getpid()}/fd/{fd}')
    assert fcntl.fcntl(fd, 1034) == 15 and identity(own) == expected
    return own

try:
    write('plan.json', dict(
        script=identity(Path(__file__)), pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Test generator-condition contrastive geometry while retaining the original pre-adapter EEG parameterization',
        mathematical_objective='C(q)=flatten(P(A(q))); symmetric InfoNCE on normalized C(q_theta(EEG)) and C(F(image)); gradients through frozen A and P, one loss only',
        reference='Pinned native BrainMLP pre-A model, same seeds, input order, normalizers, source split, initialization program, optimizer and 25epochs; existing immutable pre-A checkpoints reused',
        teacher_vs_previous='Unlike direct post-A teacher training, EEG q remains in pre-A coordinates; both teacher and query traverse the same fixed nonlinear condition map',
        seeds=SEEDS, source_fit_images=15260, source_validation_images=1280, source_epochs=25,
        updates_per_seed=375, expected_optimizer_updates=750, expected_train_EEG_states=763000,
        EEG='Source01 mean4, exact author17O+P channels and250samples; source-fit channel moments reused from immutable matched reference',
        optimizer='AdamW5e-4, betas(.9,.995), wd.05 excludingbias/LayerNorm, cosine375steps,warmup0,gradclip1,learnedscaleinitlog(1/.07)',
        architecture='Unmodified reviewed BrainMLP4250to1024,one1024residualblock,dropout.1,6,452,225trainable inclscale; no new trainable module',
        arithmetic='FP32,TF32off; condition map adds compute and activations, equal update/state budget is not equal FLOPs or walltime',
        validation='Both training seeds finish before validation; report both pre-F and actual condition retrieval for new model and fixed reference. No selection, official test or target payload.',
        decision='Actual image generation is required; retrieval or condition similarity cannot establish quality. Stop this concrete objective if independent image confirmation fails.',
        prior_art='Contrastive learning and pullback geometry are established tools; this test alone does not establish originality',
        novelty_established=False, full_paper_reproduction=False, SOTA=False))
    for name in ['eeg_brainhive_teacher_confirmation_samples_v1', 'eeg_brainhive_teacher_confirmation_metrics_v1']:
        assert (B/(name+'.exit')).read_text().strip() == '0'
    for directory in [NATIVE, PRE, AUDIT]:
        validate(directory)
    previous = json.loads((PRE/'results.json').read_text())
    assert previous['counts']['optimizer_updates'] == 750
    bindings = json.loads((PRE/'input_bindings.json').read_text())
    assert identity(DATA/'sub-01_train_mean4.pt') == bindings['source_EEG']
    assert identity(SPLIT/'split.json') == bindings['source_split']
    a = next(r for r in json.loads((NATIVE/'results.json').read_text())['exports'] if r['stage']=='pre_adapter')
    feature_path = adopt(a['memfd_path'], {k:a[k] for k in ['bytes','sha256']})
    assets = json.loads((PRIOR/'results.json').read_text())['assets']
    owned = {}
    for name in ['fusion_encoder/diffusion_pytorch_model.safetensors', 'proj/diffusion_pytorch_model.safetensors']:
        asset = next(r for r in assets if r['path']==name)
        owned[name] = adopt(asset['memfd_path'], {k:asset[k] for k in ['bytes','sha256']})
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from safetensors.torch import load_file
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 8 << 30
    SAVED = (torch.get_rng_state(),torch.cuda.get_rng_state_all(),torch.get_num_threads(),
             torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scope = dict(torch=torch,nn=nn,F=F,Dict=Dict,List=List,Tuple=Tuple,Union=Union,
                 ModelMixin=ModelMixin,ConfigMixin=ConfigMixin,register_to_config=register_to_config)
    for filename, sha, names in [
        ('main/layers.py','560b038708652fac9baa98a1f8f642812eacb7bbb172a56e2c8199ebbdc5b559',['BaseMLP','ResNet1DLayer','ResNet1DBlock']),
        ('main/models_brain.py','fc06ed93c1f1f8659605ec561d87e85f91ca72856fe034f4cf8416c2b60b39b1',['BrainMLP']),
        ('main/models_adapter.py','b6d6322b7970af91a748dafbe779a8e77a1db489446cd6d607e506c36308760f',['IPProjectionModel'])]:
        p = AUDIT/'source'/filename
        assert identity(p)['sha256'] == sha
        nodes = [n for n in ast.parse(p.read_bytes()).body if isinstance(n,ast.ClassDef) and n.name in names]
        assert [n.name for n in nodes] == names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope)
    residual = scope['ResNet1DLayer'](hidden_size=1024, intermediate_size=4096, norm=False)
    weights = load_file(str(owned['fusion_encoder/diffusion_pytorch_model.safetensors']),device='cpu')
    selected = {k[len('adapter.'):]:v for k,v in weights.items() if k.startswith('adapter.')}
    assert set(selected) == set(residual.state_dict())
    residual.load_state_dict(selected,strict=True)
    del weights, selected
    config = {k:v for k,v in json.loads((PRIOR/'prior/proj/config.json').read_text()).items() if not k.startswith('_')}
    proj = scope['IPProjectionModel'](**config)
    weights = load_file(str(owned['proj/diffusion_pytorch_model.safetensors']),device='cpu')
    assert set(weights) == set(proj.state_dict())
    proj.load_state_dict(weights,strict=True)
    del weights
    residual = residual.cuda().eval().requires_grad_(False)
    proj = proj.cuda().eval().requires_grad_(False)
    targets = torch.load(feature_path,weights_only=True).float().cuda()
    assert targets.shape == (16540,1024) and torch.isfinite(targets).all()
    # A central directional check in FP64 verifies the frozen map transmits gradients.
    residual.double(); proj.double()
    q = targets[:2].double().clone().requires_grad_(True)
    direction = torch.cos(torch.arange(q.numel(),device='cuda',dtype=torch.float64)).reshape_as(q)/math.sqrt(q.numel())
    probe = torch.sin(torch.arange(2*8192,device='cuda',dtype=torch.float64)).reshape(2,8192)/math.sqrt(2*8192)
    def scalar(v):
        return (F.normalize(proj(residual(v)).flatten(1),dim=-1)*probe).sum()
    analytic = float((torch.autograd.grad(scalar(q),q)[0]*direction).sum())
    eps = 1e-4
    with torch.no_grad():
        finite = float((scalar(q+eps*direction)-scalar(q-eps*direction))/(2*eps))
    assert abs(analytic-finite) <= 1e-8+1e-5*abs(finite), (analytic,finite)
    write('gradient_preflight.json',dict(complete=True,analytic_directional=analytic,central_directional=finite,
          epsilon=eps,FP64_check_only=True,frozen_map_parameters_have_no_grad=all(p.grad is None for m in [residual,proj] for p in m.parameters()),
          numerical_probe_is_not_EEG_training=True))
    del q, direction, probe
    residual.float(); proj.float()
    fixed = [p for m in [residual,proj] for p in list(m.parameters())+list(m.buffers())]
    versions = [p._version for p in fixed]
    def condition(v):
        return F.normalize(proj(residual(v)).flatten(1),dim=-1)
    with torch.no_grad():
        unit = []
        for part in targets.split(256):
            unit.append(condition(part))
            COUNTS['target_interface_calls'] += 1
            COUNTS['target_interface_states'] += len(part)
        unit = torch.cat(unit)
    assert unit.shape == (16540,8192) and torch.isfinite(unit).all()
    concepts = set(json.loads((SPLIT/'split.json').read_text())['heldout_concepts'])
    fit = torch.tensor([i for i in range(16540) if i//10 not in concepts])
    val = torch.tensor([i for i in range(16540) if i//10 in concepts])
    assert len(fit)==15260 and len(val)==1280
    state = torch.load(PRE/f'{SEEDS[0]}_source_final.pt',weights_only=True)
    mean, std = state['normalizer_mean'], state['normalizer_std']
    assert state['selected_channels'] == bindings['selected_channel_names']
    indices = bindings['selected_channel_indices']
    x = torch.load(DATA/'sub-01_train_mean4.pt',weights_only=True,mmap=True)[:,indices,:].clone()
    x = ((x-mean)/std).cuda()
    assert x.shape == (16540,17,250) and torch.isfinite(x).all()
    del state
    write('input_bindings.json',dict(pre_reference_commit=identity(PRE/'final_commit.json'),
        source_EEG=bindings['source_EEG'],source_split=bindings['source_split'],
        selected_channel_names=bindings['selected_channel_names'],selected_channel_indices=indices,
        native_feature_commit=identity(NATIVE/'final_commit.json'),native_features=a,
        frozen_prior_commit=identity(PRIOR/'final_commit.json'),condition_dim=8192,
        no_official_test_or_target_data=True))
    rows = []
    def state_sha(model):
        h = hashlib.sha256()
        for n,t in model.state_dict().items():
            h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    for seed in SEEDS:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = scope['BrainMLP'](in_channels=4250,hidden_size=1024,num_layers=1,dropout=.1).cuda()
        initial_sha = state_sha(model)
        log_scale = nn.Parameter(torch.tensor(math.log(1/.07),device='cuda'))
        parameters = list(model.parameters())+[log_scale]
        assert sum(p.numel() for p in parameters)==6452225
        norm_ids = {id(p) for m in model.modules() if isinstance(m,nn.LayerNorm) for p in m.parameters()}
        decay = [p for n,p in model.named_parameters() if not n.endswith('bias') and id(p) not in norm_ids]+[log_scale]
        no_decay = [p for n,p in model.named_parameters() if n.endswith('bias') or id(p) in norm_ids]
        optimizer = torch.optim.AdamW([dict(params=decay,weight_decay=.05),dict(params=no_decay,weight_decay=0.)],lr=5e-4,betas=(.9,.995))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:.5*(1+math.cos(math.pi*step/375)))
        generator = torch.Generator().manual_seed(seed+100)
        sequence = hashlib.sha256(); loss_total=0.; updates=0
        model.train()
        for epoch in range(25):
            for ids in fit[torch.randperm(len(fit),generator=generator)].split(1024):
                sequence.update(ids.numpy().tobytes())
                query = model(x[ids]); out = condition(query)
                logits = out@unit[ids].T*log_scale.exp()
                labels = torch.arange(len(ids),device='cuda')
                loss = .5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();updates+=1;loss_total+=float(loss.detach())
                for k in ['train_calls','optimizer_updates','backward_calls','train_interface_calls']:COUNTS[k]+=1
                COUNTS['train_EEG_states']+=len(ids);COUNTS['train_interface_states']+=len(ids)
            if epoch==0 or (epoch+1)%5==0:
                print(json.dumps(dict(stage='CONDITION_PULLBACK_TRAIN',seed=seed,epoch=epoch+1,updates=updates,mean_loss=loss_total/updates,counts=COUNTS)),flush=True)
        ref = next(r for r in previous['training'] if r['seed']==seed)
        assert updates == ref['updates'] == 375 and sequence.hexdigest() == ref['input_sequence_sha256']
        name = f'{seed}_source_final.pt'
        torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},logit_scale=log_scale.detach().cpu(),seed=seed,
            normalizer_mean=mean,normalizer_std=std,selected_channels=bindings['selected_channel_names'],
            input_dim=4250,output_dim=1024,source_epochs=25,updates=375),OUT/name)
        row = dict(seed=seed,checkpoint=name,checkpoint_identity=identity(OUT/name),updates=375,
            input_sequence_sha256=sequence.hexdigest(),initial_state_sha256=initial_sha,
            trainable_parameters=6452225,mean_loss=loss_total/375,matched_reference_row=ref)
        rows.append(row)
        write(f'{seed}_source_ready.json',dict(complete=True,**row,validation_not_read=True))
        del model,optimizer,scheduler,parameters,log_scale,decay,no_decay,query,out,logits,loss
        torch.cuda.empty_cache()
    write('training_complete_before_validation.json',dict(complete=True,counts=dict(COUNTS),official_test_or_target_read=False))
    evaluations=[]
    with torch.inference_mode():
        for row in rows:
            for arm,directory in [('pullback',OUT),('pre_reference',PRE)]:
                state=torch.load(directory/row['checkpoint'],weights_only=True)
                model=scope['BrainMLP'](in_channels=4250,hidden_size=1024,num_layers=1,dropout=.1).cuda().eval().requires_grad_(False)
                model.load_state_dict(state['state'],strict=True)
                prediction=[];predicted_condition=[]
                for part in val.split(256):
                    q=model(x[part]);prediction.append(q);predicted_condition.append(condition(q))
                    COUNTS['eval_EEG_calls']+=1;COUNTS['eval_EEG_states']+=len(part)
                    COUNTS['eval_interface_calls']+=1;COUNTS['eval_interface_states']+=len(part)
                q=torch.cat(prediction); c=torch.cat(predicted_condition)
                result=dict(seed=row['seed'],arm=arm)
                for space,scores in [('pre_F',F.normalize(q,dim=-1)@F.normalize(targets[val],dim=-1).T),('native_condition',c@unit[val].T)]:
                    diag=scores.diag();rank=(scores>=diag[:,None]).sum(1)-1
                    result[space]=dict(top1=float((rank<1).float().mean()),top5=float((rank<5).float().mean()),mean_cosine=float(diag.mean()))
                evaluations.append(result);print(json.dumps(result),flush=True)
                del state,model,prediction,predicted_condition,q,c,scores
                torch.cuda.empty_cache()
    assert COUNTS == dict(train_calls=750,train_EEG_states=763000,optimizer_updates=750,backward_calls=750,
        train_interface_calls=750,train_interface_states=763000,target_interface_calls=65,target_interface_states=16540,
        eval_EEG_calls=20,eval_EEG_states=5120,eval_interface_calls=20,eval_interface_states=5120), COUNTS
    assert [p._version for p in fixed]==versions and all(p.grad is None for p in fixed)
    write('results.json',dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-START,
        counts=COUNTS,training=rows,evaluations=evaluations,all_training_complete_before_validation=True,
        frozen_prior_unchanged=True,same_input_order_updates_and_trainable_parameters=True,equal_compute=False,
        official_test_or_target_read=False,image_quality_established=False,novelty_established=False,SOTA=False))
    write('final_commit.json',dict(complete=True,files={p.name:identity(p) for p in OUT.iterdir() if p.is_file()}))
except BaseException:
    write('failure.json',dict(traceback=traceback.format_exc(),counts=COUNTS,seconds=time.monotonic()-START))
    raise
finally:
    if SAVED is not None:
        cpu,cuda,threads,tf32,cudnn=SAVED
        torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda);torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=cudnn
    for fd in FDS:
        os.close(fd)
