"""Original fixed five-strength Dfit calibration after complete matching fit."""
import ast,csv,hashlib,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/HIVE_matching_ordinary_calibration_v264';out.mkdir();started=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(n,v):
    b=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();p=out/n;t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
def save(n,v):
    p=out/n;t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:
        import torch
        torch.save(v,f);f.flush();os.fsync(f.fileno())
    t.replace(p)
try:
    import torch
    from torch.nn import functional as F
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.configuration_utils import ConfigMixin,register_to_config
    from safetensors.torch import load_file
    from PKU_HIVE_native_classes_20261109_v239 import classes
    from PKU_ordinary_component_source_v253 import BoundedConditionAdapter
    torch.set_num_threads(4);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    fit=R/'runs/HIVE_matching_ordinary_sub03_v259';chain=json.loads((fit/'chain_completion.json').read_text());assert chain['status']=='COMPLETE_FIXED_SINGLE_SUB03_ORDINARY_FIT_AND_CHECKPOINT_AUDIT_PRECALIBRATION'
    proof=json.loads((R/('runs/matching_ordinary_state_audit_'+chain['job_id']+'/completion.json')).read_text());assert proof['full_training_complete'] and proof['updates']==2481 and proof['exposures']==39690
    assert sha(fit/'resume.pt')==proof['checkpoint_sha256']==chain['checkpoint_sha256']
    state=torch.load(fit/'resume.pt',map_location='cpu',weights_only=False);assert state['coverage'].all() and state['seen']==39690 and state['updates']==2481
    split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());concepts=set(split['d_fit_concepts']);check=set(split['d_check_concepts']);assert len(concepts)==82 and len(check)==83 and concepts.isdisjoint(check)
    with (R/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['dataset_split']=='train']
    ids=[int(r['array_row']) for r in rows if r['concept_id'] in concepts];check_ids=[int(r['array_row']) for r in rows if r['concept_id'] in check]
    assert len(ids)==820 and len(check_ids)==830 and set(ids)|set(check_ids)==set(split['image_rows']['D'])
    assert set(ids).isdisjoint(split['image_rows']['T']+split['image_rows']['V']+check_ids)
    parent=R/'runs/HIVE_native_sub03_5096542';cache=torch.load(parent/'full/conditions.pt',map_location='cpu',weights_only=True)
    ckpt=torch.load(parent/'full/brain_final.pt',map_location='cpu',weights_only=True);assert ckpt['subject']=='sub-03' and ckpt['role']=='full' and cache['model_sha256']==sha(parent/'full/brain_final.pt')
    positions={int(row):i for i,row in enumerate(cache['array_rows'])};c=cache['condition'][[positions[i] for i in ids]].cuda()
    gamma=ckpt['model']['block.layers.0.norm.weight'].cuda().expand(820,-1);beta=ckpt['model']['block.layers.0.norm.bias'].cuda().expand(820,-1)
    allF=torch.load(parent/'deterministic_F_all16540.pt',map_location='cpu',weights_only=True);target=allF[ids].clone().cuda();del allF
    source=R/'code/HIVE_adapter_source_v253.py';pins=json.loads((R/'code/HIVE_generator_source_pins_v253.json').read_text());assert sha(source)==pins[source.name]
    scope=classes();scope.update(dict(ModelMixin=ModelMixin,ConfigMixin=ConfigMixin,register_to_config=register_to_config))
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='IPProjectionModel'];assert len(nodes)==1
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    prior=R/'assets/native_generator_v243/prior_H14';cfg=json.loads((prior/'fusion_encoder/config.json').read_text());A=scope['ResNet1DLayer'](cfg['hidden_size'],cfg['intermediate_size'],norm=False)
    fw=load_file(str(R/'assets/native_teacher_v239/fusion_encoder.safetensors'));A.load_state_dict({k[len('adapter.'):]:v for k,v in fw.items() if k.startswith('adapter.')},strict=True);del fw
    cfg=json.loads((prior/'proj/config.json').read_text());P=scope['IPProjectionModel'](**{k:v for k,v in cfg.items() if not k.startswith('_')});P.load_state_dict(load_file(str(prior/'proj/diffusion_pytorch_model.safetensors')),strict=True)
    assert sum(p.numel() for p in A.parameters())==8388608 and sum(p.numel() for p in P.parameters())==8400896
    A=A.cuda().half().eval().requires_grad_(False);P=P.cuda().half().eval().requires_grad_(False)
    model=BoundedConditionAdapter(predict_risk=False).cuda().eval();model.load_state_dict(state['model'],strict=True)
    grid=[0.,.25,.5,.75,1.];errors=[]
    with torch.no_grad():
        for strength in grid:
            model.radius=.25*strength;values=[]
            for lo in range(0,820,256):
                sl=slice(lo,lo+256);changed,_=model(c[sl],gamma[sl],beta[sl],gate_mode='fixed')
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=P(A(changed));real=P(A(target[sl]))
                values.append((1-F.cosine_similarity(pred.float(),real.float(),dim=-1)).mean(-1).cpu())
            errors.append(torch.cat(values))
    totals=[float(v.mean()) for v in errors];selected=min(range(5),key=lambda i:(totals[i],i));strength=grid[selected];model.radius=.25
    assert torch.isfinite(torch.stack(errors)).all()
    calibration=dict(kind='Dfit_only_global_strength_calibration',strengths=grid,Dfit_token_errors=totals,selected_strength=strength,selected_without_V_or_test_images=True,
        Dfit_subject='sub-03',Dfit_queries=820,Dfit_concepts=82,inside_norm_preserving_radius=True,tie_rule='Smaller strength',no_Dcheck_forward=True)
    save('Dfit_all820_errors.pt',dict(array_rows=torch.tensor(ids),errors=torch.stack(errors),strengths=grid))
    write('calibration.json',calibration)
    save('component_final.pt',dict(model=state['model'],arm='ordinary',subject='sub-03',seed=1729,parameters=525568,updates=2481,seen_queries=39690,passes=3,calibration=calibration,
        full_parent_sha256=cache['model_sha256'],training_checkpoint_sha256=chain['checkpoint_sha256'],input_provenance_sha256=sha(fit/'input_provenance.json'),
        checkpoint_selection='Fixed final3passes; only original five-strength Dfit calibration after fit'))
    result=dict(status='COMPLETE_MATCHED_SUB03_ORDINARY_FIXED_FINAL_AND_DFIT_CALIBRATION',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),seconds=time.monotonic()-started,
        updates=2481,seen_queries=39690,calibration=calibration,full_parent_sha256=cache['model_sha256'],training_checkpoint_sha256=chain['checkpoint_sha256'],checkpoint_sha256=sha(out/'component_final.pt'),
        calibration_sha256=sha(out/'calibration.json'),all820_error_table_sha256=sha(out/'Dfit_all820_errors.pt'),array_rows=ids,source_sha256=sha(__file__),
        new_optimizer_updates=0,new_images=0,old_ordinary_changed=False,image_gain_established=False,metric_assets_and_image_evaluation_still_required=True)
    write('completion.json',result);print(json.dumps({k:v for k,v in result.items() if k!='array_rows'}),flush=True)
except BaseException as e:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
