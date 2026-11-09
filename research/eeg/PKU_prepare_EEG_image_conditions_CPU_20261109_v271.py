"""Replay every Dcheck EEG condition before a fixed, target-free image comparison."""
import csv,hashlib,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_images_v271';out.mkdir(exist_ok=True);start=time.monotonic()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,r):
    b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)

try:
    import torch
    from PKU_eeg_candidate_core_20261109_v266 import CASES,posterior_indices,make_model,encode
    from PKU_HIVE_native_classes_20261109_v239 import classes
    from PKU_ordinary_component_source_v253 import BoundedConditionAdapter
    torch.set_num_threads(4);torch.manual_seed(1729)
    study=R/'runs/EEG_candidates_v267';done=json.loads((study/'queue/completion.json').read_text())
    assert done['status']=='COMPLETE_ALL6_FIXED_T_CASES_MODELS_AND_AUDITS_IMAGES_PENDING'
    assert not (out/'conditions_completion.json').exists() and not (out/'condition_bank.pt').exists()
    policy=R/'code/EEG_image_comparison_v271.json';spec=json.loads(policy.read_text())
    split_path=R/'data/recovery_20261108_v236/current_split_manifest.json';split=json.loads(split_path.read_text());concepts=set(split['d_check_concepts'])
    manifest=R/'data/full_T_recovery_20261108_v238/image_manifest.csv'
    with manifest.open() as f:identity=[v for v in csv.DictReader(f) if v['dataset_split']=='train' and v['concept_id'] in concepts]
    assert len(identity)==830 and len({v['concept_id'] for v in identity})==83
    rows=[int(v['array_row']) for v in identity];assert len(set(rows))==830 and set(rows).issubset(split['image_rows']['D'])
    assert set(rows).isdisjoint(split['image_rows']['T']+split['image_rows']['V'])
    raw_path=R/'data/recovery_20261108_v236/sub-03_train_mean4.pt'
    assert sha(raw_path)=='85a25128f6b94e3739478c60e089d33f437a8dcbeb31cf136ce9772d2cc1615d'
    raw=torch.load(raw_path,map_location='cpu',weights_only=True,mmap=True);eeg=raw[rows].clone();del raw
    parent=R/'runs/HIVE_native_sub03_5096542';parent_path=parent/'full/brain_final.pt';parent_state=torch.load(parent_path,map_location='cpu',weights_only=True)
    assert sha(parent_path)=='8e8b1cea28a5cae88f3acc3f136c8a638109949499fdc852ea95a306e0ff9f2b'
    assert parent_state['subject']=='sub-03' and parent_state['role']=='full'
    native=classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);native.load_state_dict(parent_state['model'],strict=True)
    post,_=posterior_indices();stats=torch.load(study/'spatial_statistics.pt',map_location='cpu',weights_only=True)
    gamma=parent_state['model']['block.layers.0.norm.weight'].expand(830,-1);beta=parent_state['model']['block.layers.0.norm.bias'].expand(830,-1)
    head_path=R/'runs/HIVE_matching_ordinary_calibration_v264/component_final.pt'
    assert sha(head_path)=='f6998db9a49d566547bfbca045d47d471c771d4d0c6880322fbd9067d7024f99'
    hs=torch.load(head_path,map_location='cpu',weights_only=True);assert hs['updates']==2481 and hs['seen_queries']==39690 and hs['calibration']['selected_strength']==1.
    assert hs['full_parent_sha256']==sha(parent_path)
    head=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False);head.load_state_dict(hs['model'],strict=True);head.radius=.25
    def forward(model,case):
        with torch.inference_mode():return torch.cat([encode(model,case,eeg[i:i+64],post).float() for i in range(0,830,64)])
    c=forward(native,'A_positive');values=[c,head(c,gamma,beta,gate_mode='fixed')[0]];arms=['parent','matched_ordinary'];checks=[]
    cache=torch.load(parent/'full/conditions.pt',map_location='cpu',weights_only=True);lookup={int(v):i for i,v in enumerate(cache['array_rows'])}
    original=cache['condition'][[lookup[v] for v in rows]];assert torch.allclose(c,original,atol=5e-5,rtol=5e-5)
    checks.append(dict(case='parent',all830_forward_max_abs=float((c-original).abs().max()),model_sha256=sha(parent_path)))
    for case in CASES:
        folder=study/case;completion=json.loads((folder/'completion.json').read_text());receipt=json.loads((folder/'checkpoint_receipt.json').read_text())
        assert completion['updates']==receipt['updates']==827 and completion['exposures']==receipt['exposures']==13230
        digest=sha(folder/'state.pt');assert digest==receipt['checkpoint_sha256']==completion['checkpoint_sha256']
        audits=[json.loads(p.read_text()) for p in (study/'audits').glob(case+'_*.json')]
        assert any(v['checkpoint_sha256']==digest and v['full_T_complete'] and v['status']=='PASS_CANDIDATE_MODEL_ADAM_RNG_COVERAGE_AND_FROZEN_COORDINATES' for v in audits)
        state=torch.load(folder/'state.pt',map_location='cpu',weights_only=False);model=make_model(case,stats,device='cpu').requires_grad_(False);model.load_state_dict(state['model'],strict=True)
        assert state['updates']==827 and state['coverage'].all()
        for key in ['block.layers.0.norm.weight','block.layers.0.norm.bias']:
            assert torch.equal(state['model'][key if case.startswith('A') else 'base.'+key],parent_state['model'][key])
        c=forward(model,case);export=torch.load(folder/'full_conditions.pt',map_location='cpu',weights_only=True)
        assert export['subject']=='sub-03' and export['role']=='full_T_candidate_not_OOF' and export['checkpoint_sha256']==digest
        lookup={int(v):i for i,v in enumerate(export['array_rows'])};original=export['condition'][[lookup[v] for v in rows]]
        assert torch.allclose(c,original,atol=5e-5,rtol=5e-5),(case,float((c-original).abs().max()))
        values.extend([c,head(c,gamma,beta,gate_mode='fixed')[0]]);arms.extend([case,case+'_plus_frozen_ordinary'])
        checks.append(dict(case=case,all830_forward_max_abs=float((c-original).abs().max()),model_sha256=digest,export_sha256=sha(folder/'full_conditions.pt')))
        del state,model,export
    bank=torch.stack(values).detach().contiguous();assert bank.shape==(14,830,1024) and torch.isfinite(bank).all() and arms==spec['arms']
    payload=dict(condition=bank,array_rows=rows,stimulus_ids=[v['stimulus_id'] for v in identity],arms=arms,subject='sub-03',repeat_protocol='R4',condition_source='Independently CPU-replayed own query EEG only',
        target_image_caption_class_or_gallery_for_condition=False,policy_sha256=sha(policy),parent_sha256=sha(parent_path),ordinary_sha256=sha(head_path),model_checks=checks)
    temp=out/'condition_bank.pt.pending'
    with temp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    temp.replace(out/'condition_bank.pt')
    write(out/'conditions_completion.json',dict(status='PASS_ALL7_ENCODERS_830_OWN_EEG_FORWARDS_AND_14_FIXED_CONDITIONS',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,
        checks=checks,queries=830,arms=arms,conditions=11620,condition_bank_sha256=sha(out/'condition_bank.pt'),policy_sha256=sha(policy),identity_manifest_sha256=sha(manifest),split_sha256=sha(split_path),
        no_optimizer_updates=True,no_target_image_read=True,composition='Explicit frozen postprocessor interaction, not a separately refitted ordinary or guaranteed additive effect',new_images=0,source_sha256=sha(__file__)))
    print('PASS_ALL_830_QUERY_CONDITIONS_ALL14_ARMS',flush=True)
except BaseException as e:
    write(out/('conditions_failure_'+os.environ['SLURM_JOB_ID']+'.json'),dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
