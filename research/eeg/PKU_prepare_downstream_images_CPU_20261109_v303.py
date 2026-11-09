"""All830 own-EEG conditions; all nine complete downstream targets, no target images."""
import json,os,socket,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
import torch
from PKU_downstream_core_20261109_v303 import R,ROOT,CASES,sha,write,save
from PKU_HIVE_native_classes_20261109_v239 import classes
from PKU_eeg_candidate_core_20261109_v266 import posterior_indices
from PKU_ordinary_component_source_v253 import BoundedConditionAdapter
out=R/'runs/EEG_downstream_images_v303';out.mkdir(exist_ok=True);start=time.monotonic();torch.set_num_threads(4)
assert json.loads((ROOT/'queue/completion.json').read_text())['status']=='COMPLETE_ALL9_DOWNSTREAM_FIXED_T_MODELS'
old=R/'runs/EEG_images_v271';proof=json.loads((old/'conditions_completion.json').read_text());assert sha(old/'condition_bank.pt')==proof['condition_bank_sha256'];bank=torch.load(old/'condition_bank.pt',map_location='cpu',weights_only=True)
assert bank['arms'][1]=='matched_ordinary' and bank['condition'].shape==(14,830,1024)
parent=torch.load(R/'runs/HIVE_native_sub03_5096542/full/brain_final.pt',map_location='cpu',weights_only=True);model=classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);model.load_state_dict(parent['model'],strict=True)
raw=torch.load(R/'data/recovery_20261108_v236/sub-03_train_mean4.pt',map_location='cpu',weights_only=True,mmap=True);x=raw[bank['array_rows']].clone();del raw;post,_=posterior_indices()
headpath=R/'runs/HIVE_matching_ordinary_calibration_v264/component_final.pt';assert sha(headpath)=='f6998db9a49d566547bfbca045d47d471c771d4d0c6880322fbd9067d7024f99';hs=torch.load(headpath,map_location='cpu',weights_only=True);head=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False);head.load_state_dict(hs['model'],strict=True)
with torch.no_grad():
    c=torch.cat([model(x[i:i+64,post,:]) for i in range(0,830,64)]);g=parent['model']['block.layers.0.norm.weight'].expand(830,-1);b=parent['model']['block.layers.0.norm.bias'].expand(830,-1);o=head(c,g,b,gate_mode='fixed')[0]
assert torch.allclose(o,bank['condition'][1],atol=5e-5,rtol=5e-5)
checks=[]
for case in CASES:
    folder=ROOT/case;done=json.loads((folder/'completion.json').read_text());assert done['updates']==827 and done['exposures']==13230 and done['target_sha256']==sha(folder/'target_final.pt')
    audits=[json.loads(p.read_text()) for p in (ROOT/'audits').glob(case+'_*.json')];assert any(v['full_T_complete'] and v['checkpoint_sha256']==done['checkpoint_sha256'] for v in audits)
    s=torch.load(folder/'target_final.pt',map_location='cpu',weights_only=True);assert s['case']==case and all(torch.isfinite(v).all() for m in s['model'].values() for v in m.values());assert sum(v.numel() for m in s['model'].values() for v in m.values())==(8400896 if case.startswith('P_') else 340787200)
    checks.append(dict(case=case,target_path=str(folder/'target_final.pt'),target_sha256=done['target_sha256'],model_family='P' if case.startswith('P_') else 'IP',updates=827));del s
payload={k:bank[k] for k in ['array_rows','stimulus_ids']};payload.update(condition=bank['condition'][1:2].expand(9,-1,-1).clone(),arms=CASES,model_checks=checks,subject='sub-03',repeat_protocol='R4',condition_source='Exact same frozen matchedordinary ownEEG, all830 independently re-forwarded',normal_inference_target_access=False)
save(out/'condition_bank.pt',payload);write(out/'conditions_completion.json',dict(status='PASS_ALL830_FROZEN_EEG_ORDINARY_AND9_DOWNSTREAM_TARGETS',condition_bank_sha256=sha(out/'condition_bank.pt'),policy_sha256=sha(R/'code/EEG_downstream_image_policy_v303.json'),checks=checks,independent_EEG_head_max_abs=float((o-bank['condition'][1]).abs().max()),job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,new_images=0))
print('PASS_ALL830_OWN_EEG_AND_ALL9_MODELS',flush=True)
