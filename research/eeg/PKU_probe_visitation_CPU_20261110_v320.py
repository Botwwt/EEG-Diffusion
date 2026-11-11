"""Engineering fixture and a frozen equal-weight-update-norm control."""
import json,os,socket,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
import torch
from safetensors.torch import load_file
from PKU_conditional_visitation_core_20261110_v320 import R,ROOT,sha,save,write,check_fixture
torch.set_num_threads(4);start=time.monotonic();ROOT.mkdir(exist_ok=True);out=ROOT/'engineering';out.mkdir()
done=json.loads((R/'runs/EEG_preservation_scores_v317/analysis_completion.json').read_text());assert done['status']=='COMPLETE_FULL830_FIVE_PRESERVATION_MODELS_EIGHT_ARMS_ALL14METRICS'
base=load_file(str(R/'assets/native_generator_v243/prior_H14/proj/diffusion_pytorch_model.safetensors'))
sources={}
for case,expected in [('P_abs_free','50800da83324df37fba4bf673e13c019d26635d700c643a5ffb25bc5de169614'),('P_abs_mean','8a6bc8342944fc06226c469e0db708373e5e463dffe7eb99da04a3a608e4aa63')]:
    p=R/'runs/EEG_preservation_v317'/case/'target_final.pt';assert sha(p)==expected;sources[case]=torch.load(p,map_location='cpu',weights_only=True)
delta=sources['P_abs_free']['model']['P']['proj.weight'].double()-base['proj.weight'].float().double()
anchor=sources['P_abs_mean']['model']['P']['proj.weight'].double()-base['proj.weight'].float().double()
alpha=float(anchor.norm()/delta.norm());assert 0<alpha<1
model={k:v.float().clone() for k,v in base.items()};model['proj.weight']=(base['proj.weight'].float().double()+alpha*delta).float()
norm=float((model['proj.weight'].double()-base['proj.weight'].float().double()).norm());error=abs(norm/float(anchor.norm())-1);assert error<1e-5
p=ROOT/'P_norm_control';p.mkdir();save(p/'target_final.pt',dict(case='P_norm_control',model={'P':model},updates=827,source_updates=827,new_updates=0))
proof=dict(status='PASS_FROZEN_UPDATE_NORM_CONTROL',alpha=alpha,target_update_norm=float(anchor.norm()),actual_update_norm=norm,relative_rounding_error=error,source_updates=827,new_updates=0,target_sha256=sha(p/'target_final.pt'))
write(p/'completion.json',proof);write(out/'completion.json',dict(status='PASS_VISITATION_ALGEBRA_AND_FROZEN_CONTROL',fixture=check_fixture(),norm_control=proof,seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],heldout_effects_used_for_alpha=False));print(json.dumps(proof),flush=True)
