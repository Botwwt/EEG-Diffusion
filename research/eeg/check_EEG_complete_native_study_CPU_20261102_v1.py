"""Real saved-emission mathematics plus explicitly synthetic lifecycle fixtures.

No pretrained EEG checkpoint, complete T bank or CUDA is loaded by this check.
"""
import argparse,ast,copy,hashlib,importlib.util,json,os,tempfile,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import EEG_complete_native_study_core_20261102_v1 as M
import EEG_complete_native_runtime_20261102_v1 as R
import EEG_complete_native_study_contract_CPU_20261102_v1 as C
import audit_EEG_complete_native_study_CPU_20261102_v1 as A
import audit_EEG_complete_native_predictions_images_CPU_20261102_v1 as I
import score_EEG_complete_native_study_CPU_20261102_v1 as S
import queue_EEG_complete_native_study_CPU_20261102_v1 as Q

L=Path(__file__).resolve().parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def rejected(call):
 try:call()
 except (ValueError,AssertionError,TypeError,KeyError):return True
 raise AssertionError('Invalid case was accepted')

class SmallEEG(torch.nn.Module):
 def __init__(self):
  super().__init__();self.body=torch.nn.Linear(4,7);self.head=torch.nn.Linear(7,1024);self.logit_scale=torch.nn.Parameter(torch.zeros(()))
 def forward(self,eeg,*,input_chans):return self.head(self.body(eeg).tanh()),None,None

def run(output):
 start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(1729);rng=np.random.default_rng(1729);checks={}
 H=load('independent_complete_real_hmm',L/'audit/VAR_multiscale_native_profile_complete_20261101_v3/code/EEG_multiscale_native_class_DRAFT_CPU_20261101_v1.py')
 source=L/'audit/VAR_multiscale_native_profile_complete_20261101_v3/result/profile/scale_log_likelihoods.npy'
 emission=np.load(source,allow_pickle=False);assert emission.shape==(8,1000,10) and emission.dtype==np.float64;emission=emission.transpose(0,2,1).copy()
 errors={};gradients=0
 for arm in C.ARMS:
  x=rng.normal(0,.7,(8,10,1000));z=rng.normal(0,.5,(8,9));xt=torch.tensor(x,requires_grad=True);zt=torch.tensor(z,requires_grad=True)
  loss,_=M.training_loss(xt,zt,torch.tensor(emission),arm);loss.backward();expected,gx,gz=A.independent(arm,x,z,emission,H)
  actualz=np.zeros_like(z) if zt.grad is None else zt.grad.numpy();err=[abs(float(loss.detach())-expected),float(np.abs(xt.grad.numpy()-gx).max()),float(np.abs(actualz-gz).max())]
  assert max(err)<2e-9;errors[arm]=err;gradients+=x.size+z.size
 checks['all6_real_saved_T8_targets_and_independent_gradients']=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),class_and_stay_gradient_cells=gradients,max_errors=errors,precision='FP64',new_real_EEG_fit=False)
 x=torch.randn(2,10,1000,dtype=torch.float64,requires_grad=True);z=torch.zeros(2,9,dtype=torch.float64);e=torch.tensor(emission[:2])
 g1=torch.autograd.grad(M.training_loss(x,z,e,'global_full')[0],x,retain_graph=True)[0];g2=torch.autograd.grad(M.training_loss(x,z,e,'visual_global_full')[0],x)[0]
 assert not torch.allclose(g1,g2);checks['nonuniform_visual_teacher_and_coupled_objectives_distinct']=True
 checks['training_target_cannot_receive_gradients']=rejected(lambda:M.training_loss(x,z,e.requires_grad_(),'global_full'))
 for arm in C.ARMS:
  model=M.ImagePrior() if arm=='image_prior_full' else M.EEGDistribution(SmallEEG(),arm)
  eeg=torch.randn(3,4);v,s=model(eeg,input_chans=[0,1]);assert v.shape==(3,10,1000) and s.shape==(3,9)
  if arm!='image_prior_full':assert not model.encoder.logit_scale.requires_grad
  assert rejected(lambda:model(eeg,input_chans=[0],img_features=torch.randn(3,1024)))
  loss,_=M.training_loss(v,s,torch.tensor(emission[:3]),arm);loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
  if arm=='image_prior_full':
   a,b=model(eeg*100,input_chans=[0]);assert torch.equal(a,v) and torch.equal(b,s)
 checks['all6_strict_EEG_only_output_API_and_first_backward_on_small_synthetic_encoder']=True
 K=load('actual_native_sampler_CPU',L/'EEG_multiscale_native_kernel_DRAFT_20261101_v1.py');sampled=0
 for arm in C.ARMS:
  logits=torch.tensor(rng.normal(size=(3,10,1000)));stay=torch.tensor(rng.normal(size=(3,9)))
  logits,rho=M.sampling_parameters(logits,stay,arm);chain,u=K.draw_class_chain(logits,rho,generator=torch.Generator().manual_seed(1729))
  for i in range(3):
   captured={k:v[i:i+1].tolist() for k,v in u.items() if isinstance(v,torch.Tensor)}
   got,_=I.independent_chain(logits[i].numpy(),stay[i].numpy(),arm,captured);assert got==chain[i].tolist();sampled+=10
 checks['actual_sampler_vs_independent_inverse_CDF_all6_families']=dict(class_decisions=sampled,CPU_only=True)
 # Synthetic small model verifies the transaction's model, Adam and RNG algebra.
 torch.manual_seed(1729);model=SmallEEG().eval();opt=torch.optim.AdamW(model.parameters(),lr=1e-4);data=torch.randn(3,4)
 def step():
  opt.zero_grad();loss=model(data,input_chans=[])[0].square().mean();loss.backward();opt.step()
 step();snapshot=(R.cpu_tree(model.state_dict()),R.cpu_tree(opt.state_dict()),torch.get_rng_state());step();expected=(R.cpu_tree(model.state_dict()),R.cpu_tree(opt.state_dict()),torch.get_rng_state())
 model.load_state_dict(snapshot[0]);opt.load_state_dict(snapshot[1]);torch.set_rng_state(snapshot[2]);step();assert R.tree_equal(model.state_dict(),expected[0]) and R.tree_equal(opt.state_dict(),expected[1]) and torch.equal(torch.get_rng_state(),expected[2])
 checks['synthetic_model_Adam_CPU_RNG_exact_resume']=True
 import EEG_complete_native_bank_reader_CPU_20261102_v1 as B
 bank=B.CompleteNativeBank.__new__(B.CompleteNativeBank);bank.cache={0:np.broadcast_to(np.arange(64)[:,None,None],(64,1000,10)),206:np.broadcast_to(np.arange(13184,13230)[:,None,None],(46,1000,10))}
 positions=np.array([13229,0,13184,63,13229])
 # Production likelihoods are nonpositive; use their row identity as negative values.
 checks['bank_tail46_and_permuted_duplicate_rows']=False
 for key in bank.cache:bank.cache[key]=-bank.cache[key].astype(np.float64)
 got=bank.rows(positions);assert got.shape==(5,10,1000) and np.array_equal(got[:,0,0],-positions)
 assert rejected(lambda:bank.rows([13230])) and rejected(lambda:bank.rows([-1]));checks['bank_tail46_and_permuted_duplicate_rows']=True
 panel=[dict(stimulus_id=f's{i}',concept_id=i//10,array_row=i,query_id=f'q{i}') for i in range(400)];same,diff=I.donor_maps(panel)
 assert same[9]==0 and diff[399]==9;selected=[i for i in range(400) if i%10<2];assert len(selected)==80
 import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as locked
 from EEG_complete_native_independent_scores_CPU_20261102_v1 import independent_arrays,independent_contrast
 targets=rng.normal(size=(400,24));queries=[panel[i] for i in selected];features=targets[selected]+rng.normal(0,.3,(80,24));values=locked.metric_arrays(features,targets,queries,panel);other=independent_arrays(features,targets,queries,panel)
 assert max(np.abs(values[m]-other[m]).max() for m in S.METRICS)<2e-12
 contrasts={name:{m:dict(difference=.03,descriptive_concept_bootstrap95=[.01,.05]) for m in S.METRICS} for name in S.pairs()}
 first=S.decision(contrasts,80);assert first['expand_any'] and all(v['action']=='EXPAND400' for v in first['candidates'].values())
 last=S.decision(contrasts,400,first);assert all(v['action']=='PROMISING_FOR_MULTISUBJECT_AND_COMPLETE_METRICS' for v in last['candidates'].values())
 bad=copy.deepcopy(contrasts);bad['global_full_minus_visual_global_full']['DINO2AFC']=dict(difference=-.03,descriptive_concept_bootstrap95=[-.05,-.01]);prior=S.decision(bad,80);assert prior['candidates']['global_full']['action']=='STOP_CLEAR_HARM' and prior['expand_any']
 assert S.decision(contrasts,400,prior)['candidates']['global_full']['prior80_stop_preserved']
 none=copy.deepcopy(bad);none['chain_full_minus_global_full']['DINO2AFC']=dict(difference=-.03,descriptive_concept_bootstrap95=[-.05,-.01]);assert not S.decision(none,80)['expand_any']
 equal=copy.deepcopy(contrasts);equal['chain_full_minus_independent_full']['DINO2AFC']['difference']=0;assert S.decision(equal,400,first)['candidates']['chain_full']['action']=='STOP_INSTANCE'
 checks['stratified80_all40_concepts_metrics_and_fixed_clear_harm_mechanism_gates']=True
 # Source-level interop: exact final filename and all real CLI argument modes.
 generator=ast.parse((L/'generate_EEG_complete_native_study_GPU_20261102_v1.py').read_text(encoding='utf8'))
 assert "root/'model_final.pt'" in (L/'generate_EEG_complete_native_study_GPU_20261102_v1.py').read_text(encoding='utf8')
 q=dict(CPU_input_seconds=1200,CPU_profile_seconds=900,CPU_artifact_seconds=1800,CPU_fit_audit_seconds=900,CPU_prediction_seconds=900,CPU_score_seconds=1800)
 tasks=Q.tasks_before_admission(q)+Q.tasks_for_fits(q)+Q.tasks_for_images(q,80)+Q.tasks_for_images(q,400)
 assert len(tasks)==60 and len(set(t[0] for t in tasks))==60
 for _,name,gpu,args,_ in tasks:
  tree=ast.parse((L/name).read_text(encoding='utf8'));options={n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant)}
  assert {'--protocol'}<=options and all(x in options for x in args if x.startswith('--'))
  if gpu:assert '--job-name' in options
 checks['all60_stage_commands_match_actual_programs_and_unique_labels']=True
 assert not torch.cuda.is_initialized();result=dict(status='PASS_REAL_T8_ALL6_TARGET_GRADIENTS_AND_EXPLICIT_SYNTHETIC_INTERFACE_LIFECYCLE_CHECKS',checks=checks,pretrained_EEG_or_full_T_inputs_or_GPU_profiles_verified=False,new_GPU_jobs=0,new_scientific_images=0,seconds=time.monotonic()-start)
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);assert not output.exists();pending=output.with_suffix('.pending');pending.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');pending.replace(output);print(json.dumps(result))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
