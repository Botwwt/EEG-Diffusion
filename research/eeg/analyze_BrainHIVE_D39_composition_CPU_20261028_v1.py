"""Post hoc D39 composition audit after all400 images and fixed independent audit.

No fitting, score selection, images, hyperparameter proposals or GPU operations.
Exact condition decomposition and local head derivatives are descriptive and do
not identify the causal source of a nonlinear image-metric interaction.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import argparse,datetime,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import brainhive_measurement_images_20261028_v1 as U
from brainhive_bounded_component_v1 import BoundedConditionAdapter

def run(execution,output):
    began=time.monotonic();torch.set_num_threads(2);assert not output.exists();output.mkdir(parents=True)
    x,policy,train,base,queries=U.load_execution(execution);U.complete_fits(x)
    final=U.read(U.EVAL/'queue/completion.json');assert final['final_stage']==400
    score_path=U.EVAL/'screen400/score/completion.json';score=U.read(score_path)
    independent=U.read(U.EVAL/'screen400/score/independent_CPU_audit.json');assert independent['score_completion']['sha256']==U.sha(score_path) and independent['decision']==score['decision']
    prepared=U.read(U.EVAL/'prepared/completion.json');bank=np.load(U.checked(prepared['conditions']),allow_pickle=False);assert bank.shape==(400,10,1024)
    eng=U.read(U.checked(train['engineering']));es=U.read(U.checked(eng['spec']));frozen=U.read(U.checked(es['ordinary_spec']))
    old=torch.load(U.checked(frozen['checkpoint']),map_location='cpu',weights_only=True);assert old['calibration']['selected_strength']==1
    parent=torch.load(U.checked(base['parent_checkpoint']),map_location='cpu',weights_only=True)['model']
    gamma=parent['block.layers.0.norm.weight'];beta=parent['block.layers.0.norm.bias'];head=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False);head.load_state_dict(old['model'],strict=True)
    metrics=U.read(U.checked(score['metric_arrays']));index={name:i for i,name in enumerate(U.ARMS)}
    c0=torch.from_numpy(bank[:,index['original__native']].copy());o0=torch.from_numpy(bank[:,index['original__ordinary']].copy());r0=o0-c0
    def residual(c):return head(c,gamma.expand_as(c),beta.expand_as(c),gate_mode='fixed')[0]-c
    def bootstrap(v):
        labels=[q['concept_id'] for q in queries];groups={}
        for i,label in enumerate(labels):groups.setdefault(label,[]).append(i)
        values=np.array([np.mean(v[g],dtype=np.float64) for g in groups.values()]);assert values.shape==(40,)
        rng=np.random.default_rng(1729);draws=np.mean(values[rng.integers(0,40,(1000,40))],axis=1)
        return dict(mean=float(np.mean(values)),descriptive_concept95=np.quantile(draws,[.025,.975]).tolist())
    def correlation(a,b):
        a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
        return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
    with torch.no_grad():assert (residual(c0)-r0).abs().max()<1e-6
    stats={};arrays={}
    for encoder in U.ENCODERS[1:]:
        c=torch.from_numpy(bank[:,index[encoder+'__native']].copy());o=torch.from_numpy(bank[:,index[encoder+'__ordinary']].copy())
        b=c-c0;r=o-c;dr=r-r0;total=o-o0
        assert (total-(b+dr)).abs().max()<1e-6
        with torch.no_grad():assert (residual(c)-r).abs().max()<1e-6
        jvp=torch.autograd.functional.jvp(residual,c0,b,create_graph=False,strict=False)[1]
        higher=dr-jvp
        native_gain=np.asarray(metrics[encoder+'__native']['DINO2AFC'])-metrics['original__native']['DINO2AFC']
        ordinary_gain=np.asarray(metrics[encoder+'__ordinary']['DINO2AFC'])-metrics['original__ordinary']['DINO2AFC']
        u=(c-beta)/gamma;u0=(c0-beta)/gamma;du=u-u0
        drift=b.square().mean(-1).sqrt().numpy();shift=du.square().mean(-1).sqrt().numpy()
        n2=float(b.square().sum());dr2=float(dr.square().sum());jt2=float(jvp.square().sum())
        bd=c.double()-c0.double();rd=(o.double()-c.double())-(o0.double()-c0.double());td=o.double()-o0.double()
        reconstruction=float((td.square().sum()-(bd.square().sum()+2*(bd*rd).sum()+rd.square().sum())).abs())
        assert reconstruction<1e-6 and torch.isfinite(jvp).all()
        stats[encoder]=dict(condition_delta_RMS=float(b.square().mean().sqrt()),deaffined_unit_delta_RMS=float(du.square().mean().sqrt()),
            ordinary_residual_delta_RMS=float(dr.square().mean().sqrt()),residual_change_over_encoder_change_energy=dr2/n2,
            residual_first_order_fraction=jt2/max(dr2,1e-30),residual_higher_order_over_total_energy=float(higher.square().sum())/max(dr2,1e-30),
            dot_encoder_and_residual_change_cosine=float((b*dr).sum()/torch.sqrt(b.square().sum()*dr.square().sum())),
            exact_composition_energy_identity_absolute_error=reconstruction,
            deaffined_direction_mean_norm=float(u.mean(0).norm()),deaffined_centered_second_moment=float((u-u.mean(0)).square().mean()),
            encoder_mean_shift_energy_fraction=float(du.mean(0).square().sum()/du.square().sum()*400),
            native_DINO_gain=bootstrap(native_gain),ordinary_DINO_gain=bootstrap(ordinary_gain),ordinary_interaction=bootstrap(ordinary_gain-native_gain),
            within_query_correlations_descriptive=dict(unit_drift_vs_native_gain=correlation(shift,native_gain),unit_drift_vs_ordinary_gain=correlation(shift,ordinary_gain)),
            CPU_ordinary_replay_passed=True,not_independent_causal_mechanism=True)
        arrays[encoder+'__unit_shift']=shift;arrays[encoder+'__native_DINO_gain']=native_gain;arrays[encoder+'__ordinary_DINO_gain']=ordinary_gain
        log=U.B/'arms'/encoder/'training.jsonl';lines=[json.loads(s) for s in log.read_text().splitlines()];assert len(lines)==827 and lines[-1]['cursor']==13230
        windows={}
        for label,events in [('first64',lines[:64]),('last64',lines[-64:])]:
            weights=np.array([len(e['indices']) for e in events]);windows[label]={k:float(np.average([e[k] for e in events],weights=weights)) for k in ['epsilon_loss','token_loss','anchor_loss','unit_shift_RMS']}
        stats[encoder]['fixed_first64_last64_training_log_windows']=windows;stats[encoder]['training_log']=U.ref(log)
        assert time.monotonic()-began<240
    with (output/'descriptive_per_query.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    assert not torch.cuda.is_initialized()
    result=dict(status='D39_FROZEN_ORDINARY_COMPOSITION_AND_ALL_TRAINING_LOGS_POSTHOC_CPU_AUDITED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=U.ref(__file__),execution=U.ref(execution),
        fixed_final400_score=U.ref(score_path),independent_score_audit=U.ref(U.EVAL/'screen400/score/independent_CPU_audit.json'),conditions=prepared['conditions'],ordinary=frozen['checkpoint'],
        queries=400,concepts=40,training_log_events=3308,stats=stats,per_query=U.ref(output/'descriptive_per_query.npz'),
        linearized_head_only_not_generator_or_image_quality_derivative=True,posthoc_already_exposed_development_results=True,new_models=0,new_training=0,new_images=0,CUDA_initialized=False,seconds=time.monotonic()-began)
    U.write(output/'completion.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.execution,a.output)
