"""Check the locked two-background gates, ties, interactions and all contrasts."""
import argparse,copy,datetime,json,time
from pathlib import Path
import numpy as np
import brainhive_measurement_images_20261028_v1 as U
from score_BrainHIVE_measurement_images_CPU_20261028_v1 import readout
from audit_BrainHIVE_measurement_images_CPU_20261028_v1 import recompute,independently_decide

def native_engineering(training_spec):
    import torch
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    train=U.read(training_spec);eng=U.read(U.checked(train['engineering']));es=U.read(U.checked(eng['spec']))
    cpu=U.read(U.checked(es['CPU_completion']));cs=U.read(U.checked(cpu['spec']));d30=U.read(U.checked(cs['D30_spec']))
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=es['runtime_manifest']))
    from brainhive_reversible_eeg_20261028_v1 import author_brain_class
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from prepare_BrainHIVE_measurement_images_CPU_20261028_v1 import native_np,ordinary_np
    cls=author_brain_class(U.checked(d30['author_models_source']),U.checked(d30['author_layers_source']))
    parent=torch.load(U.checked(d30['roles']['full']['checkpoint']),map_location='cpu',weights_only=True)['model']
    frozen=U.read(U.checked(es['ordinary_spec']));old=torch.load(U.checked(frozen['checkpoint']),map_location='cpu',weights_only=True)['model']
    ordinary=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False);ordinary.load_state_dict(old,strict=True)
    rawref=next(x for x in cpu['files'] if x['path'].endswith('/T13230_mean4.npy'))
    raw=np.load(U.checked(rawref),mmap_mode='r',allow_pickle=False);ids=np.linspace(0,13229,32,dtype=np.int64);x=np.array(raw[ids],copy=True)
    result=[]
    for label in ['original','synthetic_parameter_change']:
        state={k:v.clone() for k,v in parent.items()};g=torch.Generator().manual_seed(39011729)
        if label!='original':
            for k,v in state.items():
                if k not in ['block.layers.0.norm.weight','block.layers.0.norm.bias']:state[k]=v+1e-4*torch.randn(v.shape,generator=g)
        model=cls(4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);model.load_state_dict(state,strict=True)
        gamma=state['block.layers.0.norm.weight'];beta=state['block.layers.0.norm.bias'];parts=[]
        with torch.no_grad():
            for lo in range(0,32,16):
                c=model(torch.from_numpy(x[lo:lo+16]));o=ordinary(c,gamma.expand_as(c),beta.expand_as(c),gate_mode='fixed')[0]
                parts.append((c.numpy(),o.numpy()))
        c=np.concatenate([p[0] for p in parts]);o=np.concatenate([p[1] for p in parts]);s={k:v.numpy().astype(np.float64) for k,v in state.items()}
        nc=native_np(x.astype(np.float64),s);no=ordinary_np(nc,gamma.numpy().astype(np.float64),beta.numpy().astype(np.float64),{k:v.numpy().astype(np.float64) for k,v in old.items()})
        ce=float(np.abs(c-nc).max());oe=float(np.abs(o-no).max());assert ce<1e-3 and oe<1e-3
        result.append(dict(case=label,T_queries=32,native_NumPy64_max_error=ce,ordinary_NumPy64_max_error=oe))
    assert not torch.cuda.is_initialized();return result

def test(output,training_spec=None):
    start=time.monotonic();cases=[]
    def contrast(mean,lo,hi,coslo=.001,top1=0.):
        return dict(DINO2AFC=dict(difference=mean,descriptive_concept_bootstrap95=[lo,hi]),
          target_cosine=dict(difference=.01,descriptive_concept_bootstrap95=[coslo,.02]),within_concept10_top1=dict(difference=top1))
    harm=contrast(-.04,-.06,-.01);gain=contrast(.02,.01,.03)
    fixtures=[(80,harm,harm,'STOP_INSTANCE'),(80,harm,contrast(-.015,-.02,-.001),'EXPAND400'),
      (80,gain,harm,'EXPAND400'),(80,contrast(-.02,-.03,-.001),contrast(-.02,-.03,-.001),'STOP_INSTANCE'),
      (80,contrast(-.02,-.03,0),contrast(-.02,-.03,0),'EXPAND400'),
      (400,gain,contrast(.01,.001,.02),'PROMISING_FOR_FULL_EVALUATION'),
      (400,gain,contrast(.00999999,.001,.02),'STOP_INSTANCE'),
      (400,gain,contrast(.02,.001,.03,coslo=0),'STOP_INSTANCE'),
      (400,gain,contrast(.02,.001,.03,top1=-1e-12),'STOP_INSTANCE'),
      (400,gain,harm,'STOP_INSTANCE'),(400,harm,gain,'PROMISING_FOR_FULL_EVALUATION')]
    for i,(stage,native,ordinary,action) in enumerate(fixtures):
        c=dict(measurement_vs_original__native=native,measurement_vs_original__ordinary=ordinary)
        result=U.decision(c,stage);assert result['action']==action and result==independently_decide(c,stage)
        if i==9:assert result['native_background_support'] and not result['ordinary_background_support']
        cases.append(dict(case=i,stage=stage,action=action))
    rng=np.random.default_rng(1729);target=rng.normal(size=(400,32)).astype(np.float32)
    target/=np.linalg.norm(target,axis=1,keepdims=True);target[1]=target[0]
    q=[dict(stimulus_id=f'image{i}',concept_id=f'concept{i//10:03d}') for i in range(400)]
    values=rng.normal(size=(80,10,32)).astype(np.float32);values[0,:,]=target[0]
    metrics,c,decision=readout(values,target,q,80);other,d,independent=recompute(values,target,q,80)
    assert decision==independent and len(c)==len(d)==15
    maxerror=0.
    for arm in U.ARMS:
        for key in metrics[arm]:
            error=float(np.abs(metrics[arm][key]-other[arm][key]).max());maxerror=max(maxerror,error);assert error<1e-12
    for name in c:
        for key in c[name]:
            a,b=c[name][key],d[name][key]
            assert abs(a['difference']-b['difference'])<1e-12 and a['positive_concepts']==b['positive_concepts']
            assert np.allclose(a['descriptive_concept_bootstrap95'],b['descriptive_concept_bootstrap95'],atol=1e-12,rtol=0)
    # Equal arms must give zero interactions, including tied target images.
    equal=np.repeat(values[:,:1],10,axis=1);_,zero,zdecision=readout(equal,target,q,80)
    assert zdecision['action']=='EXPAND400'
    assert all(v['difference']==0 and v['descriptive_concept_bootstrap95']==[0,0] for a in zero.values() for v in a.values())
    result=dict(status='D39_TEN_ARM_TIES15_CONTRASTS_INTERACTION_AND11_LOCKED_GATE_CASES_CPU_PASSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=U.ref(__file__),
      gate_cases=cases,all_scalar_metrics=80*10*5,contrasts=15,bootstrap_draws_each=1000,maximum_metric_absolute_error=maxerror,
      exact_zero_equal_arm_interactions=True,new_GPU_jobs=0,new_model_fits=0,new_images=0,seconds=time.monotonic()-start)
    if training_spec is not None:result['native_T32_original_and_synthetic_only_engineering']=native_engineering(training_spec)
    result['seconds']=time.monotonic()-start
    U.write(output,result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--training-spec',type=Path);a=p.parse_args();test(a.output,a.training_spec)
