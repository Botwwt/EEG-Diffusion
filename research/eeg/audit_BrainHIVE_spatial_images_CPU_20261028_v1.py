"""Independently recompute all eight rankings and fifteen fixed bootstraps."""
import argparse,datetime,json,time
from pathlib import Path
import numpy as np
import brainhive_spatial_images_20261028_v1 as U
from audit_BrainHIVE_temporal_image_score_CPU_20261028_v1 import metrics as independent_metrics,paired as independent_paired

def independently_decide(c,stage):
    p={b:c['measurement_vs_original__'+b] for b in ['native','ordinary']}
    harm={b:v['DINO2AFC']['difference']<=-.02 and v['DINO2AFC']['descriptive_concept_bootstrap95'][1]<0 for b,v in p.items()}
    support={b:v['DINO2AFC']['difference']>=.01 and v['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and v['target_cosine']['descriptive_concept_bootstrap95'][0]>0 and v['within_concept10_top1']['difference']>=0 for b,v in p.items()}
    action=('STOP_INSTANCE' if harm['native'] and harm['ordinary'] else 'EXPAND400') if stage==80 else ('PROMISING_FOR_FULL_EVALUATION' if support['ordinary'] else 'STOP_INSTANCE')
    return dict(action=action,both_backgrounds_large_harm=all(harm.values()),background_harm=harm,
      ordinary_background_support=support['ordinary'] if stage==400 else None,native_background_support=support['native'] if stage==400 else None,
      controls_not_substituted_for_primary=True)

def recompute(features,target,queries,stage):
    values={a:independent_metrics(features[:,i],target,stage,queries) for i,a in enumerate(U.ARMS)}
    pairs=U.contrast_pairs();c={name:independent_paired(values[a],values[b],queries[:stage]) for name,(a,b) in pairs.items()}
    interaction={k:(values['measurement__ordinary'][k]-values['original__ordinary'][k])-(values['measurement__native'][k]-values['original__native'][k]) for k in values[U.ARMS[0]]}
    c['ordinary_interaction']=independent_paired(interaction,{k:np.zeros(stage) for k in interaction},queries[:stage])
    return values,c,independently_decide(c,stage)

def run(path,stage):
    start=time.monotonic();execution,base,queries,prepared=U.prepared(path);folder=U.EVAL/f'screen{stage}/score'
    done=U.read(folder/'completion.json');assert done['execution']['sha256']==U.sha(path) and done['stage']==stage
    fd=U.read(U.checked(done['feature_completion']));features=np.load(U.checked(fd['features']),allow_pickle=False);target=np.load(U.checked(fd['fresh_targets']),allow_pickle=False)
    actual,c,decision=recompute(features,target,queries,stage);reference=U.read(U.checked(done['metric_arrays']));maximum=0.
    for a,m in actual.items():
        for k,v in m.items():maximum=max(maximum,float(np.abs(v-reference[a][k]).max()));assert np.allclose(v,reference[a][k],atol=1e-12,rtol=0)
    for name,v in c.items():
        for k,r in v.items():
            s=done['fixed11_contrasts'][name][k]
            assert abs(r['difference']-s['difference'])<1e-12
            assert r['concepts']==s['concepts'] and r['positive_concepts']==s['positive_concepts']
            assert np.allclose(r['descriptive_concept_bootstrap95'],s['descriptive_concept_bootstrap95'],atol=1e-12,rtol=0)
    assert decision==done['decision']
    mechanism={b:all(c['measurement_vs_'+e+'__'+b]['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and c['measurement_vs_'+e+'__'+b]['within_concept10_top1']['difference']>=0 for e in ['marginal','permuted_measurement']) for b in U.BACKGROUNDS}
    assert done['mechanism_direction_support_by_background']==(mechanism if stage==400 else None)
    v=dict(status='D40_ALL_EIGHT_ARM_RANKS_AND11_FIXED_CONTRASTS_INDEPENDENT_CPU_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      execution=U.ref(path),stage=stage,score_completion=U.ref(folder/'completion.json'),all_scalar_metrics=stage*8*5,
      maximum_metric_absolute_error=maximum,all11_1000_concept_bootstraps_replayed=True,decision=decision,
      same_NumPy_backend_no_independent_image_encoder_or_UNet=True,seconds=time.monotonic()-start)
    U.write(folder/'independent_CPU_audit.json',v);print(json.dumps(v),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution',type=Path,required=True);p.add_argument('--stage',type=int,choices=[80,400],required=True);a=p.parse_args();run(a.execution,a.stage)
