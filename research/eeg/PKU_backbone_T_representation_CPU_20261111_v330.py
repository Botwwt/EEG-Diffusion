"""Complete T-only encoder diagnosis while the immutable image chain continues."""
import ast,csv,json,os,socket,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
from PKU_backbone_core_20261110_v322 import *
torch.set_num_threads(4);start=time.monotonic();out=R/'runs/EEG_backbone_T_representation_v330';out.mkdir()
split,T,folds,post=input_contract();assert len(T)==13230 and len(set(T))==13230 and set(T).isdisjoint(split['image_rows']['D']+split['image_rows']['V'])
with (R/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:metadata={int(r['array_row']):r for r in csv.DictReader(f) if r['dataset_split']=='train' and int(r['array_row']) in set(T)}
assert len(metadata)==13230;Fbank=torch.load(R/'runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt',map_location='cpu',weights_only=True)
source=R/'code/PKU_backbone_representation_analysis_CPU_20261111_v325.py';check=json.loads((R/'runs/EEG_backbone_representation_preflight_v325/completion.json').read_text());assert sha(source)==check['source_sha256']
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='measure');exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),globals())
y=F.normalize(Fbank[T].double(),dim=-1);gallery_T=13230;fields=['F_MSE','F_cosine','F_same_concept_Top1','F_same_concept_2AFC','condition_norm','F_global_Top1','F_global_2AFC'];tables={};summaries=[];inputs={};max_reference_error=0.
with (out/'per_query.csv').open('x',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['case','exposure','array_row','concept_id','fold']+fields);writer.writeheader()
    with torch.inference_mode():
      for case in CASES:
        base=ROOT/case/'encoders';proof=json.loads((base/'OOF_completion.json').read_text());assert sha(base/'T_true_OOF_conditions.pt')==proof['sha256']
        bank=torch.load(base/'T_true_OOF_conditions.pt',map_location='cpu',weights_only=True);assert bank['array_rows'].tolist()==T and torch.equal(bank['folds'],folds)
        done=json.loads((base/'full/completion.json').read_text());assert sha(base/'full/conditions.pt')==done['conditions_sha256'];full=torch.load(base/'full/conditions.pt',map_location='cpu',weights_only=True);where={r:i for i,r in enumerate(full['array_rows'].tolist())}
        inputs[case]=dict(OOF_sha256=proof['sha256'],full_condition_sha256=done['conditions_sha256'],full_model_sha256=done['model_sha256'])
        for exposure,c in [('concept_OOF',bank['condition']),('full_T_training',full['condition'][[where[r] for r in T]])]:
            assert c.shape==(13230,1024) and torch.isfinite(c).all();metrics,geometry=measure(c,T);z=F.normalize(c.double(),dim=-1);top=[];afc=[];ties=0
            for lo in range(0,gallery_T,64):
                hi=min(gallery_T,lo+64);scores=z[lo:hi]@y.T;local=torch.arange(hi-lo);indices=torch.arange(lo,hi);correct=scores[local,indices];less=(scores<correct[:,None]).sum(1).double();equal=(scores==correct[:,None]).sum(1).double()-1;top.append((scores.argmax(1)==indices).double());afc.append((less+.5*equal)/(gallery_T-1));ties+=int(((scores==scores.max(1).values[:,None]).sum(1)>1).sum())
                assert torch.isfinite(scores).all()
                if lo==0:
                    direct=torch.stack([torch.stack([torch.dot(z[i],y[j]) for j in range(20)]) for i in range(4)]);err=float((direct-scores[:4,:20]).abs().max());assert err<1e-12;max_reference_error=max(max_reference_error,err)
            metrics.update(F_global_Top1=torch.cat(top),F_global_2AFC=torch.cat(afc));assert set(metrics)==set(fields)
            tables[(case,exposure)]=metrics;summaries.append(dict(case=case,exposure=exposure,queries=13230,means={k:float(v.mean()) for k,v in metrics.items()},geometry=geometry,global_Top1_tied_queries=ties))
            for i,r in enumerate(T):writer.writerow(dict(case=case,exposure=exposure,array_row=r,concept_id=metadata[r]['concept_id'],fold=int(folds[i]),**{k:float(metrics[k][i]) for k in fields}))
            print(json.dumps(summaries[-1]),flush=True)
keys=sorted({metadata[r]['concept_id'] for r in T});assert len(keys)==1323;groups={k:[i for i,r in enumerate(T) if metadata[r]['concept_id']==k] for k in keys};draw=np.random.default_rng(1729).integers(0,len(keys),size=(1000,len(keys)));contrasts=[]
for exposure in ['concept_OOF','full_T_training']:
    pairs=[(c,'MLP17') for c in CASES[1:]]+[('Joint63','Separated63'),('Joint63','Joint17')]
    for left,right in pairs:
        for field in fields:
            delta=tables[(left,exposure)][field]-tables[(right,exposure)][field];per=np.array([float(delta[groups[k]].mean()) for k in keys]);ci=np.quantile(per[draw].mean(1),[.025,.975]);contrasts.append(dict(exposure=exposure,comparison=left+'_minus_'+right,metric=field,mean=float(delta.mean()),concept95=ci.tolist(),positive_concepts=int((per>0).sum()),negative_concepts=int((per<0).sum())))
result=dict(status='COMPLETE_FULL13230_T_ONLY_FIVE_BACKBONES_TRAINING_VS_CONCEPT_OOF_NO_METHOD_FIT',CPU_job=os.environ['SLURM_JOB_ID'],queries_each=13230,concepts=1323,encoders=5,exposure_views=2,query_records=132300,F_gallery_candidates=13230,summary=summaries,comparisons=contrasts,inputs=inputs,deterministic_F_sha256=sha(R/'runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt'),measurement_source_sha256=sha(source),source_sha256=sha(__file__),per_query_sha256=sha(out/'per_query.csv'),explicit_global_dot_reference_max_error=max_reference_error,new_training_updates=0,new_images=0,Dfit_Dcheck_V_or_R80_effects_computed=False,ordinary_evaluated=False,original_image_and_final_CPU_chains_unchanged=True,interpretation='Encoder-only diagnostic. Concept OOF is upstream training-concept exclusion, not an independent confirmation set. Full vs OOF differs both training exposure and allowed data size/model identity; it is not an isolated overfitting estimate. F similarity/ranks are not generated-image gains. No retraining, selection, threshold or protocol change.',seconds=time.monotonic()-start)
write(out/'completion.json',result);assert sum(p.stat().st_size for p in out.iterdir())<48<<20;print('COMPLETE_ALL_T_ENCODER_DIAGNOSIS',flush=True)
