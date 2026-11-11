"""Numerically check the supplemental diagnostic on ten real native OOF queries."""
import ast,json,os,socket,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
from PKU_backbone_core_20261110_v322 import *
import csv
torch.set_num_threads(4);start=time.monotonic();split,T,folds,post=input_contract()
with (R/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:metadata={int(r['array_row']):r for r in csv.DictReader(f) if r['dataset_split']=='train'}
Fbank=torch.load(R/'runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt',map_location='cpu',weights_only=True);source=R/'code/PKU_backbone_representation_analysis_CPU_20261111_v324.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='measure');exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),globals())
concept=metadata[T[0]]['concept_id'];positions=[i for i,r in enumerate(T) if metadata[r]['concept_id']==concept];assert len(positions)==10
bank=torch.load(R/'runs/HIVE_native_sub03_5096542/T_true_OOF_conditions.pt',map_location='cpu',weights_only=True);c=bank['condition'][positions];rows=[T[i] for i in positions];metrics,geometry=measure(c,rows)
z=F.normalize(c.double(),dim=-1);y=F.normalize(Fbank[rows].double(),dim=-1);errors=[]
for i in range(10):
    scores=torch.tensor([float(torch.dot(z[i],y[j])) for j in range(10)],dtype=torch.float64);value=scores[i];wrong=[j for j in range(10) if j!=i]
    errors.append(abs(float(metrics['F_cosine'][i])-float(value)));assert float(metrics['F_same_concept_Top1'][i])==float(int(scores.argmax())==i)
    expected=sum(float(scores[j]<value)+.5*float(scores[j]==value) for j in wrong)/9;errors.append(abs(float(metrics['F_same_concept_2AFC'][i])-expected))
assert max(errors)<1e-12 and 0<=geometry['condition_mean_energy_fraction']<=1+1e-12
out=R/'runs/EEG_backbone_representation_preflight_v324';out.mkdir();write(out/'completion.json',dict(status='PASS_TEN_REAL_NATIVE_OOF_QUERIES_EXPLICIT_CANDIDATE_REFERENCE',CPU_job=os.environ['SLURM_JOB_ID'],max_error=max(errors),queries=10,source_sha256=sha(source),seconds=time.monotonic()-start,new_training=0,new_images=0));print('PASS_REAL_REPRESENTATION_MEASUREMENT',flush=True)
