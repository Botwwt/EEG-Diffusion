"""Independent full-batch Torch autograd audit of NumPy frozen-gradient readout."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def check(item):
    path=Path(item['path']);assert sha(path)==item['sha256']
    if 'bytes' in item:assert path.stat().st_size==item['bytes']
    return path


def read(p):return json.loads(Path(p).read_bytes())


def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024


def run(spec_path,completion_path,output):
    start=time.monotonic();base_rss=rss();spec=read(spec_path);done=read(completion_path)
    assert done['status']=='D35_COMPLETE_FULL_T_NATIVE_SYMMETRIC_GRADIENT_DIAGNOSTIC'
    assert sha(spec_path)==done['spec']['sha256'] and not output.exists();output.mkdir(parents=True)
    import torch
    from scipy.special import erf
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    original_spec=read(check(spec['D30_spec']));d33spec=read(check(spec['D33_spec']));d33=read(check(spec['D33_completion']))
    c=np.load(check(done['native_condition_bank']),mmap_mode='r',allow_pickle=False)
    y=np.load(check(original_spec['T_targets']),mmap_mode='r',allow_pickle=False)
    metrics=np.load(check(done['query_metrics']),mmap_mode='r',allow_pickle=False)
    concepts=np.load(check(done['concept_metrics']),mmap_mode='r',allow_pickle=False)
    order=np.load(check(done['batch_order']),allow_pickle=False)
    edges=np.load(check(done['induced_edges']),allow_pickle=False)
    baseline=np.load(check(done['baseline']),allow_pickle=False)
    assert np.array_equal(order,np.random.default_rng(1729).permutation(13230))
    assert c.shape==(13230,1024) and metrics.shape==(13230,2,3,5)
    # Independently replay 64 evenly spaced physical input rows using NumPy64.
    snapshot=torch.load(check(done['parent_checkpoint']),map_location='cpu',weights_only=True)
    weights={k:v.double().numpy() for k,v in snapshot['model'].items()}
    physical=np.load(check(d33spec['T_trials']),mmap_mode='r',allow_pickle=False)
    samples=np.linspace(0,13229,64,dtype=np.int64)
    x=np.asarray(physical[samples]).mean(1,dtype=np.float32).reshape(64,4250).astype(np.float64)
    a=x@weights['proj_in.weight'].T+weights['proj_in.bias']
    z=a@weights['block.layers.0.ff.w1.weight'].T
    z=.5*z*(1+erf(z/np.sqrt(2)))
    z=a+z@weights['block.layers.0.ff.w2.weight'].T
    z=(z-z.mean(1,keepdims=True))/np.sqrt(z.var(1,keepdims=True)+1e-6)
    z=z*weights['block.layers.0.norm.weight']+weights['block.layers.0.norm.bias']
    neural_abs=float(np.max(np.abs(z-c[samples])))
    neural_relative=float(np.max(np.linalg.norm(z-c[samples],axis=1)/np.linalg.norm(z,axis=1)))
    assert neural_abs<1e-4 and neural_relative<1e-5
    temperature=float(snapshot['logit_scale'].double().exp());assert abs(temperature-done['temperature'])<1e-12
    # Rebuild all aggregated weights independently of the producer module.
    record=next(r for r in d33['results'] if r['lag']==50)
    match=np.load(check(record['matches']),allow_pickle=False)
    graph=defaultdict(float)
    for node,target in enumerate(match['target'][:,0]):
        source,destination=node//4,int(target)//4
        weight=max(0.,min(1.,float(match['verified'][node,1])))**2*bool(match['reciprocal'][node])/4
        if source!=destination and weight>0:graph[(source,destination)]+=weight
    assert len(graph)==done['graph']['weighted_stimulus_edges']
    group=defaultdict(list)
    for i,line in enumerate(check(d33spec['T_index']).read_text().splitlines()):
        row=json.loads(line);assert row['row']==i and row['partition']=='T';group[row['concept_id']].append(i)
    rederived=np.empty(metrics.shape,np.float64);changed_weight=total_weight=0.;used=0;maximum_baseline_error=0.
    for bid,lo in enumerate(range(0,13230,1024)):
        nodes=order[lo:lo+1024];position={int(node):i for i,node in enumerate(nodes)};n=len(nodes)
        selected=[row for row in edges if int(row[0]) in position]
        assert {(int(row[0]),int(row[1])) for row in selected}=={(s,d) for s,d in graph if s in position and d in position}
        true=np.zeros((n,n));control=np.zeros_like(true)
        for source,dest,wrong,w in selected:
            source,dest,wrong=int(source),int(dest),int(wrong)
            assert dest in position and wrong in position and source!=dest and source!=wrong
            assert abs(graph[(source,dest)]-w)<1e-14
            true[position[source],position[dest]]=w;control[position[source],position[wrong]]=w
            total_weight+=w;changed_weight+=w*(dest!=wrong);used+=1
        assert np.array_equal((true>0).sum(0),(control>0).sum(0))
        assert np.array_equal((true>0).sum(1),(control>0).sum(1))
        for i in range(n):assert np.array_equal(np.sort(true[i]),np.sort(control[i]))
        tc=torch.tensor(np.asarray(c[nodes]),dtype=torch.float64,requires_grad=True)
        tf=torch.nn.functional.normalize(torch.tensor(np.asarray(y[nodes]),dtype=torch.float64),dim=-1)
        logits=temperature*torch.nn.functional.normalize(tc,dim=-1)@tf.T
        row=torch.softmax(logits,dim=1);col=torch.softmax(logits,dim=0)
        lr=(-logits.diag()+torch.logsumexp(logits,dim=1)).sum()
        lc=(-logits.diag()+torch.logsumexp(logits,dim=0)).sum()
        grad_r=torch.autograd.grad(lr,tc,retain_graph=True)[0]
        grad_c=torch.autograd.grad(lc,tc,retain_graph=True)[0]
        positive=torch.autograd.grad(logits.diag().sum(),tc,retain_graph=True)[0]
        expected=torch.stack((grad_r,grad_c,(grad_r+grad_c)/2),dim=1).detach().numpy()
        pos=positive.detach().numpy();pos_energy=np.square(pos).sum(1)
        actual_base=torch.stack((row.diag(),col.diag(),-row.diag().log(),-col.diag().log()),dim=-1).detach().numpy()
        maximum_baseline_error=max(maximum_baseline_error,float(np.max(np.abs(actual_base-baseline[nodes]))))
        for gi,w in enumerate([true,control]):
            tw=torch.tensor(w,dtype=torch.float64)
            masked_logits=logits+torch.log1p(-tw)
            mr=(-logits.diag()+torch.logsumexp(masked_logits,dim=1)).sum()
            mc=(-logits.diag()+torch.logsumexp(masked_logits,dim=0)).sum()
            gm_r=torch.autograd.grad(mr,tc,retain_graph=True)[0]
            gm_c=torch.autograd.grad(mc,tc,retain_graph=True)[0]
            shared_r=torch.autograd.grad((row.detach()*tw*logits).sum(),tc,retain_graph=True)[0]
            shared_c=torch.autograd.grad((col.detach()*tw*logits).sum(),tc,retain_graph=True)[0]
            masked=torch.stack((gm_r,gm_c,(gm_r+gm_c)/2),dim=1).detach().numpy()
            shared=torch.stack((shared_r,shared_c,(shared_r+shared_c)/2),dim=1).detach().numpy()
            h_r=(row.detach()*tw).sum(1).numpy();h_c=(col.detach()*tw).sum(0).numpy()
            mass=np.column_stack((h_r,h_c,(h_r+h_c)/2))
            fraction_r=h_r/(1-row.diag().detach().numpy());fraction_c=h_c/(1-col.diag().detach().numpy())
            fractions=np.column_stack((fraction_r,fraction_c,(fraction_r+fraction_c)/2))
            removal=expected-masked
            rederived[nodes,gi,:,:,]=np.stack((mass,fractions,
                (shared*pos[:,None,:]).sum(-1)/pos_energy[:,None],
                (removal*pos[:,None,:]).sum(-1)/pos_energy[:,None],
                np.linalg.norm(removal,axis=-1)/np.linalg.norm(expected,axis=-1)),axis=-1)
        assert time.monotonic()-start<280 and rss()-base_rss<10<<30
        print(json.dumps(dict(batch=bid,full_batch_autograd_replayed=True)),flush=True)
    error=float(np.max(np.abs(rederived-metrics)));assert error<1e-10 and maximum_baseline_error<1e-10
    assert used==done['total_inside_batch_edges'] and abs(total_weight-done['total_inside_batch_weight'])<1e-10
    derived_concept=np.stack([rederived[rows].mean(0) for rows in group.values()]);assert np.max(np.abs(derived_concept-concepts))<1e-10
    indices=np.random.default_rng(1729).integers(0,1323,size=(1000,1323))
    draws=np.concatenate([derived_concept[indices[lo:lo+20]].mean(1) for lo in range(0,1000,20)])
    point=derived_concept.mean(0);difference=point[0]-point[1];low,high=np.quantile(draws[:,0]-draws[:,1],[.025,.975],axis=0)
    for actual,key in [(point,'mean_metrics'),(difference,'shared_minus_control'),(low,'shared_minus_control_descriptive_95_lower'),(high,'shared_minus_control_descriptive_95_upper')]:assert np.max(np.abs(actual-np.asarray(done[key])))<1e-10
    real,ctrl=point[0,2],point[1,2];ratio=real[0]/ctrl[0] if ctrl[0]>0 else None;changed=changed_weight/total_weight if total_weight else 0
    gate=bool(real[0]>=.001 and ratio is not None and ratio>=1.25 and difference[2,3]>=.001 and low[2,0]>0 and low[2,3]>0 and real[4]>=.005 and changed>=.9)
    assert gate==done['registered_practical_gradient_gate_passed'] and not torch.cuda.is_initialized()
    value=dict(status='D35_ALL_QUERY_METRICS_INDEPENDENT_FULL_BATCH_TORCH_AUTOGRAD_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(__file__),primary_completion_sha256=sha(completion_path),all13230x2x3x5_scalar_metrics_replayed=True,
        scalar_metrics=396900,all13_batches_row_column_and_symmetric_gradients=True,all_graph_weights_and_within_batch_degree_profiles_verified=True,
        maximum_query_metric_abs_error=error,maximum_baseline_abs_error=maximum_baseline_error,
        raw_native_forward_NumPy64_rows=64,raw_native_forward_NumPy64_maximum_absolute_error=neural_abs,
        raw_native_forward_NumPy64_maximum_relative_error=neural_relative,entire_native_encoder_forward_independently_repeated=False,
        all1323_concepts_and1000_bootstrap_intervals_verified=True,registered_gate_recomputed=gate,
        control_random_swap_sequence_independently_replayed=False,original_stop_or_support_preserved=True,
        new_models_fit=0,new_SGD_updates=0,new_GPU_jobs=0,new_images=0,seconds=time.monotonic()-start,peak_extra_CPU_bytes=rss()-base_rss)
    with (output/'completion.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--completion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.spec,a.completion,a.output)
