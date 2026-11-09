"""D35: native full-T frozen two-way InfoNCE and measured EEG-sharing graph."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import brainhive_shared_negative_20261028_v1 as D

START = time.monotonic()


def read(path): return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def checked(item):
    p=Path(item['path']);assert p.is_file() and sha(p)==item['sha256'],str(p)
    if 'bytes' in item: assert p.stat().st_size==item['bytes']
    return p


def ref(path):return dict(path=str(path),bytes=Path(path).stat().st_size,sha256=sha(path))


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024


def run(spec_path):
    spec=read(spec_path);assert spec['status']=='D35_NATIVE_SHARED_NEGATIVE_FROZEN_DIAGNOSTIC_LOCKED'
    assert spec['seed']==1729 and spec['batch_size']==1024 and spec['bootstrap_draws']==1000
    output=Path(spec['output']);assert not output.exists();output.mkdir(parents=True)
    stage='VERIFY_INPUTS';base=rss();torch=None
    def budget():
        assert time.monotonic()-START<850 and rss()-base<10<<30
        assert sum(p.stat().st_size for p in output.iterdir() if p.is_file())<128<<20
        if torch is not None:assert not torch.cuda.is_initialized()
    try:
        for item in spec['source_pins']:checked(item)
        old_spec=read(checked(spec['D30_spec']));d33=read(checked(spec['D33_completion']))
        d33spec=read(checked(spec['D33_spec']));split=read(checked(spec['split']))
        assert not d33['registered_strict_overlap_support']
        auth=read(checked(spec['authorization']));assert auth['formal_old_D_training_resumed'] is False
        index=[json.loads(x) for x in checked(d33spec['T_index']).read_text().splitlines()]
        assert len(index)==13230 and [x['array_row'] for x in index]==split['image_rows']['T']
        groups=defaultdict(list)
        for i,row in enumerate(index):
            assert row['row']==i and row['partition']=='T' and row['subject_id']=='sub-01'
            groups[row['concept_id']].append(i)
        assert len(groups)==1323 and all(len(x)==10 for x in groups.values())
        native=read(checked(old_spec['native_input_contract']))
        subject=next(x for x in native['subjects'] if x['subject']=='sub-01')
        full=next(x for x in subject['sources'] if x['role']=='full')
        registration=Path(full['parent_registration_path'])
        assert sha(registration)==full['parent_registration_sha256']
        assert set(read(registration)['training_array_rows'])==set(split['image_rows']['T'])
        cp=checked(old_spec['roles']['full']['checkpoint'])
        assert sha(cp)==full['checkpoint_sha256']
        import torch as T
        torch=T;torch.set_num_threads(2)
        from brainhive_reversible_eeg_20261028_v1 import author_brain_class
        core_class=author_brain_class(checked(old_spec['author_models_source']),checked(old_spec['author_layers_source']))
        checkpoint=torch.load(cp,map_location='cpu',weights_only=True)
        assert checkpoint['epochs']==25 and checkpoint['updates']==325 and checkpoint['seed']==1729
        assert checkpoint['registration_sha256']==full['parent_registration_sha256']
        model=core_class(in_channels=4250,hidden_size=1024,num_layers=1,dropout=.1)
        model.load_state_dict(checkpoint['model'],strict=True);model.eval().requires_grad_(False)
        assert sum(p.numel() for p in model.parameters())==6452224
        temperature=float(checkpoint['logit_scale'].double().exp());assert 0<temperature<1e6
        raw=np.load(checked(d33spec['T_trials']),mmap_mode='r',allow_pickle=False)
        y=np.load(checked(old_spec['T_targets']),mmap_mode='r',allow_pickle=False)
        assert raw.shape==(13230,4,17,250) and y.shape==(13230,1024)
        c=np.empty((13230,1024),np.float32);primitive_error=0.
        stage='FROZEN_FULL_NATIVE_FORWARD'
        with torch.inference_mode():
            state=checkpoint['model'];F=torch.nn.functional
            for lo in range(0,len(raw),256):
                x=torch.from_numpy(np.asarray(raw[lo:lo+256]).mean(1,dtype=np.float32).reshape(-1,4250))
                native_c=model(x)
                a=F.linear(x,state['proj_in.weight'],state['proj_in.bias'])
                b=F.linear(F.gelu(F.linear(a,state['block.layers.0.ff.w1.weight'])),state['block.layers.0.ff.w2.weight'])
                primitive=F.layer_norm(a+b,(1024,),state['block.layers.0.norm.weight'],state['block.layers.0.norm.bias'],eps=1e-6)
                primitive_error=max(primitive_error,float((native_c-primitive).abs().max()))
                c[lo:lo+len(x)]=native_c.numpy();budget()
        assert primitive_error<1e-5 and np.isfinite(c).all()
        np.save(output/'full_T_native_conditions.npy',c,allow_pickle=False)
        # D33 matching remains frozen. No semantic label is used to build weights.
        rec=next(x for x in d33['results'] if x['lag']==50)
        match=np.load(checked(rec['matches']),allow_pickle=False)
        source,destination,weights,graph_note=D.stimulus_graph(match['target'][:,0],match['reciprocal'],match['verified'][:,1])
        order=np.random.default_rng(1729).permutation(len(c));batch_id=np.empty(len(c),np.int32);local=np.empty(len(c),np.int32)
        batches=[]
        for bid,lo in enumerate(range(0,len(c),1024)):
            nodes=order[lo:lo+1024];batch_id[nodes]=bid;local[nodes]=np.arange(len(nodes));batches.append(nodes)
        metrics=np.empty((len(c),2,3,len(D.METRICS)),np.float64);baseline=np.empty((len(c),4),np.float64)
        receipts=[];stored_edges=[];coverage=np.zeros(len(c),np.uint8)
        stage='FROZEN_SYMMETRIC_INFONCE_BATCHES'
        for bid,nodes in enumerate(batches):
            keep=(batch_id[source]==bid)&(batch_id[destination]==bid)
            s,d,w=local[source[keep]],local[destination[keep]],weights[keep]
            permuted,notes=D.rewire(s,d,w,len(nodes),1729+bid)
            graph=np.zeros((len(nodes),len(nodes)),np.float64);graph[s,d]=w
            control=np.zeros_like(graph);control[s,permuted]=w
            assert np.array_equal(np.count_nonzero(graph,0),np.count_nonzero(control,0))
            assert np.array_equal(np.count_nonzero(graph,1),np.count_nonzero(control,1))
            assert np.allclose(graph.sum(1),control.sum(1),rtol=0,atol=1e-15)
            for gi,matrix in enumerate([graph,control]):
                result,proof=D.symmetric_batch_gradients(c[nodes],y[nodes],matrix,temperature)
                metrics[nodes,gi]=result
                if gi==0:baseline[nodes]=np.column_stack((proof['correct_probability'],proof['original_loss']))
            coverage[nodes]+=1
            receipts.append(dict(batch=bid,rows=len(nodes),**notes))
            stored_edges.extend((int(nodes[a]),int(nodes[b]),int(nodes[z]),float(t)) for a,b,z,t in zip(s,d,permuted,w))
            print(json.dumps(dict(batch=bid,rows=len(nodes),edges=len(s),seconds=time.monotonic()-START)),flush=True)
            budget()
        assert np.all(coverage==1) and np.isfinite(metrics).all()
        np.save(output/'all13230x2x3x5_query_metrics.npy',metrics,allow_pickle=False)
        np.save(output/'all13230_native_batch_baseline.npy',baseline,allow_pickle=False)
        np.save(output/'all13230_batch_order.npy',order,allow_pickle=False)
        edge_arrays=np.asarray(stored_edges,dtype=np.float64).reshape(-1,4)
        np.save(output/'all_induced_edges_source_true_control_weight.npy',edge_arrays,allow_pickle=False)
        write(output/'batch_graph_rewiring.json',dict(batches=receipts,all_rows=13230,coverage_min=int(coverage.min()),coverage_max=int(coverage.max())))
        concept=np.stack([metrics[positions].mean(0) for positions in groups.values()])
        draws_index=np.random.default_rng(1729).integers(0,1323,size=(1000,1323))
        draws=np.concatenate([concept[draws_index[lo:lo+20]].mean(1) for lo in range(0,1000,20)])
        point=concept.mean(0);difference=point[0]-point[1];dd=draws[:,0]-draws[:,1]
        low,high=np.quantile(dd,[.025,.975],axis=0)
        real,control=point[0,2],point[1,2]
        ratio=float(real[0]/control[0]) if control[0]>0 else None
        total_weight=sum(r['total_edge_weight'] for r in receipts)
        changed_weight=sum(r['changed_destination_weight'] for r in receipts)
        changed_fraction=changed_weight/total_weight if total_weight>0 else 0.
        gate=bool(real[0]>=.001 and ratio is not None and ratio>=1.25 and difference[2,3]>=.001
                  and low[2,0]>0 and low[2,3]>0 and real[4]>=.005 and changed_fraction>=.9)
        np.save(output/'all1323x2x3x5_concept_metrics.npy',concept,allow_pickle=False)
        value=dict(status='D35_COMPLETE_FULL_T_NATIVE_SYMMETRIC_GRADIENT_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   spec=ref(spec_path),T_rows=len(c),T_concepts=len(groups),batches=len(batches),batch_sizes=[len(x) for x in batches],
                   parent_checkpoint=ref(cp),parent_registration=ref(registration),temperature=temperature,
                   encoder_mode='Original25 fullT encoder, frozen eval; FIT-only diagnostic, not OOF or historical dropout step replay',
                   native_primitive_max_abs_error=primitive_error,graph=graph_note,total_inside_batch_edges=len(edge_arrays),
                   total_inside_batch_weight=total_weight,changed_control_destination_weight_fraction=changed_fraction,
                   metric_order=D.METRICS,objective_order=['EEG_to_F','F_to_EEG','symmetric'],graph_order=['shared_waveform_proxy','degree_weight_matched_rewire'],
                   mean_metrics=point.tolist(),shared_minus_control=difference.tolist(),shared_minus_control_descriptive_95_lower=low.tolist(),shared_minus_control_descriptive_95_upper=high.tolist(),
                   symmetric_mass_enrichment=ratio,registered_practical_gradient_gate_passed=gate,
                   decision='SHARED_MEASUREMENT_HAS_MATCHED_FROZEN_GRADIENT_ASSOCIATION_REQUIRES_NEW_TRAINING_AND_IMAGES' if gate
                   else 'STOP_CURRENT_SHARED_NEGATIVE_PRIORITIZATION_NO_TRAINING_ADMISSION',
                   native_condition_bank=ref(output/'full_T_native_conditions.npy'),query_metrics=ref(output/'all13230x2x3x5_query_metrics.npy'),
                   concept_metrics=ref(output/'all1323x2x3x5_concept_metrics.npy'),batch_order=ref(output/'all13230_batch_order.npy'),
                   induced_edges=ref(output/'all_induced_edges_source_true_control_weight.npy'),baseline=ref(output/'all13230_native_batch_baseline.npy'),
                   no_new_pair_search=True,D33_strict_gate_changed=False,ordinary_read_or_modified=False,
                   new_neural_models_fit=0,SGD_updates=0,new_GPU_jobs=0,new_images=0,no_automatic_training_or_image_admission=True,
                   limits='Already-exposed sub01 T, one frozen batch partition, proxy graph lacks authenticated events, concept intervals ignore unknown recording-chain dependence; gradients are condition-space not encoder-parameter steps',
                   seconds=time.monotonic()-START,peak_extra_CPU_bytes=rss()-base)
        budget();assert sha(cp)==old_spec['roles']['full']['checkpoint']['sha256']
        write(output/'completion.json',value);print(json.dumps(value),flush=True)
    except BaseException as exc:
        write(output/('failure_'+str(time.time_ns())+'.json'),dict(status='D35_FAILED_INCOMPLETE',stage=stage,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-START))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();run(a.spec)
