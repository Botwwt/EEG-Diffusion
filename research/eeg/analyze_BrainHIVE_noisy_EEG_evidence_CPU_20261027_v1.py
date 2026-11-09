"""D13 full readout replay, all400 CPU logits, fixed gradient replay and figures."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_noisy_EEG_evidence_20261027_v1 as V
import brainhive_conditional_flow_20261027_v1 as FLOW


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--completion',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); start=time.monotonic(); torch.set_num_threads(2)
    assert not args.output.exists(); args.output.mkdir(parents=True); assert not torch.cuda.is_initialized()
    done=U.read(args.completion); assert done['status']=='COMPLETE_D13_MATCHED_NOISY_EEG_EVIDENCE_DIAGNOSTIC'
    assert done['full_passes_per_head']==10 and done['updates_per_head']==1300 and done['main_query_exposures_per_head']==1323000
    assert done['diagnostic_only'] is True and done['new_images']==done['UNet_evaluations']==0
    spec=U.read(U.checked(done['spec'])); D=args.completion.parent
    for source in spec['source_pins']: U.checked(source)
    d12=U.read(U.checked(spec['D12_fit'])); inputs=U.read(U.checked(d12['D09_inputs'])); panel=U.read(U.checked(d12['panel']))
    qs={key:panel[key] for key in ('Dfit','Dcheck')}; split=U.read(U.checked(inputs['split']))
    assert {q['concept_id'] for q in qs['Dfit']}.issubset(set(split['d_fit_concepts']))
    assert {q['concept_id'] for q in qs['Dcheck']}.issubset(set(split['d_check_concepts']))
    assert all(set(q['array_row'] for q in value).isdisjoint(split['image_rows']['T']) for value in qs.values())
    bindings=[]
    for path in sorted(D.glob('*.npy')):
        array=np.load(path,mmap_mode='r',allow_pickle=False); assert np.isfinite(array).all()
        bindings.append(U.ref(path)|dict(shape=list(array.shape),dtype=str(array.dtype),bytes=path.stat().st_size,
            local_export=not path.name.endswith(('_gradients.npy','_states.npy'))))
    U.atomic(args.output/'all_array_bindings.json',dict(status='ALL_D13_ARRAY_FILES_HASHED_AND_FINITE',files=bindings))
    logits={}; retrieval={}; mixed={}; maps=U.read(U.checked(done['pairing_maps']))
    for part,N in [('Dfit',200),('Dcheck',400)]:
        assert maps[part]['query_ids']==[q['query_id'] for q in qs[part]]
        gen=np.random.default_rng(1729+(part=='Dcheck')); k=gen.integers(0,N,N); m=gen.integers(0,N,N)
        assert k.tolist()==maps[part]['product_brain_indices'] and m.tolist()==maps[part]['independent_raw_indices']
        assert FLOW.query_noise(qs[part])[1]==maps[part]['conditional_noise_seeds']
        logits[part]={}; retrieval[part]={}
        for arm in V.ARMS:
            array=np.load(D/(part+'_'+arm+'_logits.npy')); assert array.shape==(N,4,2); logits[part][arm]=array
            score=np.load(D/(part+'_'+arm+'_sameconcept_scores.npy')); assert score.shape==(N,4,10)
            average=score.mean(1); positive=average[np.arange(N),np.arange(N)%10]
            retrieval[part][arm]=dict(top1=float((average.argmax(1)==np.arange(N)%10).mean()),
                correct_minus_other9_logit=float((positive-(average.sum(1)-positive)/9).mean()),queries=N,
                readout='held visual-state to EEG matching diagnosis; not a reconstruction input or image metric')
        sensitivity={name:V.interval(np.load(D/(part+'_'+name+'_intervention.npy')).mean(1),qs[part]) for name in ('sameconcept_raw','otherconcept_raw')}
        mixed[part]=dict(correct_c_raw_shuffle_sensitivity=sensitivity,mixed_c_b_may_be_off_joint_support=True,probability_calibration_or_causal_increment_claim=False)
    replay=V.readout(logits,retrieval,qs); replay['raw_counterfactual_sensitivity']=mixed
    assert replay==U.read(U.checked(done['readout']))
    state=torch.load(U.checked(done['final_heads']),weights_only=True,map_location='cpu',mmap=True)
    assert state['epoch']==10 and state['updates_per_head']==1300 and state['query_exposures_per_head']==1323000
    models={arm:V.Evidence().eval().requires_grad_(False) for arm in V.ARMS}
    for arm in V.ARMS: models[arm].load_state_dict(state['models'][arm],strict=True)
    assert all(sum(p.numel() for p in model.parameters())==done['parameters_per_arm'] for model in models.values())
    cache=torch.load(U.checked(inputs['native_cache']),weights_only=True,map_location='cpu',mmap=True)
    lookup={int(row):i for i,row in enumerate(cache['array_rows'])}; ids=[q['array_row'] for q in qs['Dcheck']]
    assert cache['array_rows'][[lookup[i] for i in ids]].tolist()==ids
    c=cache['condition'][[lookup[i] for i in ids]].clone()
    actual=U.read('/root/brainhive_reproduction_20261020_v1/data/actual_input_bindings.json'); src=actual['subjects']['sub-01']
    raw=torch.load(U.checked(dict(path=src['train_path'],sha256=src['train_sha256'])),weights_only=True,map_location='cpu',mmap=True)
    b=raw[ids][:,actual['selected_indices'],:].reshape(400,4250).contiguous()
    b=(b-state['raw_T_mean'][0])/state['raw_T_std'][0]
    subjects=torch.zeros(400,dtype=torch.long); product=torch.as_tensor(maps['Dcheck']['product_brain_indices']); null=torch.as_tensor(maps['Dcheck']['independent_raw_indices'])
    states=np.load(D/'real_forward_states.npy'); assert states.shape==(400,4,1024)
    max_error={}; losses={}
    with torch.no_grad():
        cc=torch.cat([c,c[product]]); bb=torch.cat([b,b[product]]); nullbb=torch.cat([b[null]]*2); ss=torch.cat([subjects]*2)
        for arm in V.ARMS:
            pieces=[]
            for ti,t in enumerate(V.TIMES):
                x=torch.from_numpy(states[:,ti].copy()); xx=torch.cat([x,x]); tt=torch.full((800,),t)
                out=models[arm](xx,tt,cc,nullbb if arm=='shuffled_raw_train' else bb,ss,arm)
                pieces.append(torch.stack([out[:400],out[400:]],-1).numpy())
            cpu=np.stack(pieces,1); expected=logits['Dcheck'][arm]
            assert np.allclose(cpu,expected,atol=1e-3,rtol=1e-4)
            max_error[arm]=float(np.max(np.abs(cpu-expected)))
            cpu_loss=V.arrays(cpu[:,:,0],cpu[:,:,1])['log_loss'].mean(); gpu_loss=V.arrays(expected[:,:,0],expected[:,:,1])['log_loss'].mean()
            losses[arm]=float(abs(cpu_loss-gpu_loss)); assert losses[arm]<1e-5
    selected=np.arange(0,400,10); gradient_error={}
    for domain in ('real_forward','old_D12_generated_trajectory'):
        xstates=np.load(D/(domain+'_states.npy'),mmap_mode='r'); gradients=np.load(D/(domain+'_gradients.npy'),mmap_mode='r')
        assert xstates.shape==(400,4,1024) and gradients.shape==(400,4,3,1024)
        errors=[]
        for ti,t in enumerate(V.TIMES):
            tt=torch.full((40,),t); ix=torch.as_tensor(selected)
            for j,arm in enumerate(V.ARMS):
                x=torch.tensor(xstates[selected,ti].copy(),requires_grad=True)
                raw_b=b[null[ix]] if arm=='shuffled_raw_train' else b[ix]
                out=models[arm](x,tt,c[ix],raw_b,subjects[ix],arm)
                g=torch.autograd.grad(out.sum(),x)[0].numpy(); expected=gradients[selected,ti,j]
                relative=np.linalg.norm(g-expected,axis=1)/np.maximum(np.linalg.norm(expected,axis=1),1e-12)
                assert relative.max()<5e-4; errors.append(float(relative.max()))
        gradient_error[domain]=max(errors)
    records=list(csv.DictReader((D/'per_query.csv').open())); assert len(records)==1800
    for i,row in enumerate(records):
        part='Dfit' if i<600 else 'Dcheck'; offset=i if i<600 else i-600; N=len(qs[part]); arm=V.ARMS[offset//N]; pos=offset%N
        assert row['query_id']==qs[part][pos]['query_id'] and row['arm']==arm and row['partition']==part
        values=V.arrays(logits[part][arm][:,:,0],logits[part][arm][:,:,1])
        for key in ('log_loss','Brier'): assert float(row[key])==float(values[key][pos].mean())
    logs=[json.loads(x) for x in (D/'training_log.jsonl').read_text().splitlines()]
    passes=[x for x in logs if x['stage']=='COMPLETE_T_PASS']; held=[x for x in logs if x['stage']=='FIXED_DFIT_NOISY_LOGLOSS']
    assert len(passes)==10 and len(held)==11 and [x['epoch'] for x in passes]==list(range(1,11))
    provenance=U.read(D/'provenance.json'); assert provenance['identical_architecture_and_initialization'] is True
    figures=[]
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.2)); colors={'c_only':'#2563eb','raw_EEG':'#dc2626','shuffled_raw_train':'#059669'}
    for arm in V.ARMS:
        axes[0].plot([x['epoch'] for x in passes],[x['train_log_loss'][arm] for x in passes],color=colors[arm],linestyle='--')
        axes[0].plot([x['epoch'] for x in held],[x['metrics'][arm]['log_loss'] for x in held],color=colors[arm],label=arm+' Dfit')
        by=replay['classification']['Dcheck'][arm]['by_time']; axes[1].plot(V.TIMES,[by[str(t)]['log_loss'] for t in V.TIMES],'-o',color=colors[arm],label=arm)
        bins=replay['classification']['Dcheck'][arm]['pooled']['calibration_bins']
        axes[2].plot([x['probability'] for x in bins],[x['observed_positive'] for x in bins],'-o',color=colors[arm],label=arm)
    axes[0].set(xlabel='Full OOF-T passes',ylabel='Balanced log loss, nats',title='Dashed: train; solid: Dfit')
    axes[1].axhline(np.log(2),color='gray',linestyle=':'); axes[1].set(xlabel='Registered noise time',ylabel='Dcheck log loss',title='All 400 queries')
    axes[2].plot([0,1],[0,1],color='gray',linestyle=':'); axes[2].set(xlabel='Predicted probability',ylabel='Observed joint frequency',title='Dcheck reliability, all four times')
    for ax in axes: ax.legend(fontsize=7)
    fig.tight_layout(); chart=args.output/'actual_loss_and_calibration.png'; fig.savefig(chart,dpi=150); plt.close(fig); figures.append(U.ref(chart))
    assert not torch.cuda.is_initialized()
    result=dict(status='COMPLETE_D13_ALL_METRICS400_LOGITS_AND_FIXED_GRADIENTS_CPU_AUDIT',source=U.ref(__file__),completion=U.ref(args.completion),
        all_readout_and_bootstrap_recomputed_exactly=True,per_query_CSV_identity_and_proper_losses_verified=1800,
        all400x4x2x3_classification_logits_CPU_replayed=True,maximum_CPU_vs_GPU_logit_absolute_error=max_error,
        CPU_vs_GPU_pooled_logloss_absolute_difference=losses,gradient_queries_per_domain=40,gradient_times=4,gradient_heads=3,
        gradient_selection='first registered stimulus of each of40 concepts, all four times, all3 heads, both domains',
        CPU_vs_GPU_gradient_maximum_relative_L2=gradient_error,all_array_bindings=U.ref(args.output/'all_array_bindings.json'),
        all10_completed_training_passes=passes,all11_fixed_Dfit_diagnostics=held,figures=figures,
        diagnostic_only=True,new_images=0,actual_reconstruction_gain=False,GPU_used=False,seconds=time.monotonic()-start)
    U.atomic(args.output/'summary.json',result)
    print(json.dumps({k:result[k] for k in ('status','maximum_CPU_vs_GPU_logit_absolute_error','CPU_vs_GPU_gradient_maximum_relative_L2','seconds')}))


if __name__=='__main__': main()
