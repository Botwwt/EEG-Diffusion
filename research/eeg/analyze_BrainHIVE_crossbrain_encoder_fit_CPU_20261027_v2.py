"""D17 full8200x3 final native encoder replay and complete fixed-fit readout."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_crossbrain_supervision_20261027_v1 as V
import fit_BrainHIVE_crossbrain_supervision_GPU_20261027_v1 as FIT
import brainhive_cross_encoder_20261027_v1 as OLD

def main():
    p=argparse.ArgumentParser()
    for key in ('completion','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();start=time.monotonic();torch.set_num_threads(2);assert not torch.cuda.is_initialized() and not a.output.exists();a.output.mkdir(parents=True)
    done=U.read(a.completion);assert done['status']=='D17_ALL10_THREE_MATCHED_NATIVE_ENCODERS_FULL_T10_AND_ALL8200_DFIT_COMPLETE'
    spec=U.read(U.checked(done['spec']));prepared=U.read(U.checked(spec['prepared_EEG']));mapping=U.read(U.checked(prepared['mapping']))
    saved=torch.load(U.checked(done['model']),weights_only=True,map_location='cpu',mmap=True)
    assert saved['epochs_per_model']==10 and saved['updates_per_model']==140 and saved['exposures_per_model']==132300
    contract=U.read(U.checked(spec['native_input_contract']));data=np.load(U.checked(prepared['Dfit']),allow_pickle=False,mmap_mode='r');dc=np.load(U.checked(prepared['Dcheck']),allow_pickle=False)
    target=np.load(U.checked(done['targets']),allow_pickle=False);condition=np.load(U.checked(done['conditions']),allow_pickle=False)
    lock=U.read(U.checked(done['condition_lock']));bank=np.load(U.checked(lock['conditions']),allow_pickle=False)
    assert condition.shape==(8200,3,1024) and bank.shape==(400,3,1024) and np.isfinite(condition).all() and np.isfinite(bank).all()
    dmap=U.read(U.checked(done['Dfit_queries']));assert mapping['Dfit_queries']==dmap['queries']
    assert lock['query_ids']==[q['query_id'] for q in mapping['Dcheck_queries']] and lock['normal_inference_other_subject_EEG'] is False
    fbank=torch.load(U.checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
    assert np.array_equal(fbank[[q['array_row'] for q in dmap['queries']]].numpy(),target)
    from brainhive_native_models_v1 import classes
    cls=classes()['BrainMLP'];original=[];replays=[];dc_replays=[];parameter_changes={};steps=[]
    for si,subject in enumerate(contract['subjects']):
        name=subject['subject'];assert name==mapping['subjects'][si]
        source=next(x for x in subject['sources'] if x['role']=='full');cache=torch.load(U.checked(dict(path=source['condition_cache_path'],sha256=source['condition_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
        rows=source['groups']['calibration_only_Dfit'];original.append(cache['condition'][[q['cache_row'] for q in rows]].clone().numpy())
        assert [q['array_row'] for q in rows]==[q['array_row'] for q in dmap['queries'][si*820:(si+1)*820]]
        x=torch.from_numpy(np.array(data[si],copy=True));old_state=torch.load(U.checked(dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256'])),weights_only=True,map_location='cpu',mmap=True)['model']
        for j,arm in enumerate(V.ARMS):
            model=cls(4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);model.load_state_dict(saved['models'][name][arm],strict=True)
            with torch.no_grad():actual=torch.cat([model(batch).float() for batch in x.split(256)]).numpy()
            reference=condition[si*820:(si+1)*820,j];relative=np.linalg.norm(actual-reference,axis=-1)/np.linalg.norm(reference,axis=-1)
            cosine=(actual*reference).sum(-1)/(np.linalg.norm(actual,axis=-1)*np.linalg.norm(reference,axis=-1))
            assert relative.max()<3e-5 and cosine.min()>.999999
            replays.append(dict(subject=name,arm=arm,queries=820,maximum_relative_L2=float(relative.max()),minimum_cosine=float(cosine.min())))
            opt=saved['optimizers'][name][arm]['optimizer'];counts=[int(v['step']) for v in opt['state'].values()];assert counts and set(counts)=={140};steps.append(dict(subject=name,arm=arm,optimizer_parameter_states=len(counts),all_steps=140))
            delta=sum(float((saved['models'][name][arm][key]-old_state[key]).double().square().sum()) for key in old_state)
            assert delta>0;parameter_changes[name+'/'+arm]=delta**.5
            if si==0:
                with torch.no_grad():actual_dc=torch.cat([model(batch).float() for batch in torch.from_numpy(dc).split(256)]).numpy()
                rel=np.linalg.norm(actual_dc-bank[:,j],axis=-1)/np.linalg.norm(bank[:,j],axis=-1);assert rel.max()<3e-5
                dc_replays.append(dict(arm=arm,queries=400,maximum_relative_L2=float(rel.max())))
        del model,x,cache,old_state
    parent=np.concatenate(original);truth=torch.from_numpy(target);values={arm:FIT.feature_metrics(torch.from_numpy(condition[:,j]),truth) for j,arm in enumerate(V.ARMS)}
    # Complete groups are contiguous within subject in the bound Dfit query list.
    for i in range(0,8200,10):assert len({q['concept_id'] for q in dmap['queries'][i:i+10]})==1
    values['original_parent']=FIT.feature_metrics(torch.from_numpy(parent),truth)
    means={arm:{k:float(v.mean()) for k,v in x.items()} for arm,x in values.items()}
    former=U.read(U.checked(done['Dfit_means_by_subject']));maxmean=0.
    for si,name in enumerate(mapping['subjects']):
        for arm in V.ARMS:
            for key in values[arm]:maxmean=max(maxmean,abs(float(values[arm][key][si*820:(si+1)*820].mean())-former[name][arm][key]))
    assert maxmean<2e-6
    subjects=[q['subject_id'] for q in dmap['queries']];concepts=[q['concept_id'] for q in dmap['queries']]
    contrasts={arm:OLD.paired_difference(values['crossbrain_soft'],value,subjects,concepts) for arm,value in values.items() if arm!='crossbrain_soft'}
    out=a.output/'Dfit_per_query.csv';fields=list(dmap['queries'][0])+['arm']+list(next(iter(values.values())))
    with out.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i,q in enumerate(dmap['queries']):
            for arm,x in values.items():writer.writerow(q|dict(arm=arm,**{k:float(v[i]) for k,v in x.items()}))
    logs=[json.loads(x) for x in (Path(a.completion).parent/'training_log.jsonl').read_text().splitlines()]
    train=[x for x in logs if x['stage']=='COMPLETE_T_PASS'];held=[x for x in logs if x['stage']=='DFIT820_LOG_ONLY'];assert len(train)==100 and len(held)==110
    for name in mapping['subjects']:
        rows=[x for x in train if x['subject']==name];assert [x['epoch'] for x in rows]==list(range(1,11))
        assert all(x['full_T_coverage_min']==x['full_T_coverage_max']==1 and x['updates_per_model']==x['epoch']*14 and x['exposures_per_model']==x['epoch']*13230 for x in rows)
    plot=None
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(14,3.8))
        for arm in V.ARMS:
            axes[0].plot(range(1,11),[np.mean([x['means'][arm]['NCE'] for x in train if x['epoch']==e]) for e in range(1,11)],label=arm)
            axes[1].plot(range(11),[np.mean([x['values'][arm]['within10_log_loss'] for x in held if x['epoch']==e]) for e in range(11)],label=arm)
            axes[2].plot(range(11),[np.mean([x['values'][arm]['target_cosine'] for x in held if x['epoch']==e]) for e in range(11)],label=arm)
        for ax,title in zip(axes,['All10 mean T InfoNCE','All8200 Dfit local10 CE (logs only)','All8200 Dfit F cosine (logs only)']):ax.set(title=title,xlabel='Fixed complete T fine-tune passes');ax.legend(fontsize=7)
        fig.tight_layout();path=a.output/'actual_T_and_Dfit_curves.png';fig.savefig(path,dpi=160);plt.close(fig);plot=U.ref(path)
    except ImportError:pass
    for ref in spec['source_pins']:U.checked(ref)
    assert not torch.cuda.is_initialized();result=dict(status='D17_ALL24600_DFIT_AND1200_DCHECK_NATIVE_ENCODER_CPU_FORWARDS_AND_ALL100_T_PASSES_VERIFIED',source=U.ref(__file__),completion=U.ref(a.completion),
        all30_native_encoders_loaded=True,all24600_Dfit_forward_replay=replays,all1200_Dcheck_forward_replay=dc_replays,
        all8200_target_identities_and_values_exact=True,full_Dfit_means=means,crossbrain_minus_references=contrasts,contrast_sign='candidate minus reference for every metric; negative improves errors',
        original_GPU_mean_max_absolute_error=maxmean,per_query=U.ref(out),per_query_rows=32800,optimizer_steps=steps,parameter_L2_change_from_original=parameter_changes,
        all100_training_passes_full_coverage_verified=True,all110_Dfit_logs_retained=True,final_model_selection='fixed10, no Dfit/epoch/arm selection',
        candidate_feature_gate_applied=False,feature_metrics_not_image_quality=True,actual_plot=plot,new_GPU_jobs=0,new_parameter_updates=0,new_images=0,
        seconds=time.monotonic()-start,no_models_or_conditions_modified=True,independent_confirmation=False,SOTA_established=False)
    U.atomic(a.output/'summary.json',result);print(json.dumps(dict(status=result['status'],seconds=result['seconds'],means=means,max_mean_error=maxmean)))

if __name__=='__main__':main()
