"""D23 independent native-weight, fold, noise-metric and all-condition replay."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import psutil
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_repeat_noise_subspace_20261027_v1 as V

def main():
    p=argparse.ArgumentParser()
    for name in ('completion','native-registration','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(2)
    process=psutil.Process();rss_start=process.memory_info().rss;peak_extra=0
    def check():
        nonlocal peak_extra
        peak_extra=max(peak_extra,process.memory_info().rss-rss_start)
        assert peak_extra<10<<30 and psutil.virtual_memory().available>10<<30 and time.monotonic()-start<270
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(C.read(a.native_registration))
    from brainhive_native_models_v1 import classes
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from fit_BrainHIVE_repeat_noise_subspace_GPU_20261028_v1 import metrics
    done=U.read(a.completion);s=U.read(U.checked(done['spec']));base=U.read(U.checked(done['scientific_spec']));profile=U.read(U.checked(done['profile']))
    for ref in s['source_pins']:U.checked(ref)
    bundle=torch.load(U.checked(done['model']),weights_only=False,map_location='cpu',mmap=True)['models'];lock=U.read(U.checked(done['condition_lock']))
    saved_oof=np.load(U.checked(done['OOF_T_ordinary']),mmap_mode='r');saved_native=np.load(U.checked(done['OOF_T_native']),mmap_mode='r');saved_df=np.load(U.checked(done['Dfit_conditions']),mmap_mode='r');saved_dc=np.load(U.checked(lock['conditions']),mmap_mode='r')
    raw=torch.from_numpy(np.array(np.load(U.checked(base['T_mean4']),mmap_mode='r'),copy=True).reshape(13230,4250));prepared=U.read(U.checked(s['prepared_EEG']));mapping=U.read(U.checked(prepared['mapping']))
    dev=torch.from_numpy(np.array(np.load(U.checked(prepared['Dfit']),mmap_mode='r')[0],copy=True).reshape(820,4250));dc=torch.from_numpy(np.array(np.load(U.checked(prepared['Dcheck']),mmap_mode='r'),copy=True).reshape(400,4250))
    contract=U.read(U.checked(base['native_input_contract']));full=next(x for x in contract['subjects'][0]['sources'] if x['role']=='full');rows=[x['array_row'] for x in full['groups']['calibration_only_Dfit']]
    bank=torch.load(U.checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True);truth=bank[rows].clone();del bank
    ordinary=BoundedConditionAdapter().eval().requires_grad_(False);ordinary.load_state_dict(torch.load(U.checked(s['ordinary_checkpoint']),weights_only=False,map_location='cpu')['model'],strict=True)
    BrainMLP=classes()['BrainMLP'];maxima=dict(fold_weight_abs=0.,ordinary_condition_relative_L2=0.,native_condition_relative_L2=0.);proofs=[];coverage=np.zeros((13230,3),np.int8);new_df={}
    hidden=np.linspace(0,1023,32).round().astype(int);columns=np.linspace(0,4249,16).round().astype(int)
    trials=np.load(U.checked(base['trials']),mmap_mode='r');perm=np.load(U.checked(profile['noise_coordinate_permutation']))
    for role,item in base['roles'].items():
        check()
        geos={name:torch.load(U.checked(ref),weights_only=True,map_location='cpu',mmap=True) for name,ref in profile['geometries'][role].items()};gtrue=geos['repeat'];gperm=geos['permuted'];old=torch.load(U.checked(item['checkpoint']),weights_only=True,map_location='cpu')['model']
        # Independently reconstruct16 complete4250dim covariance columns from
        # role-FIT physical repeats, not from exported GPU covariance entries.
        summed=np.zeros((4250,16),np.float64);trace=0.;ix=np.asarray(item['fit_positions'])
        for lo in range(0,len(ix),256):
            check()
            t=np.array(trials[ix[lo:lo+256]],dtype=np.float64).reshape(-1,4,4250);r=(t-t.mean(1,keepdims=True)).reshape(-1,4250);summed+=r.T@r[:,columns];trace+=float(np.square(r).sum())
        expected=.9*summed/(12*len(ix));expected[columns,np.arange(16)]+=.1*trace/(12*len(ix)*4250)
        error=float(np.linalg.norm(expected-gtrue['noise'][:,columns].numpy())/np.linalg.norm(expected));assert error<1e-10
        assert torch.equal(gperm['noise'][:,columns],gtrue['noise'][perm][:,perm[columns]])
        for ai,arm in enumerate(V.ARMS):
            check();g=gperm if ai==1 else gtrue;state=bundle[role][arm]
            for key,value in old.items():
                if key!='proj_in.weight':assert torch.equal(value,state[key])
            q=torch.load(U.checked(done['Q_states'][role+'_'+arm]),weights_only=True,map_location='cpu');fold=V.folded_weight(q,g);fe=float((fold-state['proj_in.weight']).abs().max());assert fe<1e-7;maxima['fold_weight_abs']=max(maxima['fold_weight_abs'],fe)
            sampled=state['proj_in.weight'][hidden].double();original=old['proj_in.weight'][hidden].double();actual=sampled@g['noise']@sampled.T;reference=original@g['noise']@original.T;me=float((actual-reference).norm()/reference.norm())
            if ai!=2:assert me<2e-5
            model=BrainMLP(4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);model.load_state_dict(state,strict=True);w=model.block.layers[0].norm.weight;b=model.block.layers[0].norm.bias
            def infer(values):
                native=[];corrected=[]
                with torch.no_grad():
                    for e in values.split(256):
                        c=model(e);native.append(c);corrected.append(ordinary(c,w.expand(len(c),-1),b.expand(len(c),-1))[0])
                return torch.cat(native),torch.cat(corrected)
            def verify(actual,expected,key):
                rel=np.linalg.norm(actual.numpy()-expected,axis=1)/np.linalg.norm(expected,axis=1);assert rel.max()<3e-5;maxima[key]=max(maxima[key],float(rel.max()))
            if role=='full':
                native,pred=infer(dev);verify(pred,saved_df[:,ai],'ordinary_condition_relative_L2');new_df[arm]=dict(native=metrics(native,truth),ordinary=metrics(pred,truth))
                _,pred=infer(dc);verify(pred,saved_dc[:,ai],'ordinary_condition_relative_L2')
            else:
                held=item['held_positions'];native,pred=infer(raw[held]);verify(native,saved_native[held,ai],'native_condition_relative_L2');verify(pred,saved_oof[held,ai],'ordinary_condition_relative_L2');coverage[held,ai]+=1
            proofs.append(dict(role=role,arm=arm,full_folded_first_weight_abs_error=fe,selected32_hidden_noise_Gram_relative_error=me,selected16_noise_columns_from_all_FIT_repeats_relative_error=error))
        print(json.dumps(dict(role=role,CPU_seconds=time.monotonic()-start)),flush=True)
    assert np.all(coverage==1);saved_metrics=U.read(U.checked(done['Dfit_metrics']))
    for arm in V.ARMS:
        for space in new_df[arm]:
            for key,value in new_df[arm][space].items():assert abs(value-saved_metrics[arm][space][key])<1e-5
    events=[json.loads(x) for x in (a.completion.parent/'training_log.jsonl').read_text().splitlines()];passes=[x for x in events if x['stage']=='COMPLETE_ROLE_T_PASS'];assert len(passes)==30
    for ri,(role,item) in enumerate(base['roles'].items()):
        seq=[x for x in passes if x['role']==role];assert [x['epoch'] for x in seq]==list(range(1,11))
        gen=torch.Generator().manual_seed(1729+ri)
        for x in seq:
            order=torch.randperm(len(item['fit_positions']),generator=gen)
            assert hashlib.sha256(order.numpy().tobytes()).hexdigest()==x['order_sha256']
            assert x['updates_per_model']==x['epoch']*item['updates_per_epoch'] and x['exposures_per_model']==x['epoch']*len(item['fit_positions']) and x['coverage_min']==x['coverage_max']==1
    assert lock['query_ids']==[q['query_id'] for q in mapping['Dcheck_queries']] and lock['Dcheck_target_F_or_images_used'] is False
    for ref in s['source_pins']:U.checked(ref)
    assert not torch.cuda.is_initialized()
    value=dict(status='D23_ALL43350_NATIVE_AND_ORDINARY_CONDITIONS_AND9_NATIVE_WEIGHTS_CPU_VERIFIED',source=U.ref(__file__),completion=U.ref(a.completion),maxima=maxima,proofs=proofs,
        all9_first_weights_reconstructed_from_Q_on_CPU=True,all_other_native_weights_bitwise_unchanged=True,all39690_OOF_native_and_ordinary_conditions_replayed=True,all2460_Dfit_and1200_Dcheck_conditions_replayed=True,
        noise_covariance_columns_checked=columns.tolist(),hidden_noise_Gram_rows_checked=hidden.tolist(),all4250_noise_covariance_columns_CPU_recomputed=False,all1024_hidden_Gram_rows_CPU_recomputed=False,
        all30_complete_T_pass_logs_verified=True,all_sources_unchanged=True,seconds=time.monotonic()-start,peak_extra_CPU_bytes=peak_extra,new_GPU_jobs=0,new_fit=0,new_images=0,formal_D_resumed=False)
    U.atomic(a.output/'summary.json',value);print(json.dumps({k:value[k] for k in ('status','seconds','maxima')}))

if __name__=='__main__':main()
