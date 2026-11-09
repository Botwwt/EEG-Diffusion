"""Independent full-fold/epoch/held-output audit and train-only OOF assembly."""
import argparse,gc,json,math,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import brainhive_exposed_OOF_assets_contract_CPU_20261101_v1 as G

def setup(path):
    s=G.read(path)
    for p in s['source_pins']:G.checked(p)
    tasks,images=G.load_scope(s);G.assert_input_aliases(s)
    admission=G.read(Path(s['output_root'])/'training_admission.json')
    assert admission['status']=='ALL20_TRAIN_ONLY_ASSET_FITS_REVIEWED_AFTER_ACTUAL_PROFILE' and admission['protocol']==G.ref(path)
    for p in admission['execution_sources']:G.checked(p)
    sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']));sys.path.insert(0,str(Path(path).parent))
    import torch,numpy as np
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    return s,tasks,images,torch,np

def manual(x,w,torch):
    import torch.nn.functional as F
    z=F.linear(x.double().reshape(len(x),4250),w['proj_in.weight'],w['proj_in.bias'])
    z=z+F.linear(F.gelu(F.linear(z,w['block.layers.0.ff.w1.weight'])),w['block.layers.0.ff.w2.weight'])
    return F.layer_norm(z,(1024,),w['block.layers.0.norm.weight'],w['block.layers.0.norm.bias'],1e-6)

def validate_final_certificate(done,protocol,task_id):
    assert done['status']=='COMPLETE_NATIVE_FULL1654_TRAIN_ONLY_FOLD_AND_HELD_EXPORT'
    assert done['protocol']==protocol and done['task_id']==task_id and done['updates']==225 and done['epochs']==25
    assert done['temporary_updates']==0 and done['learned_weights_saved'] and not done['official_test_payloads_loaded']
    assert done['final_model'] is not None and done['held_queries']==8270

def fold(path,task_id):
    start=time.monotonic();s,tasks,images,torch,np=setup(path);assert task_id in tasks
    task=tasks[task_id];root=Path(s['output_root'])/'tasks'/task_id;done=G.read(root/'completion.json')
    validate_final_certificate(done,G.ref(path),task_id)
    import brainhive_official_encoder_training_v1 as T
    import brainhive_native_models_v1 as N
    registration=G.ref(root/'registration.json')['sha256'];producer=G.ref(Path(path).parent/'run_HIVE_exposed_OOF_assets_GPU_20261101_v1.py')['sha256']
    commits,pending=T.scan_committed_epochs(root,registration,G.ref(path)['sha256'],producer_source_sha256=producer,subject=task['subject'],role='train_only_official_oof'+str(task['held_fold']),updates_per_epoch=9,total_epochs=25,full_epoch_rows=8270)
    assert len(commits)==25 and not pending
    final=torch.load(G.checked(done['final_model']),weights_only=True,map_location='cpu')
    assert final['seed']==1729 and final['updates']==225 and final['epochs']==25
    assert final['registration_sha256']==registration and final['registration_core_sha256']==G.ref(path)['sha256']
    model=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval();model.load_state_dict(final['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==6452224
    scale=torch.nn.Parameter(final['logit_scale'].clone());opt,_=T.make_optimizer(torch,model,scale,225)
    parameters=[g['params'] for g in opt.param_groups];shuffle=torch.Generator(device='cpu').manual_seed(1730)
    state_fields=set(T.RESUME_FIELDS);maximum=0.;epoch_checks=[]
    for epoch,c in enumerate(commits,1):
        v=torch.load(Path(c['directory'])/'resume.pt',weights_only=True,map_location='cpu');assert set(v)==state_fields
        assert v['epoch']==epoch and v['updates']==epoch*9
        assert v['registration_sha256']==registration and v['registration_core_sha256']==G.ref(path)['sha256'] and v['producer_source_sha256']==producer
        order=torch.randperm(8270,generator=shuffle);assert len(order.unique())==8270 and torch.equal(shuffle.get_state(),v['shuffle_rng'])
        assert len(v['optimizer']['state'])==7 and len(v['optimizer']['param_groups'])==2
        for group,ps,decay in zip(v['optimizer']['param_groups'],parameters,[.05,0.]):
            assert len(group['params'])==len(ps)
            assert group['weight_decay']==decay and tuple(group['betas'])==(.9,.995) and group['eps']==1e-8
            expected_lr=5e-4*.5*(1+math.cos(math.pi*(epoch*9)/225))
            assert abs(group['lr']-expected_lr)<1e-16
            for index,p in zip(group['params'],ps):
                a=v['optimizer']['state'][index];assert set(a)=={'step','exp_avg','exp_avg_sq'} and float(a['step'])==epoch*9
                assert all(a[k].shape==p.shape and a[k].dtype==p.dtype and torch.isfinite(a[k]).all() for k in ('exp_avg','exp_avg_sq'))
                assert (a['exp_avg_sq']>=0).all()
        assert v['scheduler']['last_epoch']==epoch*9 and v['scheduler']['_step_count']==epoch*9+1
        assert len(v['cuda_rng'])==1 and v['torch_rng'].dtype==torch.uint8
        if epoch==25:
            assert set(v['model'])==set(final['model']) and all(torch.equal(v['model'][n],final['model'][n]) for n in final['model'])
            assert torch.equal(v['logit_scale'],final['logit_scale'])
        epoch_checks.append(dict(epoch=epoch,updates=epoch*9,all7_Adam_states_checked=True,shuffle_full_permutation_and_RNG_verified=True));del v;gc.collect()
        assert time.monotonic()-start<1700
    observed=np.load(G.checked(done['raw_conditions']),allow_pickle=False,mmap_mode='r');rows=np.load(G.checked(done['export_array_rows']),allow_pickle=False)
    assert observed.shape==(8270,1024) and observed.dtype==np.float32 and rows.tolist()==task['exported_OOF_array_rows']
    raw=torch.load(G.checked(s['subjects'][task['subject']]),weights_only=True,map_location='cpu',mmap=True)
    channels=G.read(G.checked(s['bindings']))['selected_indices'];w={k:v.double() for k,v in final['model'].items()}
    with torch.no_grad():
        for lo in range(0,8270,256):
            ids=torch.tensor(rows[lo:lo+256].copy());x=raw[ids[:,None],torch.tensor(channels)[None,:],:]
            expected=manual(x,w,torch).numpy();value=observed[lo:lo+len(ids)].astype(float)
            maximum=max(maximum,float(np.max(np.abs(value-expected))))
            assert np.allclose(value,expected,atol=1e-5,rtol=5e-6)
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_INDEPENDENT_FULL25_EPOCH_AND8270_HELD_NATIVE_OOF_AUDIT',protocol=G.ref(path),task_id=task_id,
        final_model=done['final_model'],raw_conditions=done['raw_conditions'],export_array_rows=done['export_array_rows'],epochs=epoch_checks,
        all25_epoch_Adam_counters_moments_and_shuffle_checked=True,all8270_manual_CPU_FP64_forwards_checked=True,manual_CPU_GPU_max_absolute_error=maximum,
        held_concepts_excluded_from_fit=True,private_full_task_last_epoch_only=True,not_all_225_gradient_transitions_replayed=True,
        official_test_payloads_opened=False,CUDA_initialized=False,new_updates=0,seconds=time.monotonic()-start)
    G.atomic(root/'independent_CPU_audit.json',result);print(json.dumps(dict(status=result['status'],task_id=task_id,maximum_error=maximum,seconds=result['seconds'])))

def aggregate(path):
    start=time.monotonic();s,tasks,images,torch,np=setup(path);root=Path(s['output_root']);G.resources(s)
    out=root/'assembled';out.mkdir(exist_ok=False);shape=(165400,1024)
    bank=np.lib.format.open_memmap(out/'raw_conditions.npy',mode='w+',shape=shape,dtype=np.float32)
    index=np.empty(165400,dtype=np.int16);covered=np.zeros(165400,dtype=bool)
    weights=np.empty((20,1024),dtype=np.float32);bias=np.empty_like(weights);proof=[]
    for number,task_id in enumerate(sorted(tasks)):
        task=tasks[task_id];folder=root/'tasks'/task_id;a=G.read(folder/'independent_CPU_audit.json')
        assert a['status']=='PASS_INDEPENDENT_FULL25_EPOCH_AND8270_HELD_NATIVE_OOF_AUDIT' and a['protocol']==G.ref(path)
        assert a['task_id']==task_id and a['held_concepts_excluded_from_fit']
        rows=np.load(G.checked(a['export_array_rows']),allow_pickle=False);assert rows.tolist()==task['exported_OOF_array_rows']
        source=np.load(G.checked(a['raw_conditions']),allow_pickle=False,mmap_mode='r')
        subject_index=G.SUBJECTS.index(task['subject']);where=subject_index*16540+rows
        assert not covered[where].any() and np.isfinite(source).all();bank[where]=source;covered[where]=True;index[where]=number
        saved=torch.load(G.checked(a['final_model']),weights_only=True,map_location='cpu')['model']
        weights[number]=saved['block.layers.0.norm.weight'].numpy();bias[number]=saved['block.layers.0.norm.bias'].numpy()
        proof.append(dict(task_id=task_id,audit=G.ref(folder/'independent_CPU_audit.json'),model=a['final_model'],conditions=a['raw_conditions'],rows=a['export_array_rows']))
        del saved;gc.collect()
    assert covered.all() and np.isfinite(bank).all() and np.min(np.abs(weights))>1e-8
    bank.flush();del bank
    for name,value in [('affine_weight.npy',weights),('affine_bias.npy',bias),('affine_id.npy',index)]:
        with (out/name).open('xb') as f:np.save(f,value,allow_pickle=False)
    actual=np.load(out/'raw_conditions.npy',allow_pickle=False,mmap_mode='r')
    for number,item in enumerate(proof):
        t=tasks[item['task_id']];rows=np.load(G.checked(item['rows']),allow_pickle=False);source=np.load(G.checked(item['conditions']),allow_pickle=False,mmap_mode='r')
        where=G.SUBJECTS.index(t['subject'])*16540+rows
        assert np.array_equal(actual[where],source) and (index[where]==number).all()
    G.atomic(out/'image_identity.json',[dict(array_row=i,stimulus_id=images[i]['stimulus_id'],concept_id=images[i]['concept_id']) for i in range(16540)])
    result=dict(status='COMPLETE_ACTUAL_TRAIN_ONLY_FULL1654_20FOLD165400_OOF_ASSETS',protocol=G.ref(path),subjects=10,concepts=1654,images=16540,queries=165400,raw_shape=list(shape),
        raw_conditions=G.ref(out/'raw_conditions.npy'),affine_weight=G.ref(out/'affine_weight.npy'),affine_bias=G.ref(out/'affine_bias.npy'),affine_id=G.ref(out/'affine_id.npy'),image_identity=G.ref(out/'image_identity.json'),
        query_order='subject sub-01..sub-10, then exact global array_row0..16539',target_source=s['fused_target_for_future_inputs'],all20_folds_and_all165400_join_values_verified=True,folds=proof,
        training_seed=1729,official200_previously_exposed=True,independent_H=False,official_test_payloads_opened=False,final_component_or_benchmark_admitted=False,new_images=0,CUDA_initialized=False,seconds=time.monotonic()-start)
    G.atomic(root/'input_bank_completion.json',result);print(json.dumps(dict(status=result['status'],queries=165400,seconds=result['seconds'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['fold','aggregate'],required=True);p.add_argument('--task');a=p.parse_args()
    fold(a.protocol,a.task) if a.mode=='fold' else aggregate(a.protocol)
