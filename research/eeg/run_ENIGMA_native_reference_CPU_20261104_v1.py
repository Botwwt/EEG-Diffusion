"""Independent full encoder-OOF replay and fixed full400 image evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,hashlib,io,json,resource,sys,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import enigma_native_reference_contract_20261104_v1 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=('conditions','audit','score'),required=True)
a=ap.parse_args();started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(a.policy);K=c['K'];q=c['q'];out=c['out'];C.resources(c)
assert K.ref(__file__)==q['sources']['CPU']
def guard():
    assert time.monotonic()-started<850 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<c['parent']['maximum_extra_CPU_bytes']
prepared=K.read(out/'prepared.json');assert prepared['policy']==K.ref(a.policy)
conditions=np.load(K.checked(prepared['conditions']),allow_pickle=False);assert conditions.shape==(400,7,1024)
if a.mode=='conditions':
    assert not (out/'conditions_CPU_audit.json').exists()
    sys.path[:0]=q['existing_runtime_paths']
    import torch
    from enigma_native_reference_math_20261104_v1 import author_namespace
    torch.set_num_threads(2)
    scope=author_namespace(K.checked(q['original_models']).read_text(),K.checked(q['original_training']).read_text())
    raw=torch.load(K.checked(q['EEG']),map_location='cpu',weights_only=True,mmap=True)
    bank=np.load(K.checked(prepared['OOF_conditions']),allow_pickle=False)
    D=np.load(K.checked(prepared['D_conditions']),allow_pickle=False)
    positions={int(v):i for i,v in enumerate(c['T'])};errors={};certificates=[];updates=exposures=0
    for role in C.ROLES:
        fit,held=c['roles'][role];batches=(len(fit)+511)//512;logs=[]
        for end in (50,100,150):
            event=K.read(out/'fits'/role/('segment_'+str(end)+'_completion.json'))
            path=K.checked(event['checkpoint'])
            assert path.is_relative_to(Path(q['project_root'])) and path.parent.name==role and path.name=='epoch_'+str(end)+'.pt'
            state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert state['epoch']==end and state['role']==role and state['updates']==end*batches and state['exposures']==end*len(fit)
            allowed=[K.ref(a.policy)]+([q['resume_origin_policy']] if 'resume_origin_policy' in q else [])
            assert state['fit_rows_SHA256']==hashlib.sha256(fit.tobytes()).hexdigest() and state['policy'] in allowed
            assert len(state['optimizer']['state'])>0 and state['CPU_rng'].dtype==state['CUDA_rng'].dtype==torch.uint8
            assert state['scheduler']['last_epoch']==end*batches and state['scheduler']['total_steps']==150*batches
            segment=[json.loads(s) for s in K.checked(state['epoch_log']).read_text().splitlines()]
            assert [r['epoch'] for r in segment]==list(range(end-49,end+1));logs+=segment
            for row in segment:
                epoch=row['epoch'];order=np.random.default_rng(1729+epoch-1).permutation(len(fit))
                assert row['sample_order_SHA256']==hashlib.sha256(order.tobytes()).hexdigest()
                assert row['updates']==epoch*batches and row['exposures']==epoch*len(fit)
                assert row['coverage_min']==row['coverage_max']==1 and not row['heldout_concepts_or_D_used']
            certificates.append(dict(role=role,epoch=end,updates=state['updates'],checkpoint=event['checkpoint'],epoch_log=state['epoch_log']))
            if end<150:del state
        assert len(logs)==150
        net=scope['ENIGMA'](63,250,['sub-01']).eval().requires_grad_(False);net.load_state_dict(state['model'],strict=True)
        ids=c['D'] if role=='full' else held
        reference=D if role=='full' else bank[[positions[int(v)] for v in ids]]
        err=[]
        with torch.inference_mode():
            for lo in range(0,len(ids),32):
                guard();rows=ids[lo:lo+32];pred=net(raw[rows],['sub-01']*len(rows)).float().numpy()
                err.extend(np.max(abs(pred-reference[lo:lo+len(rows)]),axis=1).tolist())
        assert max(err)<8e-5;errors[role]=err
        updates+=state['updates'];exposures+=state['exposures'];del state,net
    assert updates==7800 and exposures==3969000 and sum(len(x) for x in errors.values())==14880
    idx={int(v):i for i,v in enumerate(c['D'])};assert np.array_equal(conditions[:,6],D[[idx[r['array_row']] for r in c['queries']]])
    assert np.array_equal(conditions[:,:6],np.load(K.checked(q['baseline_conditions']),allow_pickle=False))
    K.write(out/'conditions_CPU_audit.json',dict(status='PASS_NATIVE63_COMPLETE_OOF_AND_D_CPU_REPLAY',
        errors_per_query=errors,maximum_error=max(max(x) for x in errors.values()),tolerance=8e-5,
        all450_epoch_orders_and_coverage_verified=True,all9_checkpoint_optimizer_scheduler_RNG_checked=True,
        new_models=3,new_updates=updates,new_exposures=exposures,certificates=certificates,
        encoder_concept_exclusion=True,whole_pipeline_preprocessing_OOF_NOT_established=True))
else:
    queries=c['queries'];records=C.records(c,400)
    assert all([r['query_index'] for r in records[arm]]==list(range(400)) for arm in C.ARMS)
    ordered=[records[arm][i] for i in range(400) for arm in C.ARMS]
    if a.mode=='audit':
        assert not (out/'image_audit400.json').exists();paths=set();inodes=set();dups={};constant=0
        for i in range(400):
            guard()
            for j,arm in enumerate(C.ARMS):
                r=records[arm][i];assert r['arm']==arm and all(r[k]==queries[i][k] for k in C.IDENTITY)
                path=K.checked(r['image']);stat=path.stat();key=(stat.st_dev,stat.st_ino)
                assert str(path) not in paths and key not in inodes;paths.add(str(path));inodes.add(key)
                pixels=np.asarray(Image.open(path).convert('RGB')).copy()
                assert pixels.shape==(512,512,3) and hashlib.sha256(pixels.tobytes()).hexdigest()==r['pixels_sha256']
                value=np.ascontiguousarray(conditions[i,j][None])
                assert hashlib.sha256(value.tobytes()).hexdigest()==r['condition_FP32_sha256']
                assert hashlib.sha256(value.astype(np.float16).tobytes()).hexdigest()==r['condition_FP16_sha256']
                assert r['noise']==records['direct'][i]['noise'] and len(r['noise']['initial'])==1 and len(r['noise']['ancestral'])==4
                if arm=='enigma_native63':
                    assert r['normal_inference_only_EEG'] and not r['query_image_or_concept_read_at_inference']
                    assert r['prepared']==K.ref(out/'prepared.json')
                dups.setdefault(r['pixels_sha256'],[]).append([arm,i]);constant+=int(np.max(pixels.astype(float).std(axis=(0,1)))<2.55)
        K.write(out/'ordered_images400.json',ordered)
        K.write(out/'image_audit400.json',dict(status='PASS_ALL_NATIVE63_REFERENCE_IMAGE_PROVENANCE',
            images=2800,ordered_records=K.ref(out/'ordered_images400.json'),all_conditions_noise_identity_pixels_verified=True,
            near_constant_count=constant,exact_duplicate_pixel_groups=[x for x in dups.values() if len(x)>1]))
    else:
        assert not (out/'readout400.json').exists()
        assert K.read(out/'image_audit400.json')['status']=='PASS_ALL_NATIVE63_REFERENCE_IMAGE_PROVENANCE'
        import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
        import score_low_level as L
        assert K.ref(S.__file__)==q['source_metrics'] and K.ref(L.__file__)==q['source_low_level']
        features=np.load(K.checked(K.read(out/'encode_completion.json')['features']),allow_pickle=False)
        assert features.shape==(400,7,768)
        target=np.load(K.checked(prepared['target_feature_ref']),allow_pickle=False)
        selfcheck=S.metric_arrays(target,target,queries,queries)
        assert np.all(selfcheck['DINO2AFC']==1) and np.all(selfcheck['gallery_top1']==1)
        baseline_metrics=np.load(K.checked(q['baseline_metrics']),allow_pickle=False)
        arrays={arm:{key:baseline_metrics[arm+'__'+key].copy() for key in C.METRICS} for arm in C.BASES}
        arrays['enigma_native63']=S.metric_arrays(features[:,6],target,queries,queries)
        for key in C.METRICS[5:]:arrays['enigma_native63'][key]=np.empty(400)
        metadata=K.read(K.checked(prepared['target_metadata_ref']))['Dcheck'];zips={}
        try:
            for i in range(400):
                guard();m=metadata[i];assert all(m[k]==queries[i][k] for k in C.IDENTITY)
                if m['image_zip'] not in zips:zips[m['image_zip']]=zipfile.ZipFile(m['image_zip'])
                data=zips[m['image_zip']].read(m['image_member']);assert hashlib.sha256(data).hexdigest()==m['image_sha256']
                trgb,ty=L.pixels(io.BytesIO(data));tm=L.moments(ty)
                rgb,y=L.pixels(K.checked(records['enigma_native63'][i]['image']))
                arrays['enigma_native63']['PixCorr_RGB'][i]=L.pixcorr(rgb,trgb)
                arrays['enigma_native63']['SSIM_Y'][i]=L.ssim_y(y,ty,tm)
                arrays['enigma_native63']['mean_RGB_absolute_error'][i]=float(abs(rgb.mean((0,1))-trgb.mean((0,1))).mean())
        finally:
            for z in zips.values():z.close()
        matrices=[]
        for j,arm in enumerate(C.ARMS):
            z=features[:,j].astype(np.float32);z=z/np.linalg.norm(z,axis=1,keepdims=True)
            t=target.astype(np.float32);t=t/np.linalg.norm(t,axis=1,keepdims=True);matrices.append(z@t.T)
        K.save_array(out/'cosine_matrices400_FP32.npy',np.stack(matrices,axis=1))
        pairs=C.paired(arrays,queries);path=out/'metrics400.npz';tmp=path.with_name(path.name+'.pending')
        with tmp.open('xb') as f:np.savez_compressed(f,**{arm+'__'+key:value for arm,row in arrays.items() for key,value in row.items()})
        tmp.replace(path)
        K.write(out/'readout400.json',dict(status='COMPLETE_NATIVE63_REFERENCE_SEVEN_ARM_PAIRED_READOUT',
            stage=400,queries=400,gallery=400,arms=list(C.ARMS),metrics=list(C.METRICS),
            means={arm:{key:float(value.mean()) for key,value in row.items()} for arm,row in arrays.items()},
            fixed_comparisons=pairs,arrays=K.ref(path),interpretation=C.interpret(pairs,400),
            bootstrap=dict(seed=1729,unit='concept',replicates=1000),
            architecture_is_published_reference=True,different_training_budgets_from_ATMS_NOT_isolated_architecture_comparison=True,
            text_prompts_reused_from_ATM_cache_NOT_exact_ENIGMA_text_reproduction=True,
            single_subject_R4_exposed_development=True,second_independent_method_or_SOTA=False))
guard();C.resources(c)
print(json.dumps(dict(status='COMPLETE_'+a.mode,seconds=time.monotonic()-started,
    extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)),flush=True)
