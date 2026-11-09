"""D05: fixed partial raw-neural denoising followed by the unchanged BrainMLP.

Only sub01 and outer-FIT group-A EEG are method inputs.  Group-B arrays and
visual targets are never opened.  The D04 independent measurement gate is a
prerequisite, not an adaptive hyperparameter selector.
"""
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
import fit_BrainHIVE_raw_neural_recoverability_20261027_v1 as N

ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_raw_neural_partial_denoiser_20261027_v1'
STATUS = 'COMPLETE_RAW_NEURAL_PARTIAL_DENOISER_NATIVE_CONDITIONS'
COORDINATES = 'raw native BrainMLP OOF output after fixed original-EEG or partial-neural-EEG replay; original affine retained; no target access or scaling'
TEACHERS = ('sub-02', 'sub-03', 'sub-04', 'sub-05')
METHOD_SUBJECTS = ('sub-01',) + TEACHERS
ALPHA = .25
SEED = 1729

sha, ref, atomic = N.sha, N.ref, N.atomic


def verified(binding):
    path = Path(binding['path'])
    assert sha(path) == binding['sha256'], f'SHA mismatch: {path}'
    return path


def on_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: on_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [on_device(v, device) for v in value]
    return value


def select_A_state(checkpoint, fold, items):
    """Whitelist A and query states; no B-derived field enters computation."""
    fit = [i for i, row in enumerate(items) if row['fold'] != fold]
    held = [i for i, row in enumerate(items) if row['fold'] == fold]
    assert checkpoint['outer_fold'] == fold
    assert checkpoint['fit_source_rows'] == [items[i]['source_row'] for i in fit]
    assert checkpoint['held_source_rows'] == [items[i]['source_row'] for i in held]
    assert set(checkpoint['subject_models']) >= set(METHOD_SUBJECTS)
    state = dict(subject_models={name: checkpoint['subject_models'][name] for name in METHOD_SUBJECTS},
                 MCCA_A=checkpoint['MCCA_A'], query_to_A=checkpoint['query_to_A'])
    assert state['MCCA_A']['rank'] == 16 and state['MCCA_A']['regularization'] == .1
    for model in state['subject_models'].values():
        assert model['center'].shape == model['scale'].shape == (4250,)
        assert model['PCA_basis'].shape == (4250, 64)
    assert state['query_to_A']['weights'].shape == (64, 16)
    return state, fit, held


def partial_update(raw, predicted_standardized, model, alpha=ALPHA):
    if alpha == 0:
        return raw
    standard = (raw.double() - model['center'].double()) / model['scale'].double()
    mixed = (1-alpha)*standard + alpha*predicted_standardized.double()
    return (model['center'].double() + model['scale'].double()*mixed).float()


def vector_diagnostics(raw, clean):
    raw, clean = raw.double(), clean.double()
    delta = clean-raw
    return dict(raw_input_RMS=raw.square().mean(1).sqrt(), clean_input_RMS=clean.square().mean(1).sqrt(),
        update_RMS=delta.square().mean(1).sqrt(), update_relative_L2=delta.norm(dim=1)/raw.norm(dim=1).clamp_min(1e-12),
        raw_input_mean=raw.mean(1), clean_input_mean=clean.mean(1), update_mean=delta.mean(1))


def replay_diagnostics(predicted, cached):
    predicted, cached = predicted.double(), cached.double()
    delta = predicted-cached
    return dict(relative_L2=delta.norm(dim=1)/cached.norm(dim=1).clamp_min(1e-12),
        max_absolute_component_error=delta.abs().amax(1), RMS=delta.square().mean(1).sqrt(),
        cosine=F.cosine_similarity(predicted,cached,dim=1), reencoded_norm=predicted.norm(dim=1), cached_norm=cached.norm(dim=1))


def replay_gate(values):
    return bool(values['relative_L2'].max() <= 1e-4 and values['max_absolute_component_error'].max() <= 1e-3
                and values['cosine'].min() >= 1-1e-6)


def describe(values):
    return {key: dict(mean=float(value.mean()), minimum=float(value.min()), maximum=float(value.max()))
            for key, value in values.items()}


def original_brain_class(source_root):
    """Execute only pinned original BrainMLP dependencies, without optional imports."""
    source_root = Path(source_root)
    receipt = json.loads((source_root/'_fetch_receipt.json').read_bytes())
    expected = {item['path']: item['sha256'] for item in receipt['artifacts']}
    scope = dict(torch=torch, nn=torch.nn, F=F)
    references = {'fetch_receipt': ref(source_root/'_fetch_receipt.json')}
    for filename, names in [('main/layers.py',['BaseMLP','ResNet1DLayer','ResNet1DBlock']),
                            ('main/models_brain.py',['BrainMLP'])]:
        path = source_root/filename
        assert sha(path) == expected[filename]
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in names]
        assert [node.name for node in nodes] == names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
        references[filename] = ref(path)
    return scope['BrainMLP'], references


def bind_encoder_source(contract, fold, items):
    subjects = [item for item in contract['subjects'] if item['subject'] == 'sub-01']
    assert len(subjects) == 1
    sources = [item for item in subjects[0]['sources'] if item['role'] == f'oof{fold}']
    assert len(sources) == 1
    source = sources[0]
    held = [i for i, row in enumerate(items) if row['fold'] == fold]
    by_source_row = {int(row['source_row']): i for i, row in enumerate(items)}
    group = source['groups']['component_train']
    positions = [by_source_row[int(row['array_row'])] for row in group]
    assert len(positions) == len(set(positions)) and set(positions) == set(held)
    assert all(items[i]['fold'] == fold for i in positions)
    return source, positions, [int(row['cache_row']) for row in group]


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    a = torch.randn(90, 5, dtype=torch.float64)
    target = a @ torch.randn(5, 12, dtype=torch.float64) + 2
    fitted = N.ridge_fit(a,target)
    centered = a-a.mean(0)
    residual = (centered.T@centered/len(a)+.1*torch.eye(5,dtype=a.dtype))@fitted['weights']-centered.T@(target-target.mean(0))/len(a)
    assert residual.abs().max() < 1e-12
    raw = torch.randn(90,12)
    model = dict(center=raw.mean(0),scale=raw.std(0,unbiased=False).clamp_min(1e-4))
    prediction = N.ridge_predict(a,fitted)
    assert partial_update(raw,prediction,model,0) is raw
    clean = partial_update(raw,prediction,model)
    z = (raw.double()-model['center'].double())/model['scale'].double()
    clean_z = (clean.double()-model['center'].double())/model['scale'].double()
    assert torch.allclose(clean_z-prediction,.75*(z-prediction),atol=6e-7,rtol=1e-5)
    full = partial_update(raw,prediction,model,1)
    assert torch.allclose(full,(model['center'].double()+model['scale'].double()*prediction).float())
    assert replay_gate(replay_diagnostics(raw,raw))
    assert not replay_gate(replay_diagnostics(raw+1,raw))
    values = vector_diagnostics(raw,raw)
    assert values['update_RMS'].max() == values['update_relative_L2'].max() == 0
    cls, references = original_brain_class(ROOT/'reproduction_sources/BrainHIVE_f06db69f')
    brain = cls(4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False)
    assert sum(p.numel() for p in brain.parameters()) == 6452224
    x = torch.zeros(2,17,250)
    with torch.inference_mode():
        first = brain(x)
        assert first.shape == (2,1024) and torch.equal(first,brain(x.reshape(2,4250)))
        clone = cls(4250,1024,num_layers=1,dropout=.1).eval()
        clone.load_state_dict(brain.state_dict(),strict=True)
        assert torch.equal(first,clone(x))
    items = [dict(fold=i%2,source_row=i+10) for i in range(6)]
    sources = [dict(role=f'oof{f}',groups={'component_train':[dict(array_row=i+10,cache_row=j)
        for j,i in enumerate(range(f,6,2))]}) for f in (0,1)]
    source, positions, rows = bind_encoder_source(dict(subjects=[dict(subject='sub-01',sources=sources)]),0,items)
    assert positions == [0,2,4] and rows == [0,1,2]
    source['groups']['component_train'][0]['array_row'] = 11
    try:
        bind_encoder_source(dict(subjects=[dict(subject='sub-01',sources=sources)]),0,items)
        raise RuntimeError('wrong-role source was not rejected')
    except AssertionError:
        pass
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='CPU_SELF_TESTS_PASSED',tests=10,CUDA_initialized=False,
        original_source_refs=references,scope='synthetic ridge/update/drift/role checks and original-class CPU shape+strict reload; no actual learned checkpoint replay')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','d04-models-binding','contract','native-parent-binding','spec','output'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--source-root',type=Path,default=ROOT/'reproduction_sources/BrainHIVE_f06db69f')
    parser.add_argument('--gpu-job-name')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-seconds',type=float,default=600)
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test();return 0
    assert all((args.inputs,args.d04_models_binding,args.contract,args.native_parent_binding,args.spec,args.output,args.gpu_job_name))
    assert args.device == 'cuda:0' and 0 < args.max_seconds <= 600 and not args.output.exists()
    args.output.mkdir(parents=True)
    invoked = time.monotonic(); stopped=[False]
    signal.signal(signal.SIGTERM,lambda number,frame:stopped.__setitem__(0,True))
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    D.torch = torch
    document = dict(gpu_job_name=args.gpu_job_name,maximum_seconds=args.max_seconds,
        maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,
        maximum_output_bytes=256<<20,resource_authorization=ref(ROOT/'configs/resource_authorization_20261020_v3.json'))
    wrapper=D.verify_wrapper(document); budget=D.Budget(document)
    def check():
        assert not stopped[0] and time.monotonic()-invoked < args.max_seconds,'Finite D05 limit'
        budget.check()
    def log(value):
        value=value|dict(seconds=time.monotonic()-invoked)
        with (args.output/'progress.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush()
        print(json.dumps(value),flush=True)
    try:
        torch.set_num_threads(2);torch.manual_seed(SEED);torch.cuda.set_device(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        spec=json.loads(args.spec.read_bytes())
        assert spec['schema']==SCHEMA and spec['alpha']==ALPHA and spec['reverse_ridge']==.1 and spec['inference_batch']==256
        assert sha(N.__file__)==spec['D04_source_sha256']
        prereg_path=ROOT/spec['scientific_preregistration']['relative_path']
        assert sha(prereg_path)==spec['scientific_preregistration']['sha256']
        prereg=json.loads(prereg_path.read_bytes());assert prereg['alpha']==ALPHA and not prereg['other_brain_query_input']
        assert sha(args.contract)==spec['input_contract_sha256']
        inputs=json.loads(args.inputs.read_bytes())
        assert inputs['status']=='COMPLETE_RAW_NEURAL_T17_ALL10_INPUTS' and inputs['rows']==13230 and inputs['concepts']==1323
        index=Path(inputs['prediction_index']);assert sha(index)==inputs['prediction_index_sha256']==spec['prediction_index_sha256']
        items=N.read_index(index)
        assert len(items)==13230 and [row['row'] for row in items]==list(range(13230))
        assert all(row['subject_id']=='sub-01' and row['partition']=='T' for row in items)
        assert len({row['concept_id'] for row in items})==1323
        contract=json.loads(args.contract.read_bytes());assert contract['split_manifest_sha256']==spec['split_manifest_sha256']
        d04_binding=json.loads(args.d04_models_binding.read_bytes())
        assert d04_binding['status']=='D04_COMPLETE_FIXED_GATE_AND_MODELS_BOUND' and d04_binding['fixed_practical_screen_gate_passed'] is True
        d04_completion_path=verified(d04_binding['completion']);d04_summary_path=verified(d04_binding['summary'])
        d04_completion=json.loads(d04_completion_path.read_bytes());d04_summary=json.loads(d04_summary_path.read_bytes())
        assert d04_completion['status']==d04_summary['status']=='COMPLETE_RAW_NEURAL_RECOVERABILITY_SCREEN'
        assert d04_completion['summary']['sha256']==sha(d04_summary_path) and d04_summary['fixed_practical_screen_gate_passed'] is True
        assert d04_summary['query_subject']=='sub-01' and d04_summary['group_A']==list(TEACHERS)
        native=json.loads(args.native_parent_binding.read_bytes())
        assert native['status']=='COMPLETE_NATIVE_PARENT_OOF_PREDICTIONS' and native['source_contract_sha256']==sha(args.contract)
        assert native['subject_id']=='sub-01'
        assert native['prediction_index_sha256']==sha(index) and native['predictions_shape']==[13230,1024]
        assert native['oof_coverage_min']==native['oof_coverage_max']==1
        native_path=verified(dict(path=native['predictions_path'],sha256=native['predictions_sha256']))
        cached=np.load(native_path,mmap_mode='r',allow_pickle=False);assert cached.shape==(13230,1024) and cached.dtype==np.float32
        control_bindings={key:ref(path) for key,path in [('inputs',args.inputs),('spec',args.spec),('prereg',prereg_path),
            ('index',index),('D04_binding',args.d04_models_binding),('contract',args.contract),('native_parent_binding',args.native_parent_binding),
            ('D04_completion',d04_completion_path),('D04_summary',d04_summary_path),('native_predictions',native_path)]}
        raw_bindings={name:{key:inputs['subjects'][name][key] for key in ('path','sha256')} for name in METHOD_SUBJECTS}
        for binding in raw_bindings.values():check();verified(binding)
        cls,source_refs=original_brain_class(args.source_root)
        source_refs['D04_implementation']=ref(N.__file__)
        source_refs['this_implementation']=ref(__file__)
        original_digest=sha(__file__)
        # Whole D04 checkpoint decoding includes unused B fields; whitelist A/query
        # immediately afterwards. No B compact EEG array is opened anywhere.
        atomic(args.output/'provenance.json',dict(schema=SCHEMA,source_refs=source_refs,control_bindings=control_bindings,
            raw_source_bindings=raw_bindings,wrapper=wrapper,D04_model_bindings=d04_binding['fold_models'],
            checkpoint_B_fields_deserialized_then_ignored=True,B_arrays_opened=False,B_states_used_for_method=False,
            additional_subject_EEG_at_inference=False,no_query_target_access=True,visual_F_or_latent_loaded=False,
            private_information_preservation_guaranteed=False,perturbation_norm_bounded_by_alpha=False))
        raw_array=np.load(raw_bindings['sub-01']['path'],mmap_mode='r',allow_pickle=False)
        assert raw_array.shape==(13230,17,250) and raw_array.dtype==np.float32
        raw01=torch.from_numpy(np.array(raw_array,copy=True).reshape(13230,4250)).to(args.device);del raw_array
        assert torch.isfinite(raw01).all()
        predictions={arm:np.empty((13230,1024),np.float32) for arm in ('native_reencoded','neural_clean')}
        coverage=np.zeros(13230,np.int32);row_diagnostics=[None]*13230;fold_results=[];encoder_refs={}
        for fold in (0,1):
            check();model_binding=d04_binding['fold_models'][str(fold)];checkpoint_path=verified(model_binding)
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
            state,fit_positions,held_positions=select_A_state(checkpoint,fold,items);del checkpoint
            state=on_device(state,args.device)
            fit=torch.tensor(fit_positions,device=args.device);held=torch.tensor(held_positions,device=args.device)
            views=[]
            for name in TEACHERS:
                check();array=np.load(raw_bindings[name]['path'],mmap_mode='r',allow_pickle=False)
                assert array.shape==(13230,17,250) and array.dtype==np.float32
                # Index before copying: no held group-A samples enter D05 method.
                teacher_fit=torch.from_numpy(np.array(array[fit_positions],copy=True).reshape(len(fit_positions),4250)).to(args.device);del array
                assert torch.isfinite(teacher_fit).all()
                views.append(N.subject_transform(teacher_fit,state['subject_models'][name]).double());del teacher_fit
            A_fit,_=N.group_transform(views,state['MCCA_A']);del views
            model01=state['subject_models']['sub-01']
            x1_fit_z=(raw01[fit].double()-model01['center'].double())/model01['scale'].double()
            reverse=N.ridge_fit(A_fit,x1_fit_z);del x1_fit_z,A_fit
            query_A=N.ridge_predict(N.subject_transform(raw01[held],model01),state['query_to_A'])
            predicted_z=N.ridge_predict(query_A,reverse)
            raw_held=raw01[held]
            clean_held=partial_update(raw_held,predicted_z,model01)
            assert torch.isfinite(clean_held).all()
            raw_metrics=vector_diagnostics(raw_held,clean_held)
            source,source_positions,cache_rows=bind_encoder_source(contract,fold,items)
            for path_key,hash_key in [('checkpoint_path','checkpoint_sha256'),('condition_cache_path','condition_cache_sha256'),
                                      ('parent_registration_path','parent_registration_sha256')]:
                verified(dict(path=source[path_key],sha256=source[hash_key]))
            registration=json.loads(Path(source['parent_registration_path']).read_bytes())
            assert registration['role']==f'oof{fold}' and registration['source_class']=='main.models_brain.BrainMLP'
            assert set(registration['training_array_rows'])=={items[i]['source_row'] for i in fit_positions}
            assert set(registration['excluded_oof_rows'])=={items[i]['source_row'] for i in held_positions}
            condition_cache=torch.load(source['condition_cache_path'],map_location='cpu',weights_only=True,mmap=True)
            assert condition_cache['role']==f'oof{fold}' and condition_cache['model_sha256']==source['checkpoint_sha256']
            assert condition_cache['array_rows'][cache_rows].tolist()==[items[i]['source_row'] for i in source_positions]
            assert np.array_equal(condition_cache['condition'][cache_rows].float().numpy(),np.asarray(cached[source_positions]))
            del condition_cache
            encoder_refs[str(fold)]={key:value for key,value in source.items() if key!='groups'}
            encoder=cls(4250,1024,num_layers=1,dropout=.1).to(args.device).eval().requires_grad_(False)
            loaded=torch.load(source['checkpoint_path'],map_location='cpu',weights_only=True)
            assert loaded['registration_sha256']==source['parent_registration_sha256']
            encoder.load_state_dict(loaded['model'],strict=True);del loaded
            assert sum(p.numel() for p in encoder.parameters())==6452224
            def encode(values):
                pieces=[]
                with torch.inference_mode():
                    for batch in values.split(256):
                        check();result=encoder(batch).float();assert torch.isfinite(result).all();pieces.append(result.cpu())
                return torch.cat(pieces).numpy()
            # Original raw input is sent directly, avoiding numerical roundtrips.
            pred_native=encode(raw_held);pred_clean=encode(clean_held)
            predictions['native_reencoded'][held_positions]=pred_native
            predictions['neural_clean'][held_positions]=pred_clean
            coverage[held_positions]+=1
            drift=replay_diagnostics(torch.from_numpy(pred_native),torch.from_numpy(np.array(cached[held_positions],copy=True)))
            fold_passed=replay_gate(drift)
            summary=dict(outer_fold=fold,fit_rows=len(fit_positions),held_rows=len(held_positions),
                reverse_ridge=.1,alpha=ALPHA,raw_input_update=describe(raw_metrics),native_cache_replay=describe(drift),
                engineering_gate_passed=fold_passed,encoder_source=encoder_refs[str(fold)],D04_model=model_binding)
            fold_results.append(summary);atomic(args.output/f'fold{fold}_summary.json',summary)
            for local,position in enumerate(held_positions):
                row_diagnostics[position]=items[position]|dict(encoder_role=f'oof{fold}',encoder_checkpoint_sha256=source['checkpoint_sha256'],
                    encoder_condition_cache_sha256=source['condition_cache_sha256'],
                    raw_input_update={key:float(value[local]) for key,value in raw_metrics.items()},
                    native_cache_replay={key:float(value[local]) for key,value in drift.items()})
            torch.save(N.cpu_value(dict(reverse=reverse,outer_fold=fold,alpha=ALPHA,
                D04_model_sha256=model_binding['sha256'],fit_source_rows=[items[i]['source_row'] for i in fit_positions],
                held_source_rows=[items[i]['source_row'] for i in held_positions])),args.output/f'fold{fold}_reverse.pt')
            log(dict(stage='RAW_NEURAL_PARTIAL_DENOISER_FOLD_EXPORTED',**summary))
            del state,model01,reverse,query_A,predicted_z,raw_held,clean_held,raw_metrics,encoder
            torch.cuda.empty_cache()
        assert np.all(coverage==1) and all(value is not None for value in row_diagnostics)
        per_query=args.output/'per_query.jsonl.gz'
        with gzip.open(per_query,'wt',encoding='utf-8') as stream:
            for value in row_diagnostics:stream.write(json.dumps(value,allow_nan=False)+'\n')
        arms={}
        for name,value in predictions.items():
            assert np.isfinite(value).all()
            directory=args.output/name;directory.mkdir();path=directory/'predictions.npy';np.save(path,value,allow_pickle=False)
            arms[name]=dict(predictions_path=str(path.resolve()),predictions_sha256=sha(path),predictions_shape=list(value.shape),
                prediction_coordinates=COORDINATES,oof_coverage_min=int(coverage.min()),oof_coverage_max=int(coverage.max()),
                predictions_norm_mean=float(np.linalg.norm(value.astype(np.float64),axis=1).mean()))
        check();assert sha(__file__)==original_digest
        for binding in control_bindings.values():verified(binding)
        for binding in source_refs.values():verified(binding)
        for binding in raw_bindings.values():check();verified(binding)
        for binding in d04_binding['fold_models'].values():verified(binding)
        for source in encoder_refs.values():
            for pk,hk in [('checkpoint_path','checkpoint_sha256'),('condition_cache_path','condition_cache_sha256'),
                          ('parent_registration_path','parent_registration_sha256')]:verified(dict(path=source[pk],sha256=source[hk]))
        assert sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file()) < 256<<20
        passed=all(fold['engineering_gate_passed'] for fold in fold_results)
        summary=dict(status=STATUS if passed else 'STOP_RAW_NEURAL_DENOISER_NATIVE_REPLAY_DRIFT',schema=SCHEMA,
            rows=13230,concepts=1323,alpha=ALPHA,reverse_ridge=.1,query_subject='sub-01',training_teacher_subjects=list(TEACHERS),
            additional_subject_EEG_at_inference=False,no_query_target_access=True,visual_F_or_latent_loaded=False,
            B_arrays_opened=False,B_states_used_for_method=False,engineering_gate_passed=passed,
            prediction_index=str(index.resolve()),prediction_index_sha256=sha(index),arms=arms,
            outer_folds=fold_results,per_query=ref(per_query),native_parent_binding=ref(args.native_parent_binding),
            scientific_preregistration=ref(prereg_path),spec=ref(args.spec),provenance=ref(args.output/'provenance.json'),
            private_information_preservation_guaranteed=False,perturbation_norm_bounded_by_alpha=False,
            seconds=time.monotonic()-invoked,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,
            development_only=True,SOTA_established=False)
        atomic(args.output/'summary.json',summary)
        atomic(args.output/'completion.json',dict(status=summary['status'],schema=SCHEMA,summary=ref(args.output/'summary.json')))
        log(dict(stage='RAW_NEURAL_DENOISER_COMPLETE' if passed else 'STOP_NATIVE_REPLAY_DRIFT',status=summary['status'],engineering_gate_passed=passed))
        return 0 if passed else 3
    except BaseException as error:
        atomic(args.output/'failure.json',dict(status='INCOMPLETE_RAW_NEURAL_PARTIAL_DENOISER_NO_ADAPTIVE_REPLAY',error=repr(error),
            traceback=traceback.format_exc(),seconds=time.monotonic()-invoked))
        raise


if __name__=='__main__':
    raise SystemExit(main())
