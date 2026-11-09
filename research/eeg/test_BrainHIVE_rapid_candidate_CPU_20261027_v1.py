"""Critical CPU fixtures for packs, named RNG forwarding, and80->400 reuse."""
import argparse
import contextlib
import copy
import hashlib
import inspect
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
import brainhive_rapid_candidate_pack_20261027_v1 as P
import brainhive_rapid_panel_cache_20261027_v1 as C
import generate_BrainHIVE_rapid_candidate_20261027_v1 as G
import cache_BrainHIVE_rapid_candidate_DINO_20261027_v1 as F
import cache_BrainHIVE_rapid_DINO_features_20261027_v1 as B
import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v1 as S


def rejects(call):
    try:call()
    except (AssertionError,KeyError,ValueError):return
    raise AssertionError('Invalid fixture accepted')


def run(panel_path):
    torch.set_num_threads(2);torch.manual_seed(1729);panel=C.validate_panel(panel_path)
    with tempfile.TemporaryDirectory(prefix='brainhive_candidate_cpu_') as temporary:
        root=Path(temporary);index=Path(panel['index']['path'])
        conditions=root/'conditions.npy';np.save(conditions,np.zeros((13230,1024),np.float32),allow_pickle=False)
        native_path=root/'native_binding.json'
        C.atomic(native_path,dict(status='COMPLETE_NATIVE_PARENT_OOF_PREDICTIONS',prediction_index_sha256=C.sha(index),
            predictions_path=str(conditions),predictions_sha256=C.sha(conditions),predictions_shape=[13230,1024],oof_coverage_min=1,oof_coverage_max=1))
        baseline_path=root/'baseline.json';host={'fixture':'BrainHIVE_CPU_only'}
        C.atomic(baseline_path,dict(status=C.BASELINE_STATUS,schema=C.BASELINE_SCHEMA,queries=400,normal_inference_target_access=False,
            host=host,native_parent_binding=C.ref(native_path)))
        fit_path=root/'conditions_fit.json'
        C.atomic(fit_path,dict(status='COMPLETE_SYNTHETIC_RAW_CONDITIONS',prediction_index=str(index),prediction_index_sha256=C.sha(index),
            arms={'raw':dict(predictions_path=str(conditions),predictions_sha256=C.sha(conditions),predictions_shape=[13230,1024],
                oof_coverage_min=1,oof_coverage_max=1,prediction_coordinates='CPU fixture, raw1024')}))
        pack_path=root/'pack.json'
        with contextlib.redirect_stdout(io.StringIO()):
            P.build(SimpleNamespace(fit_summary=fit_path,baseline_completion=baseline_path,native_parent_binding=native_path,
                output=pack_path,mode='native_conditions',arm='candidate',source_arm='raw',expected_status='COMPLETE_SYNTHETIC_RAW_CONDITIONS'))
        pack,_,_,_=P.validate_pack(pack_path);assert pack['mode']=='native_conditions'
        bad=copy.deepcopy(pack);bad['expected_host']={'fixture':'other_host'};bad_path=root/'bad_host.json';C.atomic(bad_path,bad)
        rejects(lambda:P.validate_pack(bad_path))
        H=np.zeros((16384,16),np.float64);H[:16]=np.eye(16);b=np.arange(16,dtype=np.float64)/100;inverse=H.T/1.1
        operators={}
        for name,value in [('H',H),('b',b),('K',inverse)]:
            path=root/(name+'.npy');np.save(path,value,allow_pickle=False)
            operators[name]=dict(path=str(path),sha256=C.sha(path))
        evidence=root/'measurements.npy';np.save(evidence,np.ones((13230,16),np.float32),allow_pickle=False)
        receipt=root/'latent_receipt.json';C.atomic(receipt,dict(status='COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS',registration={'VAE_scaling_factor':.13025}))
        forward_fit=root/'forward_fit.json';fold={name+'_path':binding['path'] for name,binding in operators.items()}|{
            name+'_sha256':binding['sha256'] for name,binding in operators.items()}|dict(ridge_lambda=.1)
        C.atomic(forward_fit,dict(schema='BrainHIVE_neural_latent_measurement_20261027_v1',status='COMPLETE_CROSSBRAIN_NEURAL_LATENT_MEASUREMENT_GATE_PASSED',
            fixed_practical_forward_gate_passed=True,generation_operator_exported=True,additional_subject_EEG_at_inference=False,no_query_target_access=True,
            prediction_index=str(index),prediction_index_sha256=C.sha(index),forward_folds={'0':fold,'1':fold},query_measurements_path=str(evidence),
            query_measurements_sha256=C.sha(evidence),query_measurements_shape=[13230,16],alpha=.25,latent_flatten_order=P.ORDER,
            latent_scaling_factor=.13025,latent_scale_already_applied_once=True,latent_receipt=C.ref(receipt),query_measurement_coordinates='CPU synthetic A16'))
        linear_pack=root/'linear_pack.json'
        with contextlib.redirect_stdout(io.StringIO()):
            P.build(SimpleNamespace(fit_summary=forward_fit,baseline_completion=baseline_path,native_parent_binding=native_path,
                output=linear_pack,mode='linear_latent_update',arm='linear_candidate',source_arm=None,expected_status='COMPLETE_CROSSBRAIN_NEURAL_LATENT_MEASUREMENT_GATE_PASSED'))
        assert P.validate_pack(linear_pack)[0]['payload']['measurements_shape']==[13230,16]
        bad=copy.deepcopy(C.read(linear_pack));bad['payload']['fold_operators']['0']['ridge_lambda']=.2
        wrong_inverse=root/'bad_inverse.json';C.atomic(wrong_inverse,bad);rejects(lambda:P.validate_pack(wrong_inverse))
        # Exact forward update algebra and the explicit Diffusers signature.
        x=torch.randn(1,4,64,64,dtype=torch.float64);y=torch.ones(1,16,dtype=torch.float64)
        ht,bt,kt=map(torch.from_numpy,(H,b,inverse))
        updated=G.forward_update(x,y,ht,bt,kt,.25)
        assert G.forward_update(x,y,ht,bt,kt,0) is x
        assert torch.allclose(updated.flatten(1)-x.flatten(1),.25*(y-(x.flatten(1)@ht+bt))@kt,atol=1e-14)
        assert {'generator','return_dict'}<=set(inspect.signature(G.LinearStep.__call__).parameters)
        received={};sentinel=object();returned=object()
        def step(model_output,timestep,sample,generator=None,return_dict=True):
            received.update(generator=generator,return_dict=return_dict,model_output=model_output);return returned
        scheduler=type('EulerAncestralDiscreteScheduler',(),{})()
        scheduler.config=SimpleNamespace(prediction_type='epsilon',timestep_spacing='trailing');scheduler.step=step;scheduler.step_index=0;scheduler.sigmas=torch.tensor([.7])
        wrapper=G.LinearStep(scheduler,y.float(),ht.float(),bt.float(),kt.float(),0)
        output=torch.zeros(1,4,64,64)
        assert wrapper(output,0,output,generator=sentinel,return_dict=False) is returned
        assert received['generator'] is sentinel and received['return_dict'] is False and received['model_output'] is output
        # Real512 pixel files are fixture artifacts, never model generations.
        image_path=root/'fixture.png';Image.new('RGB',(512,512),(3,7,11)).save(image_path)
        image_sha=C.sha(image_path);pixel_sha=hashlib.sha256(np.asarray(Image.open(image_path).convert('RGB')).tobytes()).hexdigest()
        all_rows=[]
        for i,query in enumerate(panel['queries']):
            all_rows.append(query|dict(arm='candidate',pack_sha256=C.sha(pack_path),UNet_NFE=4,normal_inference_target_access=False,
                generated_path=str(image_path),generated_sha256=image_sha,generated_pixel_sha256=pixel_sha,
                target_image_sha256=panel['target_gallery'][i]['image_sha256'],initial_noise_latents_sha256=f'{i+1000:064x}',
                initial_UNet_input_sha256=f'{i+2000:064x}',condition_sha256=f'{i+3000:064x}',condition_FP16_sha256=f'{i+4000:064x}',tokens_sha256=f'{i+5000:064x}'))
        prefix_manifest=root/'prefix.jsonl';prefix_manifest.write_text(''.join(json.dumps(row)+'\n' for row in all_rows[:80]),encoding='utf-8')
        registration=root/'registration.json';C.atomic(registration,{'CPU_fixture':True})
        parity=root/'parity.json';C.atomic(parity,dict(status='ACTUAL_ZERO_MODULE_NATIVE_PIXEL_PARITY_PASSED'))
        complete80=dict(schema=P.GEN_SCHEMA,status=P.GEN_STATUS,stage='80',queries=80,arm='candidate',runtime_sources=G.runtime_sources(),
            pack=C.ref(pack_path),panel=C.ref(panel_path),baseline_completion=C.ref(baseline_path),native_registration=C.ref(registration),
            manifest=C.ref(prefix_manifest),host=host,host_cache_key=P.host_key(host),zero_module_parity=C.ref(parity))
        generation80=root/'generation80.json';C.atomic(generation80,complete80)
        _,prefix_rows=G.validate_previous(generation80,pack_path,panel_path,baseline_path,registration,G.runtime_sources(),panel['queries'])
        assert prefix_rows==all_rows[:80]
        bad_sources=copy.deepcopy(G.runtime_sources());bad_sources['generator']['sha256']='0'*64
        rejects(lambda:G.validate_previous(generation80,pack_path,panel_path,baseline_path,registration,bad_sources,panel['queries']))
        # End-to-end common scorer fixtures: stage1candidate80 and stage2 only320
        # newly encoded synthetic rows, with the same fixed target400 gallery.
        features=np.eye(400,768,dtype=np.float32);wrong=np.array([i//10*10+(i+1)%10 for i in range(400)])
        native_features=root/'native_features.npy';target_features=root/'target_features.npy'
        np.save(native_features,features[wrong],allow_pickle=False);np.save(target_features,features,allow_pickle=False)
        cache_dir=root/'baseline_cache';cache_dir.mkdir();dummy=C.ref(__file__)
        provenance=dict(host=host,baseline_completion=C.ref(baseline_path),DINO_weight=dummy,image_evaluation_source=dummy,locked_metrics_source=dummy)
        base_views=B.compatible_views(cache_dir,C.ref(panel_path),native_features,target_features,all_rows,provenance)
        native_doc=C.read(base_views['native']['path'])
        out80=root/'features80';out80.mkdir();records80=[F.feature_record(row) for row in all_rows[:80]]
        cache80=F.export_cache(out80,features[:80],records80,complete80,C.ref(generation80),base_views['native'],native_doc,F.extractor_sources())
        score80=S.score(panel_path,base_views['target']['path'],base_views['native']['path'],[cache80],'80',root/'score80')
        assert score80['candidates']['candidate']['decision']['action']=='EXPAND400'
        G.validate_expansion(root/'score80'/'summary.json','candidate',panel_path,baseline_path,generation80,pack_path,P.host_key(host))
        stopped=copy.deepcopy(score80);stopped['candidates']['candidate']['decision']['action']='STOP_INSTANCE'
        stopped_path=root/'stopped_score80.json';C.atomic(stopped_path,stopped)
        rejects(lambda:G.validate_expansion(stopped_path,'candidate',panel_path,baseline_path,generation80,pack_path,P.host_key(host)))
        merged,records400=F.merge_features(features[:80],records80,features[80:],[F.feature_record(row) for row in all_rows[80:]])
        assert np.array_equal(merged,features) and merged[:80].tobytes()==features[:80].tobytes() and records400[:80]==records80
        out400=root/'features400';out400.mkdir();complete400=complete80|dict(stage='400')
        cache400=F.export_cache(out400,merged,records400,complete400,C.ref(generation80),base_views['native'],native_doc,F.extractor_sources(),C.ref(cache80))
        scored=S.score(panel_path,base_views['target']['path'],base_views['native']['path'],[cache400],'400',root/'score400',root/'score80'/'summary.json')
        assert scored['candidates']['candidate']['decision']['action']=='PROMISING_FOR_FULL_EVALUATION'
        altered=merged.copy();altered[0]=altered[1]
        bad_dir=root/'features400_bad';bad_dir.mkdir()
        bad_cache=F.export_cache(bad_dir,altered,records400,complete400,C.ref(generation80),base_views['native'],native_doc,F.extractor_sources(),C.ref(cache80))
        rejects(lambda:S.score(panel_path,base_views['target']['path'],base_views['native']['path'],[bad_cache],'400',root/'badscore',root/'score80'/'summary.json'))
    assert not torch.cuda.is_initialized()
    return dict(status='RAPID_TYPED_CANDIDATE_CRITICAL_CPU_FIXTURES_PASSED',checks=14,
        modes=['native_conditions','linear_latent_update'],full_bank_rows=13230,full_gallery_rows=400,
        exact_stage1_feature_rows_reused=80,new_stage2_feature_rows=320,native_or_target_features_reencoded=0,
        named_generator_and_return_dict_forwarded=True,wrong_host_and_wrong_inverse_rejected=True,
        changed_prefix_features_rejected=True,actual_model_images_generated=0,GPU_initialized=False,
        stage2_requires_explicit_EXPAND400=True,scoring_policy_not_embedded_in_generator=True,
        limitation='CPU synthetic pack/operator/pixel/cache/scorer fixtures; real per-process GPU pixel parity remains mandatory')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--panel',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();result=run(args.panel)
    result['source_pins']={Path(module.__file__).name:C.ref(module.__file__) for module in (P,G,F,C,S)}|{Path(__file__).name:C.ref(__file__)}
    args.receipt.parent.mkdir(parents=True,exist_ok=True);C.atomic(args.receipt,result);print(json.dumps(result))
