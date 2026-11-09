"""Typed BrainHIVE rapid candidate packs; CPU adapters, no model execution.

Fits export full13230 held predictions or a fixed linear latent operator.
One common generator consumes those packs for either query stage.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
import brainhive_rapid_panel_cache_20261027_v1 as C

ROOT=Path(__file__).resolve().parent
SCHEMA='BrainHIVE_rapid_candidate_pack_20261027_v1'
STATUS='COMPLETE_BRAINHIVE_RAPID_CANDIDATE_PACK'
GEN_SCHEMA='BrainHIVE_rapid_candidate_images_20261027_v1'
GEN_STATUS='COMPLETE_BRAINHIVE_RAPID_CANDIDATE_IMAGES'
ORDER='C-order channel0..3,y0..63,x0..63; native VAE MODE * one scaling_factor'
sha,ref,read,rows,atomic=C.sha,C.ref,C.read,C.rows,C.atomic


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def host_key(host):
    return digest(dict(host_family='BrainHIVE',host=host))


def array(binding,shape):
    path=C.verified(binding);value=np.load(path,mmap_mode='r',allow_pickle=False)
    assert value.shape==tuple(shape) and value.dtype in (np.float32,np.float64) and np.isfinite(value).all()
    return value


def native_binding(path,index_sha):
    native=read(path)
    assert native['status']=='COMPLETE_NATIVE_PARENT_OOF_PREDICTIONS'
    assert native['prediction_index_sha256']==index_sha and native['predictions_shape']==[13230,1024]
    assert native['oof_coverage_min']==native['oof_coverage_max']==1
    value=array(dict(path=native['predictions_path'],sha256=native['predictions_sha256']),[13230,1024])
    assert value.dtype==np.float32
    return native,value


def validate_pack(path,baseline_completion=None,panel=None):
    pack=read(path)
    assert pack['schema']==SCHEMA and pack['status']==STATUS and pack['host_family']=='BrainHIVE'
    assert pack['mode'] in ('native_conditions','linear_latent_update')
    assert re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,63}',pack['arm']) and pack['arm'] not in ('native_parent','target_gallery')
    assert pack['normal_inference_target_access'] is False and pack['additional_subject_EEG_at_inference'] is False
    assert pack['no_query_target_scaling'] is True and pack['no_prediction_L2_normalization'] is True
    assert C.verified(pack['adapter_source']).resolve()==Path(__file__).resolve()
    baseline_path=C.verified(pack['baseline_completion']);baseline=read(baseline_path)
    assert baseline['schema']==C.BASELINE_SCHEMA and baseline['status']==C.BASELINE_STATUS
    assert baseline['queries']==400 and baseline['normal_inference_target_access'] is False
    assert pack['expected_host']==baseline['host'] and pack['host_cache_key']==host_key(baseline['host'])
    if baseline_completion is not None:assert sha(baseline_completion)==pack['baseline_completion']['sha256']
    index_path=C.verified(pack['prediction_index']);index=sorted(rows(index_path),key=lambda row:row['row'])
    assert len(index)==13230 and [int(row['row']) for row in index]==list(range(13230))
    assert len({row['source_row'] for row in index})==13230 and {row['subject_id'] for row in index}=={'sub-01'}
    if panel is not None:assert pack['prediction_index']['sha256']==panel['index']['sha256']
    assert pack['native_parent_binding']['sha256']==baseline['native_parent_binding']['sha256']
    native,_=native_binding(C.verified(pack['native_parent_binding']),pack['prediction_index']['sha256'])
    C.verified(pack['source_fit_summary'])
    assert read(pack['source_fit_summary']['path'])['status']==pack['source_fit_status'] and pack['source_fit_status'].startswith('COMPLETE')
    if pack['mode']=='native_conditions':
        assert pack['payload']['shape']==[13230,1024]
        value=array(pack['payload']['conditions'],[13230,1024]);assert value.dtype==np.float32
        assert pack['payload']['coordinates']=='raw1024 native A input; original fitted coordinates; no query-target scaling'
    else:
        payload=pack['payload'];shape=payload['measurements_shape']
        assert len(shape)==2 and shape[0]==13230 and 1<=shape[1]<=256
        m=shape[1];value=array(payload['measurements'],shape);assert value.dtype==np.float32
        assert 0<float(payload['alpha'])<=1 and math.isfinite(float(payload['alpha']))
        assert payload['latent_flatten_order']==ORDER and payload['latent_scaling_factor']==.13025
        assert payload['latent_scale_already_applied_once'] is True
        receipt=read(C.verified(payload['latent_receipt']))
        assert receipt['status']=='COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS'
        assert receipt['registration']['VAE_scaling_factor']==.13025
        assert set(payload['fold_operators'])=={'0','1'}
        for operator in payload['fold_operators'].values():
            H=array(operator['H'],[16384,m]).astype(np.float64);array(operator['b'],[m])
            K=array(operator['K'],[m,16384]).astype(np.float64)
            penalty=float(operator['ridge_lambda']);assert math.isfinite(penalty) and penalty>0
            gram=H.T@H;assert np.trace(gram)>1e-20
            residual=(gram+penalty*np.eye(m))@K-H.T
            assert np.linalg.norm(residual)/max(np.linalg.norm(H.T),1e-12)<=1e-4,'K is not the declared regularized inverse'
    return pack,baseline,index,native


def build(args):
    source=read(args.fit_summary);baseline=read(args.baseline_completion);native=read(args.native_parent_binding)
    assert baseline['status']==C.BASELINE_STATUS and baseline['schema']==C.BASELINE_SCHEMA
    assert baseline['native_parent_binding']['sha256']==sha(args.native_parent_binding)
    assert source['status']==args.expected_status and source['status'].startswith('COMPLETE')
    index=Path(source['prediction_index']);assert sha(index)==source['prediction_index_sha256']==native['prediction_index_sha256']
    if args.mode=='native_conditions':
        assert args.source_arm
        item=source['arms'][args.source_arm]
        assert item['predictions_shape']==[13230,1024] and item['oof_coverage_min']==item['oof_coverage_max']==1
        payload=dict(conditions=dict(path=item['predictions_path'],sha256=item['predictions_sha256']),shape=[13230,1024],
            coordinates='raw1024 native A input; original fitted coordinates; no query-target scaling',
            fitted_coordinates_description=item['prediction_coordinates'])
    else:
        assert source['schema']=='BrainHIVE_neural_latent_measurement_20261027_v1'
        assert source['status']=='COMPLETE_CROSSBRAIN_NEURAL_LATENT_MEASUREMENT_GATE_PASSED'
        assert source['fixed_practical_forward_gate_passed'] is True and source['generation_operator_exported'] is True
        assert source['additional_subject_EEG_at_inference'] is False and source['no_query_target_access'] is True
        folds={}
        for fold,item in source['forward_folds'].items():
            folds[fold]={name:dict(path=item[name+'_path'],sha256=item[name+'_sha256']) for name in ('H','b','K')}
            folds[fold]['ridge_lambda']=item['ridge_lambda']
        payload=dict(measurements=dict(path=source['query_measurements_path'],sha256=source['query_measurements_sha256']),
            measurements_shape=source['query_measurements_shape'],alpha=source['alpha'],fold_operators=folds,
            latent_flatten_order=source['latent_flatten_order'],latent_scaling_factor=source['latent_scaling_factor'],
            latent_scale_already_applied_once=source['latent_scale_already_applied_once'],latent_receipt=source['latent_receipt'],
            measurements_are_EEG_predictions_not_direct_observations=True,measurement_coordinates=source['query_measurement_coordinates'])
    pack=dict(schema=SCHEMA,status=STATUS,arm=args.arm,mode=args.mode,host_family='BrainHIVE',expected_host=baseline['host'],
        host_cache_key=host_key(baseline['host']),baseline_completion=ref(args.baseline_completion),native_parent_binding=ref(args.native_parent_binding),
        prediction_index=ref(index),source_fit_summary=ref(args.fit_summary),source_fit_status=source['status'],payload=payload,
        normal_inference_target_access=False,additional_subject_EEG_at_inference=False,no_query_target_scaling=True,no_prediction_L2_normalization=True,
        adapter_source=ref(__file__),development_only=True,independent_confirmation=False)
    assert not args.output.exists()
    staged=args.output.with_name(args.output.name+'.validation')
    atomic(staged,pack);validate_pack(staged);staged.rename(args.output)
    print(json.dumps(dict(status=STATUS,pack=ref(args.output),mode=args.mode,arm=args.arm,GPU_started=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    adapt=sub.add_parser('adapt')
    for name in ('fit-summary','baseline-completion','native-parent-binding','output'):adapt.add_argument('--'+name,type=Path,required=True)
    adapt.add_argument('--mode',choices=('native_conditions','linear_latent_update'),required=True)
    adapt.add_argument('--arm',required=True);adapt.add_argument('--source-arm');adapt.add_argument('--expected-status',required=True)
    valid=sub.add_parser('validate');valid.add_argument('--pack',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='adapt':build(args)
    else:
        pack,_,_,_=validate_pack(args.pack);print(json.dumps(dict(status='CPU_TYPED_CANDIDATE_PACK_VERIFIED',mode=pack['mode'],arm=pack['arm'],GPU_started=False)))


if __name__=='__main__':main()
