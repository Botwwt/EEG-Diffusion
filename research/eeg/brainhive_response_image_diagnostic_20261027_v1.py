"""D16 separate frozen80 image diagnosis; does not undo the failed prediction gate."""
import datetime
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_response_residual_20261027_v1 as V

SCHEMA='BrainHIVE_D16_fixed80_frozen_image_diagnostic_after_failed_proxy_gate_20261027_v1'
ARMS=V.ARMS


def build_conditions(fit,spec,inputs,panel,output,device='cuda'):
    saved=torch.load(U.checked(fit['model']),weights_only=True,map_location='cpu',mmap=True);state={k:v.to(device) for k,v in saved['state'].items()}
    exported=U.read(U.checked(fit['response_export']));lineage=U.read(U.checked(exported['inputs']));old=U.read(U.checked(exported['D15_export']))
    native_cache=torch.load(U.checked(inputs['native_cache']),weights_only=True,map_location='cpu',mmap=True)
    checkpoint=torch.load(U.checked(inputs['native_checkpoint']),weights_only=True,map_location='cpu')['model']
    lookup={int(row):j for j,row in enumerate(native_cache['array_rows'])};qs=panel['Dcheck'];assert len(qs)==400 and {q['subject_id'] for q in qs}=={'sub-01'}
    raw=native_cache['condition'][[lookup[q['array_row']] for q in qs]].clone().to(device)
    d09=U.read(U.checked(spec['D09_completion']))
    original_conditions=np.load(U.checked(d09['check_conditions']),allow_pickle=False)
    assert np.array_equal(raw.cpu().numpy(),original_conditions[:,0])
    gamma=checkpoint['block.layers.0.norm.weight'].to(device);beta=checkpoint['block.layers.0.norm.bias'].to(device);u=(raw-beta)/gamma
    c=(u-state['c_mean'][0])/state['c_scale'][0];dl={row:j for j,row in enumerate(lineage['D_array_rows'])};ix=[dl[q['array_row']] for q in qs]
    aa=np.load(U.checked(old['D_ATMS_full']),mmap_mode='r',allow_pickle=False);rr=np.load(U.checked(exported['D_responses']),mmap_mode='r',allow_pickle=False)
    atms=(torch.from_numpy(np.array(aa[ix],copy=True)).to(device)-state['ATMS_mean'][0])/state['ATMS_scale'][0]
    response=(torch.from_numpy(np.array(rr[ix],copy=True)).to(device)-state['response_mean'][0])/state['response_scale'][0]
    import brainhive_visual_support_20261027_v1 as OLD
    permutation=OLD.control_permutation(qs);contexts=V.contexts(c,atms,response,torch.as_tensor(permutation,device=device))
    bank=[]
    for arm in ARMS:
        model=V.ResponseAdapter().to(device).eval().requires_grad_(False);model.load_state_dict(saved['models'][arm],strict=True)
        with torch.no_grad(),torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):correction,_=model(c,contexts[arm],u)
        bank.append((raw+gamma*correction).cpu());del model
    result=torch.stack(bank,1).numpy();assert result.shape==(400,4,1024) and np.isfinite(result).all()
    U.save_array(output/'all400_four_head_conditions.npy',result)
    lock=dict(status='D16_ALL400_FOUR_FINAL_HEADS_LOCKED_BEFORE_FIRST_DIAGNOSTIC_IMAGE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),query_ids=[q['query_id'] for q in qs],arms=ARMS,
        conditions=U.ref(output/'all400_four_head_conditions.npy'),model=fit['model'],fit_spec=fit['spec'],diagnostic_spec=U.ref(spec['self_path']),
        native_condition_cache=inputs['native_cache'],native_checkpoint=inputs['native_checkpoint'],native_condition_matches_old_D09_parent_bitwise=True,
        original_prediction_gate_passed=False,original_candidate_admitted=False,diagnostic_only=True,automatic_expansion_allowed=False,
        Dcheck_target_F_or_image_used_for_condition=False,target_rank_or_DINO_used_for_choices=False,
        shuffled_response_source_query_permutation=permutation.tolist(),normal_inference_only_EEG_and_frozen_models=True,
        shuffled_arm_is_batch_control_not_single_query_method=True,
        max_radius=.25,old_native_output_affines_preserved=True,new_parameter_updates=0,original_D_resumed=False)
    U.atomic(output/'all400_condition_lock.json',lock);return result,lock


def score(generated,targets,baseline,queries,gallery,stage):
    assert stage==80 and generated.shape==(80,4,768)
    old={a:U.S.metric_arrays(baseline[:80,j],targets,queries,gallery) for j,a in enumerate(U.ARMS)}
    new={a:U.S.metric_arrays(generated[:,j],targets,queries,gallery) for j,a in enumerate(ARMS)}
    primary={a:U.S.compare(new['paired_responses'],v,queries) for a,v in (old|{a:v for a,v in new.items() if a!='paired_responses'}).items()}
    secondary={a:{b:U.S.compare(v,base,queries) for b,base in old.items()} for a,v in new.items() if a!='paired_responses'}
    return dict(status='D16_COMPLETE_FIXED80_FOUR_HEAD_IMAGE_DIAGNOSTIC',stage=80,queries=80,concepts=8,gallery_targets=400,
        means={a:{k:float(v.mean()) for k,v in x.items()} for a,x in (old|new).items()},fixed_primary_paired_minus_references=primary,
        exploratory_other_heads_minus_fixed_baselines=secondary,
        decision=dict(action='DIAGNOSTIC_ONLY_NO_CANDIDATE_ADMISSION_OR_AUTOMATIC_EXPANSION',original_prediction_gate_passed=False,
            original_candidate_admitted=False,automatic_expansion_allowed=False),diagnostic_only=True,new_training=0,
        normal_inference_target_access=False,independent_confirmation=False,SOTA_established=False)
