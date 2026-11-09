"""Independent CPU verification of all 40 trained native parent artifacts."""
import csv
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_native_models_v1 import *


def main():
    torch.set_num_threads(2)
    out=WORK/'audit/all40_parent_training_CPU_verification_v1.json'
    assert not out.exists(),'Preserve an existing verification'
    scope=classes()
    proof=[]
    for role in ['published','full','oof0','oof1']:
        directory=WORK/'runs'/(role+'_source_dropout_v2')
        registration=json.loads((directory/'registration.json').read_text())
        complete=json.loads((directory/'completion.json').read_text())
        assert complete['status']=='ALL10_FIXED_EPOCH_ALIGNMENTS_AND_CONDITION_EXPORTS_COMPLETE'
        assert registration['epochs']==25 and registration['frozen_visual_training_dropout']==.1
        assert not registration['deterministic_F_cache_used_as_training_target']
        assert len(complete['models'])==10
        for row in complete['models']:
            subject=row['subject'];folder=directory/subject
            checkpoint=folder/'brain_final.pt'
            assert row['model_sha256']==sha(checkpoint)
            state=torch.load(checkpoint,weights_only=True,map_location='cpu')
            assert state['registration_sha256']==sha(directory/'registration.json')
            assert state['updates']==registration['updates_per_subject']==row['updates']
            assert state['epochs']==25 and state['seed']==1729
            model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval()
            model.load_state_dict(state['model'],strict=True)
            assert sum(p.numel() for p in model.parameters())==6452224
            assert all(torch.isfinite(t).all() for t in state['model'].values())
            epochs=[json.loads(x) for x in (folder/'epochs.jsonl').read_text().splitlines()]
            assert [e['epoch'] for e in epochs]==list(range(1,26))
            assert all(e['full_epoch_rows']==len(registration['training_array_rows']) for e in epochs)
            assert epochs[-1]['updates']==state['updates']
            conditions=torch.load(folder/'conditions.pt',weights_only=True,map_location='cpu')
            assert conditions['model_sha256']==row['model_sha256']
            assert conditions['condition'].shape==(len(conditions['array_rows']),1024)
            assert torch.isfinite(conditions['condition']).all()
            ids=conditions['array_rows'].tolist();assert len(ids)==len(set(ids))
            if role=='published':assert ids==list(range(200))
            elif role=='full':assert ids==list(range(16540))
            else:assert ids==registration['excluded_oof_rows'] and set(ids).isdisjoint(registration['training_array_rows'])
            proof.append(dict(role=role,subject=subject,checkpoint=str(checkpoint),
                checkpoint_sha256=row['model_sha256'],parameters=6452224,epochs=25,updates=row['updates'],
                train_rows=len(registration['training_array_rows']),condition_rows=len(ids),
                condition_cache_sha256=sha(folder/'conditions.pt'),
                final_epoch_InfoNCE=epochs[-1]['mean_train_InfoNCE'],
                training_loss_is_not_image_success=True))
    write(out,dict(status='ALL40_CHECKPOINTS_LOGS_AND_CONDITION_IDENTITIES_VERIFIED_CPU',
        models=proof,actual_images_generated_by_this_check=0,component_gain_established=False))
    print(json.dumps(dict(status='ALL40_CPU_VERIFIED',models=len(proof),actual_images_generated=0)),flush=True)


if __name__=='__main__':main()
