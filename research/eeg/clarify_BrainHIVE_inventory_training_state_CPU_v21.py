"""Keep v19 tensor audit; distinguish completed training from registered totals."""
import csv
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    source=ROOT/'audit/BrainHIVE_current_model_inventory_CPU_v19.json'
    data=json.loads(source.read_text())
    assert sha(ROOT/'audit_BrainHIVE_current_model_inventory_CPU_v19.py')==data['source_sha256']
    for row in data['models']:
        registered=row.pop('epochs')
        row['registered_epochs_or_passes']=registered
        if row['module']=='EEG':
            row['completed_epochs_or_passes']=25
            row['training_status']='COMPLETE_FIXED_FINAL_EEG_PARENT'
        elif row['module'].startswith('component_'):
            arm=row['module'].split('_',1)[1]
            folder=WORK/'components/bounded_v1'/arm
            progress=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else None
            completion=json.loads((folder/'completion.json').read_text()) if (folder/'completion.json').exists() else None
            row['updates']=completion['updates'] if completion else progress.get('updates',0) if progress else 0
            row['completed_epochs_or_passes']=3 if completion else progress.get('seen_queries',0)//132300 if progress else 0
            row['training_status']='COMPLETE_FIXED_FINAL_COMPONENT' if completion else progress['status'] if progress else 'NOT_STARTED'
            row['checkpoint_sha256']=sha(row['checkpoint']) if Path(row['checkpoint']).exists() else None
        else:
            row['registered_epochs_or_passes']=None
            row['completed_epochs_or_passes']=None
            row['training_status']='EXISTING_PRETRAINED_ORIGINAL_TRAINING_HISTORY_UNKNOWN'
        row['checkpoint_exists_at_snapshot']=Path(row['checkpoint']).exists()
    data.update(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        annotation_source_sha256=sha(__file__),tensor_audit_receipt=str(source),tensor_audit_receipt_sha256=sha(source),
        status='ACTUAL_V19_TENSOR_COUNTS_WITH_EXPLICIT_COMPLETED_VS_REGISTERED_TRAINING_STATE',
        completed_epochs_or_passes_are_complete_full_data_passes_only=True,
        original_v19_component_epochs_were_registered_total_not_completed=True,
        original_tensor_audit_preserved=True,new_tensor_loads=0,new_GPU_jobs=0,new_images=0,new_candidate_configurations=0)
    output=ROOT/'audit/BrainHIVE_current_model_inventory_CPU_v21.json';assert not output.exists()
    output.write_text(json.dumps(data,indent=2)+'\n')
    output=ROOT/'audit/BrainHIVE_current_model_inventory_CPU_v21.csv';assert not output.exists()
    with output.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data['models'][0]));writer.writeheader();writer.writerows(data['models'])
    print(json.dumps({r['module']:dict(updates=r['updates'],completed_passes=r['completed_epochs_or_passes'],
        registered_passes=r['registered_epochs_or_passes'],status=r['training_status'],final_checkpoint_exists=r['checkpoint_exists_at_snapshot'])
        for r in data['models'] if r['module'].startswith('component_')}))


if __name__=='__main__':main()
