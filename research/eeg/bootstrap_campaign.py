"""Create an isolated campaign and import evidence without promoting pilots."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parent
OLD=Path('/root/atm_diffusion_20261015/research/eeg_diffusion_next_20261018_v1')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name in ['audit','research','runs','evaluation','gallery','logs','configs']:(ROOT/name).mkdir(exist_ok=True)
    inventory=json.loads((OLD/'forensics/model_inventory.json').read_text())
    shutil.copy2(OLD/'forensics/model_inventory.json',ROOT/'audit/historical_model_inventory.json')
    for name in ['retrieval.json','mapper_bridge.json','pairing_contract.json','physical_rows.json','metric_checks.json','loss_interpretation.json']:
        shutil.copy2(OLD/'forensics'/name,ROOT/'audit'/name)
    shutil.copy2(OLD/'reports/原因定位报告.md',ROOT/'audit/历史原因定位报告.md')
    fields='module source_repo source_commit class_name checkpoint sha256 input_shape output_shape depth width heads total_params trainable_params pretraining_data training_data trainable precision normalization prediction_type load_missing_keys load_unexpected_keys'.split()
    rows=[]
    def add(name,entry,repo,shape_in,shape_out,depth,width,trained=False):
        row={key:'' for key in fields}
        row.update(module=name,source_repo=repo,source_commit='UNRESOLVED_SOURCE_COMMIT',class_name=entry['class_name'],
            checkpoint=entry['checkpoint'],sha256=entry['sha256'],input_shape=shape_in,output_shape=shape_out,depth=depth,width=width,
            total_params=entry['parameters'],trainable_params=entry['parameters'] if trained else 0,
            trainable=trained,pretraining_data='existing EEG T1526' if name=='legacy_EEG' else 'none; trained from scratch in prior pilot',
            training_data='historical sub-01 only; not formal full-subject baseline',precision='float32',
            load_missing_keys=0,load_unexpected_keys=0)
        rows.append(row)
    add('legacy_EEG',inventory['EEG'],'Brain-HIVE','B,17,250','B,1024',1,1024)
    add('legacy_REF_PRIOR',inventory['PRIOR'],'ncclab-sustech/EEG_Image_decode','x:B,1024;c:B,1024;t:B','v:B,1024',4,'1024/512/256/128/64',True)
    add('legacy_REF_REG_mapper',inventory['point_mapper'],'ncclab-sustech/EEG_Image_decode','x=0;c:B,1024;t=0','native CLIP:B,1024',4,'1024/512/256/128/64',True)
    for name,entry in inventory['adapters'].items():
        add('legacy_'+name,entry,'isolated run image_engine.py:Controller','x:B,4,64,64;c:B,1024;b:B,17,250;t:B','epsilon residual:B,4,64,64',3,64,True)
    with (ROOT/'audit/architecture_inventory.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    result_fields='run_id parent_run hypothesis status git_commit config_hash data_manifest_hash protocol_version train_seed subject_scope R encoder_ckpt prior_ckpt generator_ckpt adapter_ckpt trainable_params total_params train_images train_trials train_updates effective_epochs gpu_hours eval_queries expected_images generated_images failed_images dino_2afc semantic_metric object_f1 quality_metric delta_vs_best_matched_ref paired_ci_low paired_ci_high seconds_per_image peak_vram keep_decision reason'.split()
    if not (ROOT/'results.tsv').exists():
        with (ROOT/'results.tsv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=result_fields,delimiter='\t');writer.writeheader()
            for name in ['REF-REG','REF-PRIOR','REF-ADAPTER','REF-CONLY','A1','A2','C1','C2']:
                metric=json.loads((OLD/'metrics'/(name+'_V.json')).read_text())
                row={key:'' for key in result_fields}
                row.update(run_id='historical_'+name,status='completed_historical_diagnostic',parent_run=str(OLD),
                    hypothesis='Prior fixed small discovery; not eligible as formal full-data baseline',
                    protocol_version='legacy_sub01_R4_V64_seed1103',train_seed=1729,subject_scope='sub-01',R=4,
                    eval_queries=64,expected_images=64,generated_images=64,failed_images=0,
                    dino_2afc=metric['summary']['DINO2AFC'],keep_decision='not_formal',
                    reason='Preserve negative results and forensic provenance; full-subject training/benchmark not done')
                writer.writerow(row)
    state=dict(stage='S0_ACTIVE',training_seed=1729,previous_goal_turn_classification='progress',
        historical_evidence=str(OLD),historical_evidence_revalidated=True,formal_training_started=False,
        candidate_limit=12,structural_revision_limit=2,resource_budget='inherit existing40GPUh authorization; profile before formal long training',
        protected=['subject/stimulus alignment','split manifests','metric definition','query RNG','final-test access'],
        primary_bottleneck_evidence='EEG joint V64 Top1 .65625 -> native point .109375; mapper T pool .97265625; frozen visual-joint oracle .6875; C2 fixed T objective worsens',
        verified_wait=None)
    (ROOT/'campaign_state.json').write_text(json.dumps(state,indent=2)+'\n')
    (ROOT/'STATUS.md').write_text('# Full-data single-seed EEG autoresearch\n\nStage: S0 active. Formal full-subject experiments have not started.\n\n'
        'Master prompt read completely (700 lines). Fixed training seed1729 is inherited; no second training replicate. '
        'Historical sub-01/V64 pilots remain diagnostics. They do not satisfy full data or SOTA requirements.\n\n'
        'Evidence: current EEG is BrainMLP6,452,224 parameters, not ATM. Native conditional mapper has a strong train/development gap. '
        'True-image native conditions work; learned adapters cause finite-value artifacts. C2 optimization is not established.\n\n'
        'Current resource: one RTX4080SUPER reporting32760MiB, GPU idle. Root disk~1.2GB and data disk~3GB free; RAM asset area~21GB free. '
        'Storage, full-subject data provenance, official architecture and evaluator requirements are being audited before new long jobs. '
        'No old experiments were removed.\n\n'
        'Next: inventory all subjects/physical repeats, read pinned official ATM/FOSA entry points, profile full references, '
        'lock full protocol and resource reservation, then train and evaluate complete partitions.\n')
    print(json.dumps(dict(status='CAMPAIGN_INITIALIZED',root=str(ROOT),historical_rows=8,architecture_rows=len(rows))))


if __name__=='__main__':main()
