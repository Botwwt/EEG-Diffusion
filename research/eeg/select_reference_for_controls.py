"""Apply the pre-result rule once, using complete paired V image evidence."""
from pathlib import Path
import hashlib
import json
import time

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    rule_path=ROOT/'configs/reference_control_selection_rule.json'
    rule=json.loads(rule_path.read_text())
    assert rule['status']=='LOCKED_BEFORE_EITHER_COMPLETE_V_IMAGE_RESULT'
    post=json.loads((ROOT/'audit/S1_postprocessing_state.json').read_text())
    assert post['status']=='COMPLETED_BOTH_FULL_REFERENCES_POSTPROCESSED',post
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    expected_subjects=set(protocol['subjects']);reports={};inputs={};checkpoints={}
    for method in rule['eligible_references']:
        arm=f'{method}_correct_R4';evaluation=DATA/'evaluation/V'/arm
        image_root=DATA/'images/S1_full_native_batchsafe_v2/V'/arm
        summary_path=evaluation/'metrics/summary.json';summary=json.loads(summary_path.read_text())
        assert summary['status']=='COMPLETED' and summary['queries']==16600 and summary['subjects']==10
        assert summary['method']==method and summary['intervention']=='correct' and summary['repeat_protocol']=='R4'
        assert set(summary['per_subject'])==expected_subjects and summary['gallery_images']==1660
        low=json.loads((evaluation/'low_level/summary.json').read_text())
        assert low['status']=='COMPLETED' and low['expected_images']==low['verified_and_decoded']==16600
        assert low['missing_count']==low['corrupt_or_mismatched_count']==0
        manifest=image_root/'manifest.jsonl';completion=json.loads((image_root/'completion.json').read_text())
        assert completion['status']=='COMPLETED' and completion['queries']==16600 and sha(manifest)==completion['image_manifest_sha256']
        rows=[json.loads(line) for line in manifest.read_text().splitlines()]
        assert len(rows)==16600 and len({r['query_id'] for r in rows})==16600
        assert all(r['image_interface_version']=='native_query_batchsafe_v2' for r in rows)
        assert all(r['EEG_source_stimulus_id']==r['stimulus_id'] for r in rows)
        checkpoints[method]={}
        for subject in sorted(expected_subjects):
            subset=[r for r in rows if r['subject']==subject];assert len(subset)==1660
            for field in ['encoder_sha256','prior_sha256','image_interface_config_sha256']:
                assert len({r[field] for r in subset})==1,(method,subject,field)
            checkpoints[method][subject]={f:subset[0][f] for f in ['encoder_checkpoint','encoder_sha256','prior_checkpoint','prior_sha256']}
        reports[method]=summary
        inputs[method]=dict(summary_path=str(summary_path),summary_sha256=sha(summary_path),
            image_manifest=str(manifest),image_manifest_sha256=sha(manifest),
            low_level_summary_sha256=sha(evaluation/'low_level/summary.json'),
            feature_completion_sha256=sha(evaluation/'features/completion.json'))
    paired=DATA/'evaluation/paired/V_PRIOR_V_vs_REG_v1/summary.json';pair=json.loads(paired.read_text())
    assert pair['status']=='COMPLETED_PAIRED_FULL_PARTITION' and pair['queries']==16600
    a=reports['REF_REG'];b=reports['REF_PRIOR_V']
    criteria=dict(PRIOR_V_DINO_higher=b['summary']['DINO2AFC']['mean']>a['summary']['DINO2AFC']['mean'],
        PRIOR_V_SEED_not_lower=b['summary']['SEED']['mean']>=a['summary']['SEED']['mean'],
        PRIOR_V_ObjectF1_not_lower=b['summary']['ObjectF1']['mean']>=a['summary']['ObjectF1']['mean'],
        PRIOR_V_blank_proxy_not_higher=b['blank_proxy_count']<=a['blank_proxy_count'])
    selected='REF_PRIOR_V' if all(criteria.values()) else 'REF_REG'
    output=ROOT/'audit/reference_control_selection.json'
    body=dict(status='SELECTED_ON_COMPLETE_V_BY_PREDECLARED_RULE',selected_reference=selected,
        criteria=criteria,rule_sha256=sha(rule_path),split_sha256=sha(ROOT/'configs/split_manifest.json'),
        inputs=inputs,paired_summary_sha256=sha(paired),frozen_reference_checkpoints=checkpoints[selected],
        exact_V_images_used_per_method=16600,subjects=10,repeat_protocol='R4',
        source_sha256=sha(__file__),quality_noninferiority_established=False,
        EEG_error_adapter_mechanism_established=False,SOTA_established=False,final_test_unlocked=False,
        limits=['The selected reference is for controlled EEG interventions. Report all metric disagreements.',
            'Development selection is not independent confirmation; these are internal prior-only SDXL systems, not complete paper multipipeline reproductions.',
            'One training seed; full prior inference has additional compute beyond equal UNet NFE.'])
    if output.exists():
        old=json.loads(output.read_text());old.pop('selection_utc',None);assert old==body,'Do not reselect silently'
    else:
        body['selection_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());output.write_text(json.dumps(body,indent=2)+'\n')
    print(json.dumps(body),flush=True)


if __name__=='__main__':main()
