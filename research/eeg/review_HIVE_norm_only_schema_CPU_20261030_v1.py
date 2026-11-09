"""Real protocol/reference-schema review before norm-only GPU admission."""
import csv,json
from pathlib import Path
from brainhive_loss_control_R80_statistics_CPU_20261030_v1 import COMMON,PAPER

def review(study):
    assert study['status']=='REGISTERED_SINGLE1729_FROZEN_EUCLIDEAN_F_NORM_ONLY_COUNTERFACTUAL_v1'
    assert study['training_seed']==1729 and not study['new_training_allowed'] and not study['multiple_training_seeds_allowed']
    assert study['comparisons']==['Norm_only-Euclidean_F','Norm_only-bounded_F','Norm_only-ordinary','Norm_only-parent']
    assert set(study['reference_CSVs'])==set(study['reference_manifests'])=={'parent','ordinary','bounded_F','Euclidean_F'}
    tables={}; headers={}
    for role,parts in study['reference_CSVs'].items():
        assert set(parts)=={'common','paper'};tables[role]={}
        for kind,ref in parts.items():
            with Path(ref['path']).open(newline='') as f:r=csv.DictReader(f);rows=list(r)
            assert len(rows)==2000 and len({row['query_id'] for row in rows})==2000
            assert set(COMMON if kind=='common' else PAPER).issubset(r.fieldnames)
            tables[role][kind]={row['query_id']:row for row in rows};headers[role+'_'+kind]=r.fieldnames
        rows=[json.loads(line) for line in Path(study['reference_manifests'][role]['path']).read_text().splitlines()]
        assert len(rows)==2000 and {row['query_id'] for row in rows}==set(tables[role]['common'])
        assert all('array_row' in row and 'generated_sha256' in row and 'generated_path' in row for row in rows)
    reference=tables['parent']['common'];assert len({r['subject'] for r in reference.values()})==10 and len({r['concept_id'] for r in reference.values()})==200
    for parts in tables.values():
        for rows in parts.values():
            assert set(rows)==set(reference)
            for key,row in rows.items():assert all(row[field]==reference[key][field] for field in ('subject','stimulus_id','concept_id','score_target_path','image_seed'))
    return dict(status='PASS_ACTUAL_STUDY_AND_ALL8_REFERENCE_CSV_ALL4_MANIFEST_SCHEMAS',query_rows=16000,manifest_rows=8000,metric_names=list(COMMON+PAPER),reference_headers=headers,no_new_effects_or_training=True)
