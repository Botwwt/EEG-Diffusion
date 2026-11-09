"""Bind existing immutable input pins; never invent missing protocol fields."""
from pathlib import PurePosixPath

def reference_bindings(study):
    assert study['status']=='REGISTERED_SINGLE1729_EUCLIDEAN_F_POSTFIT_AMENDED_NUMERIC_VERIFICATION_v2'
    assert study['comparisons']==['bounded_F-Euclidean_F','Euclidean_F-parent','ordinary-Euclidean_F']
    pins={item['path']:item for item in study['input_pins']}
    assert len(pins)==len(study['input_pins'])
    def unique(suffix):
        found=[item for path,item in pins.items() if path.endswith(suffix)]
        assert len(found)==1, suffix
        return found[0]
    w='/root/brainhive_reproduction_20261020_v1/'
    parent_common=study['R80_execution_template']['baseline_common_csv']
    parent_paper=study['R80_execution_template']['baseline_paper_csv']
    parent_manifest=study['R80_spec_template']['baseline_scoring_manifest']
    for item in (parent_common,parent_paper,parent_manifest): assert pins[item['path']]==item
    assert parent_common['path'].startswith(w) and parent_manifest['path'].startswith(w)
    tables={
        'parent':dict(common=parent_common,paper=parent_paper),
        'ordinary':dict(common=unique('/HIVE_ordinary_R80_benchmark_20261029_v1/evaluation/scores/per_query.csv'),paper=unique('/HIVE_ordinary_R80_benchmark_20261029_v1/evaluation/original_paper_metrics/per_query.csv')),
        'bounded_F':dict(common=unique('/HIVE_loss_controls_R80_single_seed_20261030_v1/F_only/evaluation/scores/per_query.csv'),paper=unique('/HIVE_loss_controls_R80_single_seed_20261030_v1/F_only/evaluation/original_paper_metrics/per_query.csv')),
    }
    manifests={
        'parent':parent_manifest,
        'ordinary':unique('/HIVE_ordinary_R80_benchmark_20261029_v1/ordinary/manifest.jsonl'),
        'bounded_F':unique('/HIVE_loss_controls_R80_single_seed_20261030_v1/F_only/loss_control/manifest.jsonl'),
    }
    assert all(PurePosixPath(item['path']).is_absolute() for group in tables.values() for item in group.values())
    return tables,manifests
