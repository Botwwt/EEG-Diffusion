"""Fresh remaining-life storage checks; immutable registration is a forecast."""
import json
from pathlib import Path
import shutil

from brainhive_native_models_v1 import WORK


def bytes_under(path):
    path=Path(path)
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink()) if path.exists() else 0


def verify_remaining(recipe):
    spec=json.loads(Path(recipe['storage_lifecycle_receipt']).read_text());roots=spec['allowed_private_roots']
    need=[512<<20,512<<20];feature_bytes=0;metadata_bytes=bytes_under(WORK/'components/bounded_v1')
    for arm,run in recipe['run_names'].items():
        folder=WORK/'images'/run;manifest=folder/'inference_manifest.jsonl'
        rows=[json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
        done={r['global_query_index'] for r in rows};assert len(done)==len(rows)
        for segment in spec['segments'][arm]:
            count=sum(i not in done for i in range(segment['start'],segment['stop']))
            need[roots.index(segment['root'])]+=count*spec['reserved_bytes_per_image']
        feature_bytes+=bytes_under(WORK/'evaluation'/run/'features')
        metadata_bytes+=bytes_under(folder)+bytes_under(WORK/'evaluation'/run/'scores')+bytes_under(WORK/'evaluation'/run/'low_level')
    feature_bytes+=bytes_under(WORK/'evaluation/bounded_V_gallery_features_v1')
    metadata_bytes+=max(0,bytes_under(WORK/'gallery/V')-spec['initial_V_gallery_bytes'])
    metadata_bytes+=bytes_under(WORK/'gallery/bounded_component_V_v1')
    need[0]+=max(0,650000000-feature_bytes)+max(0,300000000-metadata_bytes)
    free=[shutil.disk_usage(p).free for p in roots]
    assert all(a>=b for a,b in zip(free,need)),f'Remaining main lifecycle storage forecast is not funded: free={free}, required={need}'
    return dict(free=free,remaining_required=need,feature_bytes_written=feature_bytes,metadata_bytes_written=metadata_bytes)
