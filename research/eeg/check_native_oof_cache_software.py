"""CPU round-trip/overwrite/ID checks only; DOES NOT create scientific OOF data.

The clearly marked fixture uses four existing full-encoder vectors solely to
test serialization. It bypasses OOF construction intentionally, uses invalid
unit_fixture teacher labels, and cannot pass the scientific cache prerequisites.
"""
from pathlib import Path
import csv
import json
import os
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
from build_native_oof_cache import CacheBuilder,torch,ROOT,sha


def main():
    started=time.monotonic()
    fixture=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/diagnostics/cache_software_fixture_v1')
    assert not fixture.exists(),'Use a new explicit fixture version'
    fixture.mkdir(parents=True)
    (fixture/'NOT_SCIENTIFIC_OOF_DATA.json').write_text(json.dumps(dict(test_only=True,OOF=False,
        reason='Serialization test using four full-encoder vectors, not out-of-fold predictions. Invalid unit_fixture teacher labels.'))+'\n')
    config_path=fixture/'software_check_config.json';config=dict(condition_chain='REF_REG',coordinate_space='raw_pooled_CLIP_H_1024_before_native_IP_projection',test_only=True)
    config_path.write_text(json.dumps(config)+'\n')
    cfg=json.loads((ROOT/'configs/full_reference_resolved.json').read_text())
    split=json.loads((ROOT/'configs/split_manifest.json').read_text());rows=split['image_rows']['T'][:4]
    proof=json.loads((Path(cfg['output_root'])/'full/sub-01/completion.json').read_text())
    vectors=torch.load(proof['condition_cache'],map_location='cpu',weights_only=True)['condition'][rows].clone()
    ci=torch.load(cfg['visual_features_path'],map_location='cpu',weights_only=True,mmap=True)[rows].clone()
    bank=fixture/'C_I_T.pt';torch.save(dict(test_only=True,C_I=ci,array_rows=torch.tensor(rows)),bank)
    queries={}
    with (ROOT/'audit/query_manifest_R4.csv').open() as file:
        for row in csv.DictReader(file):
            key=(row['subject_id'],int(row['array_row']))
            if key[0]=='sub-01' and key[1] in rows:queries[key]=row
            if len(queries)==4:break
    assert len(queries)==4
    # Constructor deliberately not invoked: real OOF checkpoints do not exist yet.
    b=CacheBuilder.__new__(CacheBuilder);b.out=fixture;b.config_path=config_path;b.config=config
    b.queries=queries;b.banks={'T':dict(path=str(bank),sha256=sha(bank),array_rows=rows)}
    b.counts=dict(loaded_cached_queries=0)
    parts=[]
    for index in range(2):
        path=fixture/'parts'/f'fixture_{index}.pt';path.parent.mkdir(exist_ok=True)
        torch.save(dict(test_only=True,C_E=vectors[index*2:index*2+2]),path)
        parts.append(dict(path=str(path),sha256=sha(path),queries=2))
    roles=['unit_fixture_oof0']*2+['unit_fixture_oof1']*2
    proofs={'unit_fixture':'No OOF model trained; software test only'}
    report=b.export('sub-01','T',vectors,roles,proofs,parts)
    assert not any(Path(p['path']).exists() for p in parts)
    path=Path(report['path']);original_sha=sha(path)
    loaded=torch.load(path,weights_only=True,map_location='cpu')
    assert torch.equal(loaded['C_E'],vectors) and loaded['query_ids']==[queries[('sub-01',r)]['query_id'] for r in rows]
    assert b.existing_export('sub-01','T','T_oof',roles,proofs)['sha256']==original_sha
    changed=vectors.clone();changed[0,0]+=1
    try:b.export('sub-01','T',changed,roles,proofs,[])
    except AssertionError:overwrite_rejected=True
    else:raise AssertionError('Existing condition vectors were overwritten')
    assert sha(path)==original_sha
    first=queries[('sub-01',rows[0])];old=dict(first);first['query_id']=queries[('sub-01',rows[1])]['query_id']
    try:b.export('sub-01','T',vectors,roles,proofs,[])
    except AssertionError:duplicate_ID_rejected=True
    else:raise AssertionError('Repeated query IDs accepted')
    queries[('sub-01',rows[0])]=old;assert sha(path)==original_sha
    result=dict(status='PASSED_CPU_CACHE_SERIALIZATION_AND_INTEGRITY_SOFTWARE_CHECKS',test_only=True,
        lossless_condition_roundtrip=True,completed_export_reused_without_resampling=True,
        overwritten_condition_vectors_rejected=overwrite_rejected,duplicate_query_IDs_rejected=duplicate_ID_rejected,
        redundant_shards_removed_only_after_verified_merge=True,GPU_initialized=torch.cuda.is_initialized(),
        actual_trained_OOF_folds_verified=False,scientific_OOF_data_created=False,fixture=str(fixture),
        cache_builder_sha256=sha(ROOT/'build_native_oof_cache.py'),check_source_sha256=sha(__file__),
        seconds=time.monotonic()-started,
        remaining_checks='Actual20 fold models, exact sampled chain, full132300 T and16500 D coverage must be checked after real prerequisite completion.')
    assert not result['GPU_initialized']
    (ROOT/'audit/native_oof_cache_software_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
