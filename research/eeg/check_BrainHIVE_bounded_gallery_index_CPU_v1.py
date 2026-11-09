"""Synthetic saved-score joins and old-asset URLs; no real V rows or images."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from urllib.parse import unquote

import build_BrainHIVE_bounded_gallery_index_CPU_v1 as builder


def check():
    with tempfile.TemporaryDirectory(prefix='hive_gallery_CPU_') as temporary:
        root = Path(temporary)
        work = root / 'work'
        gate_paths = [work / 'comparison/bounded_v1/summary.json', work / 'gallery/bounded_component_V_v1/coverage.json',
            work / 'audit/main_storage_continuation_20261021_v1.json']
        gate_rows = [dict(status=builder.COMPARISON_DONE), dict(rows=16600, generated_images=49800, pages=166),
            dict(status='RUNNING', pid=12345, completed=[], failures={})]
        for path, row in zip(gate_paths, gate_rows):
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(row))
        actual_gate, actual_read = builder.prerequisites, builder.read
        with patch.object(builder, 'prerequisites', side_effect=lambda: actual_gate(work)), patch.object(builder, 'read', wraps=actual_read) as observed:
            assert builder.build(root / 'unread_recipe', root / 'not_created') is None
            assert not (root / 'not_created').exists()
            assert [call.args[0] for call in observed.call_args_list] == [gate_paths[2]]
        gate_paths[2].write_text(json.dumps(dict(status=builder.MAIN_DONE, pid=12345, completed=list(builder.ARMS), failures={})))
        with patch.object(builder, 'prerequisites', side_effect=lambda: actual_gate(work)), patch.object(builder, 'process_live', return_value=True), patch.object(builder, 'read', wraps=actual_read) as observed:
            assert builder.build(root / 'unread_recipe', root / 'not_created') is None
            assert not (root / 'not_created').exists()
            assert [call.args[0] for call in observed.call_args_list] == [gate_paths[2]]
        work, old, output = root / 'work', root / 'work/gallery/bounded_component_V_v1', root / 'work/gallery/new'
        roots = [root / 'volume0', root / 'volume1']
        for directory in roots + [old / 'assets/targets', work / 'gallery/V']:
            directory.mkdir(parents=True, exist_ok=True)
        # File aliases stand in for existing cross-volume symlinks; builder never copies them.
        targets, manifests, scores, summaries, sources = {}, {}, {}, {}, {}
        for index in (1, 0):
            stimulus = f'q{index}.jpg'; target_path = work / 'gallery/V' / stimulus
            target_path.write_bytes(b'target fixture ' + bytes([index]))
            (old / 'assets/targets' / stimulus).hardlink_to(target_path)
            targets[stimulus] = dict(stimulus_id=stimulus, concept_id='concept', array_row=100 + index,
                image_zip='/registered/no_read.zip', image_member=stimulus, image_sha256=builder.sha(target_path))
        for arm_index, arm in enumerate(builder.ARMS):
            manifests[arm], scores[arm] = {}, {}
            sources[arm] = dict(manifest_path=arm + '.jsonl', manifest_sha256='a' * 64, score_csv_path=arm + '.csv', score_csv_sha256='b' * 64)
            for index in (1, 0):
                stimulus = f'q{index}.jpg'; query = f'sub-01/{stimulus}/R4'
                volume = arm_index % 2; path = roots[volume] / arm / f'q{index}.webp'
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((arm + str(index)).encode())
                alias = old / 'assets' / f'volume{volume}' / arm / path.name
                alias.parent.mkdir(parents=True, exist_ok=True); alias.hardlink_to(path)
                target_path = str(work / 'gallery/V' / stimulus)
                row = dict(query_id=query, subject='sub-01', stimulus_id=stimulus, concept_id='concept', array_row=100 + index,
                    image_seed=1000 + index, EEG_source_stimulus_id=stimulus, repeat_protocol='R4', intervention='correct',
                    target_image_sha256=targets[stimulus]['image_sha256'], score_target_path=target_path,
                    generated_path=str(path), generated_sha256=builder.sha(path))
                manifests[arm][query] = row
                values = {metric: .2 + .1 * arm_index + .01 * index for metric in builder.METRICS}
                scores[arm][query] = dict(query_id=query, subject='sub-01', stimulus_id=stimulus, concept_id='concept',
                    EEG_source_stimulus_id=stimulus, image_seed=str(1000 + index), generated_path=str(path),
                    score_target_path=target_path, target_image_path=target_path, blank_proxy='False', **values)
            means = {metric: sum(float(row[metric]) for row in scores[arm].values()) / 2 for metric in builder.METRICS}
            summaries[arm] = dict(status='COMPLETED', queries=2, repeat_protocol='R4', intervention='correct',
                summary={metric: dict(mean=value) for metric, value in means.items()}, per_subject={'sub-01': means})
        comparison = dict(table=[dict(arm=arm, **{metric: summaries[arm]['summary'][metric]['mean'] for metric in builder.METRICS}) for arm in builder.ARMS])
        def joined(m=manifests, s=scores, t=targets, summaries_arg=summaries):
            urls = lambda path, target: builder.asset_url(path, output, old, roots, target)
            with patch.object(builder, 'WORK', work):
                return builder.join(m, s, t, summaries_arg, comparison, urls, sources, expected_n=2)
        rows, trace, subject_rows, cases = joined()
        assert [row['id'] for row in rows] == sorted(manifests['parent'])
        assert len(rows) == len(trace) == 2 and len(subject_rows) == 3
        assert all(row['pageLocator'] == '../bounded_component_V_v1/page_000.html' for row in rows)
        for row in rows:
            assert row['target'] == row['targetFull'] and row['target'].startswith('../bounded_component_V_v1/assets/targets/')
            for cell in row['cells']:
                assert cell['full'] == cell['thumb'] and '/assets/volume' in cell['full']
                assert (output / cell['full']).resolve().is_file()
        for reference, values in cases.items():
            assert len(values) == 2 and {v['query_id'] for v in values} == set(manifests['parent'])
            tied = [dict(value, DINO2AFC_delta=.1) for value in reversed(values)]
            assert [r['query_id'] for r in sorted(tied, key=lambda r: (-r['DINO2AFC_delta'], r['query_id']))] == sorted(manifests['parent'])
            assert [r['query_id'] for r in sorted(tied, key=lambda r: (r['DINO2AFC_delta'], r['query_id']))] == sorted(manifests['parent'])
        query = sorted(manifests['parent'])[0]
        bad_variants = [('query_id', 'wrong_query'), ('image_seed', 999), ('array_row', 999), ('concept_id', 'wrong'),
            ('EEG_source_stimulus_id', 'wrong.jpg'), ('target_image_sha256', '0' * 64),
            ('generated_path', manifests['parent'][query]['generated_path'])]
        for field, wrong in bad_variants:
            bad = copy.deepcopy(manifests); bad['risk'][query][field] = wrong
            try: joined(m=bad)
            except AssertionError: pass
            else: raise AssertionError('Mixed manifest identity accepted: ' + field)
        bad_scores = copy.deepcopy(scores); bad_scores['risk'][query]['score_target_path'] = '/wrong/target.jpg'
        try: joined(s=bad_scores)
        except AssertionError: pass
        else: raise AssertionError('Wrong scoring target accepted')
        bad_summary = copy.deepcopy(summaries); bad_summary['risk']['per_subject']['sub-01']['SEED'] += .1
        try: joined(summaries_arg=bad_summary)
        except AssertionError: pass
        else: raise AssertionError('Wrong subject metric mean accepted')
        try: builder.indexed([dict(query_id='same'), dict(query_id='same')])
        except AssertionError: pass
        else: raise AssertionError('Duplicate ID accepted')
        wrong_alias = old / 'assets/volume0/wrong.webp'; wrong_alias.write_bytes(b'wrong file')
        unrelated = roots[0] / 'wrong.webp'; unrelated.write_bytes(b'different actual file')
        try: builder.asset_url(unrelated, output, old, roots)
        except AssertionError: pass
        else: raise AssertionError('Wrong samefile asset mapping accepted')
        odd = roots[1] / 'odd #.webp'; odd.write_bytes(b'URL escaping fixture')
        (old / 'assets/volume1/odd #.webp').hardlink_to(odd)
        odd_url = builder.asset_url(odd, output, old, roots)
        assert odd_url.endswith('odd%20%23.webp') and (output / unquote(odd_url)).resolve().samefile(odd)
        return dict(schema='BrainHIVE_bounded_gallery_index_CPU_contract_v1', status='PASS_SYNTHETIC_SCHEMA_AND_JOINS_ONLY',
            checks=['actual_early_gate_no_score_reads_or_output', 'live_main_even_with_complete_gate_files_rejected',
                'unordered_complete_ID_join', 'duplicate_ID_rejected',
                'seed_row_concept_EEG_target_generated_path_mismatch_rejected', 'wrong_score_target_rejected',
                'two_volume_samefile_URL_mapping', 'samefile_wrong_alias_rejected', 'URL_space_and_fragment_escaped',
                'saved_subject_and_global_means_verified',
                'full_case_coverage_stable_ID_ties'], actual_V_score_rows_read=False, images_decoded_or_copied_by_builder=False,
            GPU_used=False, source_sha256={Path(__file__).name: builder.sha(__file__), Path(builder.__file__).name: builder.sha(builder.__file__)})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output'); args = parser.parse_args()
    result = check()
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream: stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
