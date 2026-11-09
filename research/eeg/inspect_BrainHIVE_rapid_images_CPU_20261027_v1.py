"""Read-only full-stage identity/hash audit and deterministic contact sheets.

Every concept contributes its first registered stimulus. Images are resized
only for these labelled review sheets; source images and metrics stay intact.
No trained model, feature extractor, target fitting, or GPU is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(item):
    path = Path(item['path']); assert sha(path) == item['sha256']; return path


def read(path): return json.loads(Path(path).read_bytes())
def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('candidate-completion', 'baseline-completion', 'score-summary', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    assert not args.output.exists(); args.output.mkdir(parents=True)
    generated = read(args.candidate_completion); native = read(args.baseline_completion); score = read(args.score_summary)
    assert generated['status'] == 'COMPLETE_BRAINHIVE_RAPID_CANDIDATE_IMAGES'
    assert score['status'] == 'COMPLETE_CPU_FIXED400_GALLERY_PAIRED_SCREEN'
    assert generated['queries'] == score['queries'] in (80, 400)
    assert score['gallery_targets'] == 400 and score['hidden_exclusions'] == 0
    assert generated['baseline_completion']['sha256'] == sha(args.baseline_completion)
    assert generated['normal_inference_target_access'] is False
    n = generated['queries']; arm = generated['arm']; assert set(score['candidates']) == {arm}
    candidate_rows = rows(checked(generated['manifest']))
    native_rows = {row['query_id']: row for row in rows(checked(native['manifest']))}
    assert len(candidate_rows) == n and len({row['query_id'] for row in candidate_rows}) == n
    selected = {}; identities = []
    for i, row in enumerate(candidate_rows):
        other = native_rows[row['query_id']]
        for key in ('row', 'source_row', 'array_row', 'subject_id', 'concept_id', 'stimulus_id', 'fold',
                    'image_seed', 'initial_noise_latents_sha256', 'initial_UNet_input_sha256',
                    'target_image_path', 'target_image_sha256'):
            assert row[key] == other[key], (i, key)
        assert row['UNet_NFE'] == other['UNet_NFE'] == 4 and row['arm'] == arm
        for item in (row, other):
            assert sha(item['generated_path']) == item['generated_sha256']
            with Image.open(item['generated_path']) as image:
                pixels = np.asarray(image.convert('RGB')); assert pixels.shape == (512, 512, 3)
                assert hashlib.sha256(pixels.tobytes()).hexdigest() == item['generated_pixel_sha256']
        assert sha(row['target_image_path']) == row['target_image_sha256']
        selected.setdefault(row['concept_id'], row)
        identities.append(dict(query_id=row['query_id'], concept_id=row['concept_id'], stimulus_id=row['stimulus_id'],
            native_sha256=other['generated_sha256'], candidate_sha256=row['generated_sha256'], target_sha256=row['target_image_sha256']))
    assert len(selected) == n // 10
    identities_path = args.output / 'all_query_identity_audit.json'
    identities_path.write_text(json.dumps(identities, indent=2) + '\n')
    selected_rows = list(selected.values()); sheets = []
    for start in range(0, len(selected_rows), 10):
        batch = selected_rows[start:start + 10]
        canvas = Image.new('RGB', (600, 34 + len(batch) * 220), 'white'); draw = ImageDraw.Draw(canvas)
        for column, title in enumerate(('TARGET', 'NATIVE', arm)): draw.text((column * 200 + 5, 10), title, fill='black')
        for offset, row in enumerate(batch):
            y = 34 + offset * 220; other = native_rows[row['query_id']]
            for column, path in enumerate((row['target_image_path'], other['generated_path'], row['generated_path'])):
                with Image.open(path) as image: canvas.paste(image.convert('RGB').resize((196, 196), Image.Resampling.LANCZOS), (column * 200 + 2, y))
            draw.text((3, y + 198), row['concept_id'] + ' / ' + row['stimulus_id'], fill='black')
        path = args.output / f'fixed_all_concepts_first_stimulus_page{start // 10 + 1}.jpg'
        canvas.save(path, quality=92); sheets.append(ref(path))
    summary = dict(status='COMPLETE_CPU_ALL_STAGE_ARTIFACT_IDENTITY_AND_FIXED_REVIEW_SHEETS', queries=n, concepts=len(selected),
        full_registered_stage_denominator=True, mismatches=0, candidate=arm, GPU_used=False,
        candidate_completion=ref(args.candidate_completion), native_completion=ref(args.baseline_completion), score_summary=ref(args.score_summary),
        all_query_audit=ref(identities_path), source=ref(__file__), sheets=sheets,
        selection='first stimulus in registered query order of every concept, no score/pixel selection',
        image_gain_or_perceptual_quality_proved_by_audit=False, seconds=time.monotonic() - started)
    path = args.output / 'summary.json'; path.write_text(json.dumps(summary, indent=2) + '\n'); print(json.dumps(summary)); return 0


if __name__ == '__main__': raise SystemExit(main())
