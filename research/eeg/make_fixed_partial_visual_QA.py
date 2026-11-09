"""Fixed first8 distinct V concepts for visual saved-file/pairing QA only.

No ranking, metric computation, checkpoint/seed choice or final-test access.
The selected query list cannot change after first creation.
"""
from pathlib import Path
import hashlib
import io
import json
import zipfile
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    source=DATA/'images/S1_full_native_batchsafe_v2/V/REF_REG_correct_R4'
    records=[json.loads(p.read_text()) for p in (source/'receipts/sub-01').glob('*.json')]
    records.sort(key=lambda row:row['array_row'])
    chosen=[];concepts=set()
    for row in records:
        if row['concept_id'] not in concepts:
            concepts.add(row['concept_id']);chosen.append(row)
        if len(chosen)==8:break
    assert len(chosen)==8
    output=DATA/'diagnostics/V_fixed8_partial_QA_REG_v1';assert not output.exists();output.mkdir(parents=True)
    (output/'targets').mkdir();archives={};sheet=Image.new('RGB',(1080,1300),'white')
    draw=ImageDraw.Draw(sheet);font=ImageFont.load_default()
    draw.text((12,10),'PARTIAL V VISUAL QA ONLY: sub-01, fixed first8 distinct concepts; no scores',fill='black',font=font)
    for i,row in enumerate(chosen):
        assert row['partition']=='V' and row['subject']=='sub-01' and row['method']=='REF_REG'
        assert row['intervention']=='correct' and row['EEG_source_stimulus_id']==row['stimulus_id']
        assert row['image_interface_version']=='native_query_batchsafe_v2'
        assert sha(row['generated_path'])==row['generated_sha256']
        archive=row['target_image_zip']
        if archive not in archives:archives[archive]=zipfile.ZipFile(archive)
        raw=archives[archive].read(row['target_image_member'])
        assert hashlib.sha256(raw).hexdigest()==row['target_image_sha256']
        target=output/'targets'/f'{row["array_row"]:08d}.jpg';target.write_bytes(raw)
        assert sha(target)==row['target_image_sha256']
        y=42+(i//2)*310;x=12+(i%2)*536
        label=f'{row["array_row"]} {row["stimulus_id"]}'
        draw.text((x,y),label,fill='black',font=font)
        draw.text((x,y+15),'TARGET                         REF_REG',fill='black',font=font)
        for j,path in enumerate([target,Path(row['generated_path'])]):
            with Image.open(path) as im:
                im.load();image=ImageOps.contain(im.convert('RGB'),(252,252))
                square=Image.new('RGB',(256,256),'#eeeeee');square.paste(image,((256-image.width)//2,(256-image.height)//2))
            sheet.paste(square,(x+j*264,y+30))
        row['QA_extracted_original_target_path']=str(target)
        row['QA_target_verified_against_source_zip_sha']=True
        row['QA_score_target_path']=None
        row['QA_score_status']='NOT_SCORED; full evaluation/gallery not completed'
    for archive in archives.values():archive.close()
    image=output/'fixed8_contact.jpg';sheet.save(image,quality=90,subsampling=0)
    report=dict(status='FIXED_PARTIAL_V_VISUAL_QA_READY_NOT_A_FULL_EVALUATION',queries=chosen,selection_rule='sub-01, first8 different concepts in ascending actual array_row',
        source=str(source),contact_sheet=str(image),contact_sheet_sha256=sha(image),source_sha256=sha(__file__),
        current_image_method_ranked=False,any_new_metrics_computed=False,best_seed_selected=False,final_test_accessed=False,
        original_event_stimulus_labels_available=False,
        limits=['Partial development visual/pairing QA only, not a full16600-query performance result.',
            'Concept metadata IDs and ZIP folder numeric prefixes are different namespaces; original ZIP bytes and stimulus filename hashes are checked.',
            'Released-array order and image metadata do not independently replay unavailable acquisition-event labels.'])
    (ROOT/'audit/V_fixed8_partial_visual_QA.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['queries','limits']}),flush=True)


if __name__=='__main__':main()
