"""Finite wait for complete public weights, then one fixed oracle diagnostic.

Stops after the saved-image audit and all-32 preview; full EEG arms require
inspection of these actual outputs, not a success assumption from training.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')
PYTHON='/root/miniconda3/bin/python'


def write(path,value):
    temp=path.with_name(path.name+'.pending')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    report=WORK/'audit/oracle_readiness_queue_v1.json'
    assert not report.exists(),'Preserve existing queue receipt'
    started=time.monotonic()
    write(report,dict(status='WAITING_ONLY_FOR_COMPLETE_NATIVE_SHA_VERIFIED_ASSETS',GPU_started=False))
    while not (WORK/'assets/transfer_complete.json').exists():
        if time.monotonic()-started>7200:
            write(report,dict(status='FINITE_ASSET_WAIT_EXPIRED_NO_GPU',GPU_started=False));return 1
        time.sleep(5)
    command=[PYTHON,str(ROOT/'run_gpu_job_v2.py'),'--name','BrainHIVE_oracle_Tdiag32_v1',
             '--max-seconds','1800','--',PYTHON,str(ROOT/'generate_BrainHIVE_native_images_v1.py'),
             '--role','published','--intervention','oracle','--split','Tdiag','--run-name','oracle_Tdiag32_v1']
    write(report,dict(status='RUNNING_FIXED_ORACLE32',GPU_started=True,argv=command))
    code=subprocess.call(command,cwd=ROOT)
    if code:
        write(report,dict(status='ORACLE_GENERATION_FAILED_PRESERVED_FOR_DIAGNOSIS',exit_code=code));return code
    audit=[PYTHON,str(ROOT/'audit_BrainHIVE_image_arm_CPU_v1.py'),'--run-name','oracle_Tdiag32_v1']
    result=subprocess.run(audit,cwd=ROOT,timeout=180)
    if result.returncode:
        write(report,dict(status='SAVED_ORACLE_PAIRING_AUDIT_FAILED',exit_code=result.returncode));return result.returncode
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    out=WORK/'images/oracle_Tdiag32_v1'
    rows=[json.loads(x) for x in (out/'manifest.jsonl').read_text().splitlines()]
    assert len(rows)==32
    canvas=Image.new('RGB',(1280,8*192),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',11)
    for i,row in enumerate(rows):
        x=(i%4)*320;y=(i//4)*192
        for side,path in enumerate([row['score_target_path'],row['generated_path']]):
            with Image.open(path) as im:
                thumb=ImageOps.contain(im.convert('RGB'),(156,156))
                canvas.paste(thumb,(x+side*160+(156-thumb.width)//2,y+18+(156-thumb.height)//2))
        draw.text((x,y),f"{i+1:02d} {Path(row['stimulus_id']).stem[:35]}",fill='black',font=font)
        draw.text((x,y+176),'Target                  True-image condition',fill='black',font=font)
    preview=WORK/'audit/oracle_Tdiag32_all_preview_v1.jpg'
    assert not preview.exists()
    canvas.save(preview,quality=88)
    with preview.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    write(report,dict(status='ORACLE32_ACTUAL_IMAGES_AND_PAIRING_COMPLETE_AWAITING_INSPECTION',
        images=32,preview=str(preview),preview_sha256=digest,component_gain_established=False,
        full_EEG_generation_started=False,all_preselected_queries_shown=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
