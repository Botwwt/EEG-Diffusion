"""Fixed coverage preview: 20 evenly spaced stimuli, all ten subjects, no ranking."""
import json
from PIL import Image,ImageOps,ImageDraw
from brainhive_native_models_v1 import ROOT,WORK,sha,write


def main():
    arms=['correct','permuted','missing'];lookup={}
    for arm in arms:
        p=WORK/'images'/('native_R80_'+arm+'_v1')/'manifest.jsonl'
        lookup[arm]={r['query_id']:r for r in map(json.loads,p.read_text().splitlines())}
    subjects=sorted({r['subject'] for r in lookup['correct'].values()});assert len(subjects)==10
    gallery={r['stimulus_id']:r for r in map(json.loads,(WORK/'gallery/benchmark/manifest.jsonl').read_text().splitlines())}
    selected=[]
    for j,subject in enumerate(subjects):
        available=sorted((r for r in lookup['correct'].values() if r['subject']==subject),key=lambda r:r['stimulus_id'])
        assert len(available)==200
        selected.extend([available[20*j],available[20*j+10]])
    receipt=WORK/'audit/native_R80_fixed20_preview_v10.json';assert not receipt.exists()
    files=[]
    for page in range(2):
        canvas=Image.new('RGB',(4*192,10*218+32),'white');draw=ImageDraw.Draw(canvas)
        for column,title in enumerate(['Target','Correct EEG','Permuted EEG','Missing EEG']):draw.text((column*192+5,8),title,fill='black')
        for i,row in enumerate(selected[page*10:(page+1)*10]):
            q=row['query_id'];target=gallery[row['stimulus_id']]
            assert target['image_sha256']==row['target_image_sha256']
            paths=[target['image_path']]
            paths.extend(lookup[arm][q]['generated_path'] for arm in arms)
            y=32+i*218
            for column,path in enumerate(paths):
                with Image.open(path) as image:tile=ImageOps.contain(image.convert('RGB'),(192,192))
                canvas.paste(tile,(column*192+(192-tile.width)//2,y))
            draw.text((4,y+194),row['subject']+' | '+row['stimulus_id'],fill='black')
        path=WORK/'audit'/f'native_R80_fixed20_v10_page{page}.jpg';assert not path.exists();canvas.save(path,quality=90)
        files.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
    write(receipt,dict(selection='Per sorted subject j: sorted stimulus indices20*j and20*j+10; chosen without scores',
        query_ids=[r['query_id'] for r in selected],subjects=10,queries=20,pages=files,
        full_generated_images_retained=True,human_blind_quality_ratings=False,source_sha256=sha(__file__)))
    print('FIXED20_PREVIEW_SAVED',flush=True)


if __name__=='__main__':main()
