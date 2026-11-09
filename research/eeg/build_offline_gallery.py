"""Complete offline gallery, stable pagination, and separate blinded materials.

All image assets stay on the server. Hard links avoid copying full JPEG bytes.
The blinded directory omits method labels, scores, and the unblinding key.
No human ratings are fabricated, and the gallery never selects a generation seed.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import html
import json
import os
import time
from PIL import Image


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def mapping(path,kind):
    with Path(path).open() as file:
        rows=list(csv.DictReader(file)) if kind=='csv' else [json.loads(line) for line in file]
    index={r['query_id']:r for r in rows};assert len(index)==len(rows)
    return index


STYLE='''body{font:15px system-ui,sans-serif;margin:20px;background:#f6f6f6;color:#222}header{position:sticky;top:0;background:#fff;padding:12px;z-index:2;border-bottom:1px solid #aaa}select,input,button{font:inherit;margin:4px;padding:6px}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border:1px solid #ccc;vertical-align:top;padding:6px;overflow-wrap:anywhere}img{width:100%;max-width:256px;height:220px;object-fit:contain;background:#eee}small{display:block}code{font-size:12px}.notice{padding:12px;background:#fff3ca}nav{padding:12px}a{color:#1363a2}'''


def frame(title,body,tail=''):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+STYLE+'</style><body>'+body+tail+'</body></html>'


APP=r'''
const D=window.GALLERY;let page=0;const size=40;
const E=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const [id,key] of [['subject','subject'],['concept','concept']]){
 for(const v of [...new Set(D.rows.map(x=>x[key]))].sort()){
  const o=document.createElement('option');o.value=v;o.textContent=v;E(id).appendChild(o);
 }
}
function render(){
 let rows=D.rows.filter(x=>(!E('subject').value||x.subject===E('subject').value)&&(!E('concept').value||x.concept===E('concept').value)&&(!E('search').value||x.id.toLowerCase().includes(E('search').value.toLowerCase())));
 if(!D.blinded&&E('error').value){const f=E('error').value;rows=rows.filter(x=>x.cells.some(c=>f==='blank'?c.blank:f==='dino'?c.dino<.5:c.object<.5));}
 if(!D.blinded&&E('sort').value)rows.sort((a,b)=>{const sign=E('sort').value==='best'?-1:1;return sign*(a.cells.reduce((s,c)=>s+c.dino,0)-b.cells.reduce((s,c)=>s+c.dino,0))||a.id.localeCompare(b.id);});
 page=Math.max(0,Math.min(page,Math.max(0,Math.ceil(rows.length/size)-1)));
 E('count').textContent=`${rows.length} / ${D.rows.length} queries；第 ${page+1} / ${Math.max(1,Math.ceil(rows.length/size))} 页。`;
 const head='<tr><th>目标 / ID</th>'+D.labels.map(x=>`<th>${esc(x)}</th>`).join('')+'</tr>';
 E('images').innerHTML=head+rows.slice(page*size,(page+1)*size).map(x=>'<tr><td>'+`<code>${esc(x.id)}</code><small>${esc(x.concept)}</small><a href="${esc(x.targetFull)}" target="_blank"><img loading="lazy" src="${esc(x.target)}"></a>`+'</td>'+x.cells.map(c=>`<td><a href="${esc(c.full)}" target="_blank"><img loading="lazy" src="${esc(c.thumb)}"></a><small>${esc(c.label)}${D.blinded?'':` · DINO2AFC ${c.dino.toFixed(4)} · SEED ${c.seed.toFixed(4)} · ObjectF1 ${c.object.toFixed(4)}${c.blank?' · blank proxy':''}`}</small></td>`).join('')+'</tr>').join('');
}
for(const id of ['subject','concept','search','error','sort'])E(id).addEventListener('input',()=>{page=0;render();});
E('previous').onclick=()=>{page--;render();};E('next').onclick=()=>{page++;render();};render();
'''


def app_page(blinded=False):
    title='完整盲评材料' if blinded else '完整 EEG 图像重建画廊'
    notice=('每行方法顺序由固定哈希打乱；A/B 等标签只在该行有效。此目录不包含方法名、分数或揭盲表。尚未开展人工评分。'
            if blinded else '包含完整分区，每 query 一张图；默认按稳定 ID 排序。DINO2AFC 不是对象正确率。最佳/最差排序只供诊断，不改变评价分母。人工评分未开展；质量非劣尚未证明。')
    body=f'<h1>{title}</h1><p class="notice">{notice}</p><header><label>被试<select id="subject"><option value="">全部</option></select></label><label>概念<select id="concept"><option value="">全部</option></select></label><input id="search" placeholder="query ID / stimulus 文件名">'
    body+='<select id="error" '+('hidden' if blinded else '')+'><option value="">全部诊断类型</option><option value="blank">空白代理</option><option value="dino">任一列 DINO2AFC &lt; 0.5</option><option value="object">任一列 ObjectF1 &lt; 0.5</option></select>'
    body+='<select id="sort" '+('hidden' if blinded else '')+'><option value="">稳定 ID</option><option value="worst">平均 DINO 最低（诊断）</option><option value="best">平均 DINO 最高（诊断）</option></select><div id="count"></div><button id="previous">上一页</button><button id="next">下一页</button></header><table id="images"></table>'
    return frame(title,body,'<script src="data.js"></script><script>'+APP+'</script>')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',nargs=3,action='append',required=True,metavar=('LABEL','MANIFEST','METRICS'))
    parser.add_argument('--output',required=True);args=parser.parse_args();started=time.monotonic()
    out=Path(args.output);assert not out.exists(),'Use a fresh versioned gallery; preserve scientific history'
    manifests={label:mapping(path,'json') for label,path,_ in args.arm}
    metrics={label:mapping(path,'csv') for label,_,path in args.arm}
    assert len(manifests)==len(args.arm)
    labels=list(manifests);keys=sorted(manifests[labels[0]])
    assert len(keys) in [16600,2000] and len({manifests[labels[0]][k]['subject'] for k in keys})==10
    for label in labels:assert set(keys)==manifests[label].keys()==metrics[label].keys()
    out.mkdir(parents=True);blind=out/'blinded';blind.mkdir();(blind/'full').mkdir();(blind/'thumbs').mkdir();(out/'pages').mkdir()
    assets={};duplicates=collections.defaultdict(list);unblind=[];rows=[];blind_rows=[]
    def asset(path,expected):
        key=hashlib.sha256((str(path)+'|'+expected).encode()).hexdigest()[:32]
        if key not in assets:
            assert sha(path)==expected
            full=blind/'full'/f'{key}.jpg';thumb=blind/'thumbs'/f'{key}.jpg'
            os.link(path,full)
            with Image.open(full) as image:
                image.load();image.convert('RGB').resize((256,256),Image.Resampling.LANCZOS).save(thumb,quality=86,subsampling=0)
            assets[key]=dict(source=str(path),source_sha256=expected,full=str(full.relative_to(blind)),thumb=str(thumb.relative_to(blind)))
        return assets[key]
    for index,key in enumerate(keys):
        first=manifests[labels[0]][key];score=metrics[labels[0]][key]
        target=asset(score['score_target_path'],first['target_image_sha256'])
        record=dict(id=key,subject=first['subject'],concept=first['concept_id'],
            target='blinded/'+target['thumb'],targetFull='blinded/'+target['full'],cells=[])
        blinded=dict(id=key,subject=first['subject'],concept=first['concept_id'],target=target['thumb'],targetFull=target['full'],cells=[])
        blind_order=sorted(labels,key=lambda label:hashlib.sha256(f'1729|blind-gallery|{key}|{label}'.encode()).hexdigest())
        blind_label={label:chr(65+i) for i,label in enumerate(blind_order)}
        for label in labels:
            row=manifests[label][key];m=metrics[label][key]
            for field in ['query_id','subject','stimulus_id','concept_id','image_seed','target_image_sha256','initial_image_noise_sha256']:
                assert first[field]==row[field],(key,label,field)
            assert row['generated_path']==m['generated_path'] and row['stimulus_id']==m['stimulus_id']
            assert Path(score['score_target_path'])==Path(m['score_target_path'])
            a=asset(row['generated_path'],row['generated_sha256']);duplicates[(label,row['generated_sha256'])].append(key)
            record['cells'].append(dict(label=label,full='blinded/'+a['full'],thumb='blinded/'+a['thumb'],
                dino=float(m['DINO2AFC']),seed=float(m['SEED']),object=float(m['ObjectF1']),blank=m['blank_proxy']=='True'))
            blinded['cells'].append(dict(label=blind_label[label],full=a['full'],thumb=a['thumb']))
            unblind.append(dict(query_id=key,blind_label=blind_label[label],method_label=label,
                stimulus_id=row['stimulus_id'],EEG_source_stimulus_id=row['EEG_source_stimulus_id'],
                target_image_path=score['target_image_path'],generated_path=row['generated_path'],scoring_target_path=m['score_target_path'],
                target_sha256=row['target_image_sha256'],generated_sha256=row['generated_sha256'],image_seed=row['image_seed']))
        blinded['cells'].sort(key=lambda c:c['label']);rows.append(record);blind_rows.append(blinded)
        if (index+1)%1000==0:print(json.dumps(dict(stage='offline_gallery_CPU',completed=index+1,expected=len(keys))),flush=True)
    for folder,data in [(out,dict(labels=labels,rows=rows,blinded=False)),
                        (blind,dict(labels=[chr(65+i) for i in range(len(labels))],rows=blind_rows,blinded=True))]:
        (folder/'data.js').write_text('window.GALLERY='+json.dumps(data,separators=(',',':'))+';\n')
        (folder/'index.html').write_text(app_page(folder==blind))
    with (out/'unblinding_and_pairing.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(unblind[0]));writer.writeheader();writer.writerows(unblind)
    with (blind/'human_ratings_blank.csv').open('x',newline='') as file:
        writer=csv.writer(file);writer.writerow(['rater_id','query_id','blind_label','target_object_fidelity_1_to_5','visible_artifacts_1_to_5','notes'])
    pages=(len(rows)+39)//40;links=[]
    for page in range(pages):
        body=f'<h1>全量缩略图 {page+1}/{pages}</h1><nav><a href="../index.html">交互检索</a> '
        if page:body+=f'<a href="page_{page:04d}.html">上一页</a> '
        if page+1<pages:body+=f'<a href="page_{page+2:04d}.html">下一页</a>'
        body+='</nav><table><tr><th>目标 / query</th>'+''.join('<th>'+html.escape(l)+'</th>' for l in labels)+'</tr>'
        for row in rows[page*40:(page+1)*40]:
            body+='<tr><td><code>'+html.escape(row['id'])+'</code><img loading="lazy" src="../'+row['target']+'"></td>'
            for cell in row['cells']:body+='<td><img loading="lazy" src="../'+cell['thumb']+'"><small>DINO '+str(round(cell['dino'],4))+' · SEED '+str(round(cell['seed'],4))+'</small></td>'
            body+='</tr>'
        (out/'pages'/f'page_{page+1:04d}.html').write_text(frame('完整缩略图分页',body+'</table>'))
        links.append(f'<li><a href="page_{page+1:04d}.html">{page*40+1}–{min((page+1)*40,len(rows))}</a></li>')
    (out/'pages/index.html').write_text(frame('全量分页目录','<h1>全量分页目录</h1><ul>'+''.join(links)+'</ul>'))
    report=dict(status='COMPLETED_FULL_OFFLINE_GALLERY',queries=len(keys),subjects=10,arms=labels,
        expected_generated_images=len(keys)*len(labels),verified_generated_images=len(unblind),
        missing_images=0,corrupt_images=0,duplicate_hash_images_beyond_first=sum(len(v)-1 for v in duplicates.values()),
        assets=len(assets),full_image_storage='Hard links to exact saved JPEG bytes; no second full-image byte copy',
        thumbnail_pages=pages,human_ratings_performed=False,blinded_materials_complete=True,
        sources=[dict(label=l,manifest_sha256=sha(m),metrics_sha256=sha(s)) for l,m,s in args.arm],
        builder_sha256=sha(__file__),seconds=time.monotonic()-started,
        limits=['Blinded means the separate blinded directory only; do not give raters the unblinding table or unblinded directory.',
            'A/B labels are reassigned within each query; no human evaluation has been conducted.',
            'Copy the complete blinded directory for portable offline rating; local C/D bulk copying is not authorized.',
            'Original event IDs are unavailable; pairing is to released metadata and image SHA receipts.'])
    (out/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
    (blind/'README.txt').write_text('Open index.html offline. Complete fixed-query materials; no method labels or automatic scores. A/B labels are specific to each query. Human ratings have not been collected.\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
