"""Rebuild static figures and the report from released aggregate results.

Standard-library only. No model, image generation or new inference is performed.
"""
from pathlib import Path
import argparse
import csv
import html
import io
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))


def svg_chart(rows, *, maximum, title, suffix='', dual=False):
    width, height, left, right = 540, 233, 154, 480
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#333} .muted{fill:#66706e;font-size:11px}</style>']
    for i in range(5):
        value=maximum*i/4;x=left+(right-left)*i/4
        lines.append(f'<line x1="{x}" y1="28" x2="{x}" y2="183" stroke="#e6e9e8"/>')
        label=f'{value:g}{suffix}'
        lines.append(f'<text x="{x}" y="207" text-anchor="middle" class="muted">{label}</text>')
    for i,(label,values) in enumerate(rows):
        y=57+i*57
        lines.append(f'<text x="{left-12}" y="{y+5}" text-anchor="end">{html.escape(label)}</text>')
        end=left+(right-left)*sum(values)/len(values)/maximum
        lines.append(f'<line x1="{left}" y1="{y}" x2="{end}" y2="{y}" stroke="#c8d9d5" stroke-width="8"/>')
        for j,value in enumerate(values):
            x=left+(right-left)*value/maximum
            offset=(-5 if j==0 else 5) if dual else 0
            color='#287d75' if j==0 else '#8a5e9a'
            lines.append(f'<circle cx="{x}" cy="{y+offset}" r="4" fill="{color}"/>')
        val=' / '.join(f'{v:.1f}{suffix}' if dual else f'{v:.4f}' for v in values)
        lines.append(f'<text x="{max(end+11, left+14)}" y="{y+4}" style="font-size:12px">{val}</text>')
    lines.append(f'<text x="{left}" y="228" class="muted">{html.escape(title)}</text></svg>')
    return '\n'.join(lines)+'\n'


def artifacts():
    summary=load('summary.json');pairing=load('image_pairing.json')
    order=[('final_fixed','Final, fixed'),('final_joint','Final + joint head'),('middle12_joint','Layer 12 + joint head')]
    retrieval=[]
    for method,label in order:
        values=[100*r['top1'] for r in sorted(summary['retrieval']['rows'],key=lambda x:x['seed']) if r['method']==method]
        retrieval.append((label,values))
    arms={r['arm']:r for r in pairing['final']}
    image=[(label,[arms[key]['DINO_2AFC']]) for key,label in [
        ('native_full_target','Correct EEG'),('constant_training_mean','Training mean'),('native_full_target_cyclic_shift','Shuffled EEG')]]
    comparisons=summary['v_parameterization']['comparisons']
    vt=''.join(f'<tr><td>{html.escape(r["comparison"])}</td><td>{r["difference"]:+.4f}</td><td>[{r["ci95"][0]:+.4f}, {r["ci95"][1]:+.4f}]</td></tr>' for r in comparisons)
    pair=next(c for c in pairing['paired_comparisons'] if c['control']=='native_full_target_cyclic_shift')['metrics']['DINO_2AFC']
    replacements={
        '{{V_TABLE}}':vt,
        '{{PAIR_DELTA}}':f'{pair["mean_difference"]:+.4f}',
        '{{PAIR_CI}}':f'[{pair["paired_image_bootstrap95"][0]:+.4f}, {pair["paired_image_bootstrap95"][1]:+.4f}]',
    }
    page=(ROOT/'scripts/report.template.html').read_text(encoding='utf-8')
    for key,value in replacements.items():page=page.replace(key,value)
    # GitHub Pages serves docs/ as its root; repository evidence is linked to
    # GitHub, rather than to nonexistent files above the deployed site root.
    base='https://github.com/Botwwt/EEG-Diffusion/blob/main/'
    page=re.sub(r'href="\.\./([^"]+)"',lambda m:'href="'+base+m.group(1)+'"',page)
    for name in ['EXPERIMENTS.md','REPRODUCIBILITY.md']:
        page=page.replace('href="'+name+'"','href="'+base+'docs/'+name+'"')
    if '{{' in page:raise ValueError('Unresolved report template value')
    out={ROOT/'docs/index.html':page,
         ROOT/'docs/assets/retrieval.svg':svg_chart(retrieval,maximum=100,title='EEG retrieval Top-1 (%)',suffix='%',dual=True),
         ROOT/'docs/assets/image_pairing.svg':svg_chart(image,maximum=1,title='Generated-image DINOv3 2AFC')}
    buffer=io.StringIO(newline='');writer=csv.writer(buffer,lineterminator='\n')
    writer.writerow(['experiment','arm','training_seed','metric','value','unit'])
    for row in summary['retrieval']['rows']:
        writer.writerow(['retrieval_mean80',row['method'],row['seed'],'Top-1',row['top1'],'proportion'])
    for row in pairing['final']:
        writer.writerow(['image_pairing_32',row['arm'],'fixed single seed','DINOv3_2AFC',row['DINO_2AFC'],'proportion'])
    out[ROOT/'results/figure_data.csv']=buffer.getvalue()
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    failures=[]
    for path,content in artifacts().items():
        if a.check:
            if not path.exists() or path.read_text(encoding='utf-8')!=content:failures.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf-8',newline='\n')
    if failures:raise SystemExit('Stale report artifacts: '+', '.join(failures))
    print('Report artifacts match the released aggregates.' if a.check else 'Built HTML, two SVG figures and figure_data.csv.')


if __name__=='__main__':main()

