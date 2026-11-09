"""Declare inherited sample positions, or render only after full terminal success."""
import base64,datetime,hashlib,io,json,os,resource,sys,time,zipfile
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
q=json.loads(sys.stdin.readline());assert q['host']=='old' and q['mode'] in ('declare','complete')
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=digest)
def checked(item):assert ref(item['path'])==item;return Path(item['path'])
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def live(pid,ticks):
    try:f=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return f[0] not in ('Z','X') and int(f[19])==ticks
out=Path(q['panel_root']);assert ref(__file__)['sha256']==q['renderer_SHA256']
study=read(checked(q['study']));code=Path(q['study']['path']).parent;root=Path(study['output_root']);launch=read(code/'actual_launch.json')
assert study['training_seed']==1729 and not study['new_training_allowed']
sys.path.insert(0,str(root.parent/'HIVE_Euclidean_F_comparison_recovery_code_20261030_v3'))
from brainhive_Euclidean_F_reference_bindings_CPU_20261030_v3 import reference_bindings
reference_CSVs,reference_manifests=reference_bindings(study)
assert study['status']=='REGISTERED_SINGLE1729_EUCLIDEAN_F_POSTFIT_AMENDED_NUMERIC_VERIFICATION_v2'
assert not study['original_fullT_FP64_guard_passed'] and study['numeric_verification_amended_after_original_failure']
old=read(checked(q['inherited_policy']));positions=old['policy']
assert positions['subjects']==['sub-01','sub-05','sub-10'] and positions['array_rows']==[0,40,80,119,159,199]
policy=dict(subjects=positions['subjects'],array_rows=positions['array_rows'],columns=['target','parent','ordinary','bounded_F','Euclidean_F'],inherited_positions_not_selected_using_new_images=True,source_policy=q['inherited_policy'],query_subset_not_accuracy_estimate=True,normal_inference_target_access=False,original_images_unchanged=True)
if q['mode']=='declare':
    assert live(launch['controller_PID'],launch['start_ticks'])
    absent={name:not (root/name).exists() for name in ('R80/evaluation/scores/per_query.csv','R80/evaluation/scores/summary.json','R80/evaluation/original_paper_metrics/per_query.csv','joint_comparison/summary.json')}
    assert all(absent.values()),'Cannot label a new declaration pre-effect after effect files exist'
    write(out/'fixed_panel_policy.json',dict(status='INHERITED_FIXED_POSITIONS_BOUND_BEFORE_NEW_EUCLIDEAN_EFFECT_SCORES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),study=q['study'],policy=policy,renderer=ref(__file__)))
    result=dict(status='PASS_PRE_EFFECT_INHERITED_18QUERY_QUALITATIVE_DECLARATION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),study=q['study'],policy=ref(out/'fixed_panel_policy.json'),renderer=ref(__file__),new_effect_score_files_absent=absent,exact_main_live=True,controller_PID=launch['controller_PID'],start_ticks=launch['start_ticks'],new_control_images_read=0,new_GPU_calls=0,new_training_updates=0)
    write(out/'pre_effect_declaration.json',result);print(json.dumps(result));raise SystemExit(0)
proof=read(out/'pre_effect_declaration.json');assert proof['status']=='PASS_PRE_EFFECT_INHERITED_18QUERY_QUALITATIVE_DECLARATION'
assert ref(out/'fixed_panel_policy.json')==proof['policy']==q['policy_reference'] and proof['renderer']==q['original_renderer'] and ref(q['original_renderer']['path'])==q['original_renderer']
assert read(out/'fixed_panel_policy.json')['policy']==policy and all(proof['new_effect_score_files_absent'].values())
assert not live(launch['controller_PID'],launch['start_ticks'])
assert (root/'failure.json').exists() and not (root/'completion.json').exists()
done=read(root/'comparison_recovery_v3/completion.json');assert done['status']=='COMPLETE_CPU_ONLY_SCHEMA_RECOVERY_ALL2000_ALL16_THREE_FIXED_COMPARISONS_v3' and done['study']==q['study']
assert not done['original_v2_lifecycle_success'] and ref(root/'failure.json')==done['original_failure_unchanged']
assert ref(done['summary']['path'])==done['summary']
output=out/'complete_recovery_v2';assert not output.exists();output.mkdir();began=time.monotonic()
from PIL import Image,ImageDraw,ImageFont,ImageOps
tables={};pins=[]
for role in policy['columns'][1:]:
    item=reference_manifests[role] if role in reference_manifests else read(root/'R80/loss_control/artifact_audit.json')['manifest']
    rows=[json.loads(x) for x in checked(item).read_text().splitlines()];assert len(rows)==2000
    tables[role]={(x['subject'],int(x['array_row'])):x for x in rows};assert len(tables[role])==2000;pins.append(item)
spec=read(code/'Euclidean_F_R80_protocol.json');gallery={r['stimulus_id']:r for r in map(json.loads,checked(spec['gallery_manifest']).read_text().splitlines())};assert len(gallery)==200
font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';large=ImageFont.truetype(font_path,20);small=ImageFont.truetype(font_path,14)
tile=240;gap=12;top=100;rowheight=276;width=5*tile+6*gap;height=top+6*rowheight+12
pages=[];records=[]
for sub in policy['subjects']:
    canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),'R80 fixed-index geometry panel | '+sub+' | training seed 1729',fill='black',font=large)
    draw.text((12,37),'Fixed subset; numerical protocol amended before effects. Use full paired metrics for conclusions.',fill=(40,40,40),font=small)
    for j,label in enumerate(['Target (display only)','Native','Ordinary','F bounded','F Euclidean']):draw.text((12+j*(tile+gap),73),label,fill='black',font=small)
    for i,row in enumerate(policy['array_rows']):
        reference=tables['parent'][sub,row];stimulus=reference['stimulus_id'];y=top+i*rowheight
        draw.text((12,y),f'row {row:03d} | stimulus {stimulus}',fill='black',font=small)
        for j,role in enumerate(policy['columns']):
            item=gallery[stimulus] if role=='target' else tables[role][sub,row]
            if role=='target':path=Path(item['image_path']);expected=item['image_sha256']
            else:
                assert item['query_id']==reference['query_id'] and item['image_seed']==reference['image_seed'] and item['stimulus_id']==stimulus
                path=Path(item['generated_path']);expected=item['generated_sha256']
            source=ref(path);assert source['sha256']==expected
            with Image.open(path) as image:
                pixels=image.convert('RGB');pixel_SHA=hashlib.sha256(pixels.tobytes()).hexdigest();thumbnail=ImageOps.contain(pixels,(tile,tile),Image.Resampling.LANCZOS)
                if role!='target' and 'generated_pixel_sha256' in item:assert pixel_SHA==item['generated_pixel_sha256']
                x=12+j*(tile+gap);canvas.paste(thumbnail,(x+(tile-thumbnail.width)//2,y+25+(tile-thumbnail.height)//2));draw.rectangle((x,y+25,x+tile,y+25+tile),outline=(190,190,190))
            records.append(dict(subject=sub,array_row=row,query_id=reference['query_id'],role=role,source=source,decoded_pixel_SHA256=pixel_SHA))
    path=output/(sub+'_fixed_geometry.png');canvas.save(path);pages.append(ref(path))
assert len(records)==90 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<512<<20
result=dict(status='COMPLETE_INHERITED_FIXED_18QUERY_GEOMETRY_QUALITATIVE_PANEL',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),study=q['study'],policy=proof['policy'],pages=pages,source_images=records,source_manifests=pins,new_control_images_read=18,metrics_read=0,new_GPU_calls=0,new_training_updates=0,query_subset_not_accuracy_estimate=True,original_fullT_FP64_guard_passed=False,original_v2_lifecycle_success=False,renderer=ref(__file__),original_renderer=proof['renderer'],recovery_completion=ref(root/'comparison_recovery_v3/completion.json'),seconds=time.monotonic()-began)
write(output/'completion.json',result);files={'fixed_panel_policy.json':out/'fixed_panel_policy.json','pre_effect_declaration.json':out/'pre_effect_declaration.json','completion.json':output/'completion.json'};files.update({Path(x['path']).name:Path(x['path']) for x in pages});b=io.BytesIO();manifest={}
with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
    for name,path in files.items():
        raw=path.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),remote_path=str(path));z.writestr(name,raw)
assert len(b.getvalue())<12<<20
print(json.dumps(dict(status=result['status'],files=len(files),manifest=manifest,archive_SHA256=hashlib.sha256(b.getvalue()).hexdigest(),archive_base64=base64.b64encode(b.getvalue()).decode())))
