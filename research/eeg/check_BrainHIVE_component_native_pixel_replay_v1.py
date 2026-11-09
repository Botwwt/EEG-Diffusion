"""One predeclared existing native query: exact new-loader pixel replay, no scoring."""
import hashlib
import json
import os
import time

from brainhive_native_models_v1 import *
from brainhive_component_generator_v1 import load_native
from brainhive_bounded_component_v1 import BoundedConditionAdapter


def main():
    assert b'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    out=WORK/'audit/component_native_pixel_replay_v1.json'
    assert not out.exists()
    manifest=WORK/'images/native_R80_correct_v1/inference_manifest.jsonl'
    # This first manifest row is fixed before seeing the replay, never selected
    # by a metric or appearance. No target image/visual feature is loaded.
    row=json.loads(manifest.read_text().splitlines()[0])
    folder=WORK/'runs/published_source_dropout_v2'/row['subject']
    cache=torch.load(folder/'conditions.pt',weights_only=True,map_location='cpu')
    assert cache['model_sha256']==sha(folder/'brain_final.pt')
    index=[int(x) for x in cache['array_rows']].index(row['array_row'])
    c=cache['condition'][index:index+1]
    assert hashlib.sha256(c.contiguous().numpy().tobytes()).hexdigest()==row['condition_sha256']
    state=torch.load(folder/'brain_final.pt',weights_only=True,map_location='cpu')['model']
    w=state['block.layers.0.norm.weight'].cuda()[None]
    b=state['block.layers.0.norm.bias'].cuda()[None]
    pipe,adapter,projector,prompts,host=load_native(training=False)
    started=time.monotonic()
    with torch.inference_mode():
        c=c.cuda()
        for risk in [False,True]:
            model=BoundedConditionAdapter(predict_risk=risk).cuda().eval()
            changed,_=model(c,w,b)
            assert torch.equal(changed,c)
        tokens=projector(adapter(c.half()))
        image=pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(row['image_seed']),
            prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],
            cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,
            guidance_scale=0.,output_type='pil').images[0].convert('RGB')
    import numpy as np
    from PIL import Image
    assert sha(row['generated_path'])==row['generated_sha256']
    with Image.open(row['generated_path']) as original:expected=np.asarray(original.convert('RGB'))
    actual=np.asarray(image)
    same=np.array_equal(actual,expected)
    receipt=dict(status='EXACT_NATIVE_FIRST_ROW_PIXEL_REPLAY_PASSED' if same else 'NATIVE_PIXEL_REPLAY_FAILED',
        query_id=row['query_id'],original_manifest_sha256=sha(manifest),original_image_sha256=row['generated_sha256'],
        replay_RGB_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        original_RGB_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        max_absolute_channel_difference=int(np.abs(actual.astype('int16')-expected.astype('int16')).max()),
        zero_initialized_ordinary_and_risk_exact_input_equality=True,host=host,
        diagnostic_source_sha256=sha(__file__),queries=1,seconds=time.monotonic()-started,
        target_or_class_or_visual_features_read=False,no_score_or_checkpoint_selection=True)
    write(out,receipt)
    assert same,receipt
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
