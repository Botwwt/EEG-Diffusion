"""Register additional author metrics before seeing native image scores.

This registers a not-yet-implemented evaluator. It does not replace or alter the
campaign's protected DINOv3/SEED metrics or claim paper-comparable SOTA.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    source=ROOT/'reproduction_sources/BrainHIVE_f06db69f'
    out=ROOT/'configs/BrainHIVE_original_image_metrics_registration_v1.json'
    assert not out.exists()
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    plan=dict(status='REGISTERED_EVALUATOR_NOT_YET_IMPLEMENTED_ASSETS_INCOMPLETE',
        implementation_entrypoint=None,original_source_commit='f06db69fcc4a029439e369c5c3dba6099e57560e',
        sources={str(p.relative_to(ROOT)):digest(p) for p in [source/'main/utils_eval.py',source/'build_reconstruction.py']},
        common_DINO_SEED_definitions_unchanged=True,native_image_scores_seen_at_registration=False,
        image_inputs='Existing saved lossless images only; never regenerate or select a seed',
        arms=['native_R80_correct_v1','native_R80_permuted_v1','native_R80_missing_v1'],
        subjects=10,queries_per_arm=2000,real_gallery_per_subject=200,images_per_query=1,
        initial_preprocessing='PIL RGB -> Resize512 BILINEAR -> CenterCrop512 -> ToTensor [0,1], as save_output source branch',
        metrics={
            'PixCorr425':'Source bilinear Tensor Resize425 then flattened Pearson, per paired query',
            'SSIM425':'Source Resize425/rgb2gray/Gaussian sigma1.5/population covariance/L1; verify actual scikit-image multichannel keyword behavior',
            'Alex2_2AFC':'Original AlexNet features.4, Tensor Resize256, ImageNet normalization, Pearson identification',
            'Alex5_2AFC':'Original AlexNet features.11, same preprocessing/direction',
            'Inception_2AFC':'InceptionV3 avgpool, Tensor Resize342, ImageNet normalization, Pearson identification',
            'CLIP_L14_2AFC':'OpenAI ViT-L/14 encode_image, Tensor Resize224 bilinear and CLIP normalization, Pearson identification',
            'EffNet_distance':'EfficientNetB1 avgpool, Resize255, original correlation DISTANCE (lower better)',
            'SwAV_distance':'SwAV ResNet50 avgpool, Resize224, original correlation DISTANCE (lower better)'},
        twoAFC_direction='For generated query j, compare corr(real_j,gen_j) against all other fixed real_i; strict inequality as source, N-1 negatives',
        required_outputs=['Every query with generated/target identity SHA','Equal-weight subject summaries','Original-image self check','Fixed deranged pairing check','Actual loader/source/weight receipts and runtime versions'],
        source_assets={
            'AlexNet':dict(status='NOT_ACQUIRED',url='https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',sha256_prefix='7be5be79'),
            'CLIP_L14':dict(status='NOT_ACQUIRED_SOURCE_PACKAGE_NOT_YET_PINNED',url='https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt',sha256='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'),
            'Inception':dict(status='EXISTING_FILE_FOUND_NOT_YET_REVERIFIED',path='/root/rime_var_20261015/assets/evaluators/inception_v3_google-0cc3c7bd.pth'),
            'SwAV':dict(status='EXISTING_FILE_FOUND_NOT_YET_REVERIFIED',path='/root/rime_var_20261015/assets/evaluators/swav_800ep_pretrain.pth.tar'),
            'EffNet':dict(status='EXISTING_VERIFIED_SEED_ASSET_RECHECK_BEFORE_NATIVE_USE',path='/dev/shm/eeg_sota_autoresearch_20261018_v1/semantic_evaluation/efficientnet_b1-c27df63c.pth')},
        public_primary_references=['https://raw.githubusercontent.com/openai/CLIP/main/clip/clip.py',
            'https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/alexnet.py',
            'https://raw.githubusercontent.com/facebookresearch/swav/main/hubconf.py'],
        missing_asset_policy='Record unavailable metrics; continue independent checks. Never substitute CLIP B32 for L14 or relabel protected metrics.',
        GPU_policy='Use the same single-GPU wrapper/ledger and a finite job, after current fixed image/evaluation queue completes',
        limitations=['Author ten images and multiple seeds replaced with user-required one seed/one image; no exact paper repeated-mean claim.',
            'CPU/GPU library versions differ from source requirements; numerical compatibility must be checked and recorded.',
            'Previously exposed public benchmark is not independent H.',
            'An additional native metric table cannot replace full matched component improvement, quality and EEG interventions.'])
    out.write_text(json.dumps(plan,indent=2)+'\n')
    print('ORIGINAL_METRICS_REGISTERED_NOT_EXECUTED',digest(out))


if __name__=='__main__':main()
