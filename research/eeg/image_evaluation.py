"""Frozen historical DINOv3 features and exact public-code SEED pair algebra."""
from semantic_evaluator import *
import hashlib
import torch.nn.functional as F

DINO_WEIGHT=Path('/root/mced_brain_dinov3_s_v1/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
DINO_SHA256='08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'


def dino_model():
    with DINO_WEIGHT.open('rb') as file:assert hashlib.file_digest(file,'sha256').hexdigest()==DINO_SHA256
    sys.path.insert(0,'/root/mced_representation_sources_v2/dinov3')
    from dinov3.hub.backbones import dinov3_vits16
    model=dinov3_vits16(pretrained=False).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(DINO_WEIGHT,weights_only=True,map_location='cpu'),strict=True)
    return model


@torch.inference_mode()
def dino_features(paths,model,batch_size=32,real_images=False):
    output=[]
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    for start in range(0,len(paths),batch_size):
        pixels=[]
        for path in paths[start:start+batch_size]:
            with Image.open(path) as im:a=torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1).float()[None]/255
            # Preserve historical routes: real image CPU resize; generated image CUDA resize.
            if not real_images:a=a.cuda()
            pixels.append(F.interpolate(a,(256,256),mode='bilinear',align_corners=False,antialias=True))
        x=torch.cat(pixels).cuda()
        patch,cls=model.get_intermediate_layers((x-mean)/std,n=[11],return_class_token=True,norm=True)[0]
        y=F.normalize(torch.cat([cls,patch.mean(1)],-1),dim=-1)
        assert torch.isfinite(y).all();output.append(y.cpu())
    return torch.cat(output)


def detected_categories(row,threshold):
    cats=[]
    for label,score,box in zip(row['labels'],row['scores'],row['bboxes']):
        if score<threshold:break
        # Python double arithmetic matches the official JSON implementation.
        box=[float(x) for x in box]
        if (box[2]-box[0])*(box[3]-box[1])<50:continue
        if int(label) not in cats:cats.append(int(label))
    return cats


def object_f1_pair(reconstruction,target):
    recall=[];precision=[]
    for threshold in np.arange(.05,.95,.05):
        rc=detected_categories(reconstruction,float(threshold));gt=detected_categories(target,float(threshold))
        overlap=len(set(rc)&set(gt))
        if gt:recall.append(overlap/len(gt))
        if rc:precision.append(overlap/len(rc))
        if not gt and not rc:break
    r=sum(recall)/len(recall) if recall else 1.
    p=sum(precision)/len(precision) if precision else 1.
    return 2*p*r/(p+r) if p+r else 0.


def quality_statistics(path):
    with Image.open(path) as im:a=np.asarray(im.convert('RGB')).astype(np.float32)/255.
    return dict(pixel_std=float(a.std()),near_black_fraction=float((a<1/255).mean()),
                near_white_fraction=float((a>254/255).mean()),
                blank_proxy=bool(a.std()<.01),width=a.shape[1],height=a.shape[0])
