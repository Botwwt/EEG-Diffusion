"""Protected fixed-real-gallery metrics with explicit stimulus-ID joins.

The gallery has unique stimulus IDs. Queries may repeat a stimulus across
subjects; their positive column is never inferred from the query row number.
"""
import torch
import torch.nn.functional as F

VERSION = 'fixed_real_gallery_explicit_stimulus_v1'


@torch.no_grad()
def fixed_gallery_metrics(query_features, gallery_features, query_stimulus_ids,
                          gallery_stimulus_ids, batch_size=256):
    assert query_features.ndim == gallery_features.ndim == 2
    assert query_features.shape[1] == gallery_features.shape[1]
    assert len(query_stimulus_ids) == len(query_features)
    assert len(gallery_stimulus_ids) == len(gallery_features) > 1
    assert len(set(gallery_stimulus_ids)) == len(gallery_stimulus_ids)
    assert torch.isfinite(query_features).all() and torch.isfinite(gallery_features).all()
    assert (query_features.float().norm(dim=-1) > 0).all()
    assert (gallery_features.float().norm(dim=-1) > 0).all()
    positions = {stimulus: i for i, stimulus in enumerate(gallery_stimulus_ids)}
    assert set(query_stimulus_ids) <= set(positions)
    device=query_features.device
    gallery=F.normalize(gallery_features.to(device=device,dtype=torch.float32),dim=-1)
    positive=torch.tensor([positions[s] for s in query_stimulus_ids],device=device)
    n=len(gallery);afc=[];top1=[];top5=[];correct_cosine=[]
    for start in range(0,len(query_features),batch_size):
        query=F.normalize(query_features[start:start+batch_size].float(),dim=-1)
        scores=query @ gallery.T
        correct=positive[start:start+len(query)]
        diagonal=scores.gather(1,correct[:,None])
        # Remove the positive image itself; retain half-credit for all ties.
        score=((scores < diagonal).sum(1) + .5*((scores == diagonal).sum(1)-1))/(n-1)
        candidates=scores.topk(min(5,n),dim=1).indices
        afc.append(score.cpu());top1.append((candidates[:,0]==correct).cpu())
        top5.append((candidates==correct[:,None]).any(1).cpu());correct_cosine.append(diagonal[:,0].cpu())
    return dict(two_afc=torch.cat(afc),top1=torch.cat(top1),top5=torch.cat(top5),
                correct_cosine=torch.cat(correct_cosine),positive_gallery_index=positive.cpu(),
                version=VERSION,gallery_images=n)
