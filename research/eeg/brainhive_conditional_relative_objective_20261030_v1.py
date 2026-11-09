"""Classical conditional softmax control, not claimed as a new invention.

Gallery is a T-only training supervision tensor; labels and target galleries
must never become EEG-to-image inference inputs. Kappa is the frozen native
OOF encoder temperature. Division by kappa keeps the feature-gradient scale
comparable to the existing .1 cosine loss; no tuned mixture weight is added.
"""
import torch
import torch.nn.functional as F


def relative_evidence_loss(condition, unit_gallery, true_slot, kappa):
    assert condition.ndim == 2 and condition.shape[-1] == 1024
    assert unit_gallery.shape == (len(condition),10,1024)
    assert true_slot.shape == kappa.shape == (len(condition),)
    assert torch.all(kappa>0) and torch.isfinite(kappa).all()
    q=F.normalize(condition,dim=-1)
    scores=torch.einsum('bd,bkd->bk',q,unit_gallery)
    logits=kappa[:,None]*scores
    loss=.1*(torch.logsumexp(logits,dim=-1)-logits.gather(1,true_slot[:,None]).squeeze(1))/kappa
    assert torch.isfinite(loss).all()
    return loss
