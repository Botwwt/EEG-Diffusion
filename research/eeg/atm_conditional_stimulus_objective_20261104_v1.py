"""Known conditional contrastive objective, explicitly a diagnostic baseline."""
import numpy as np

def numpy_loss_gradient(logits,arm,group_size=10):
    s=np.asarray(logits,dtype=np.float64);n=len(s)
    assert s.shape==(n,n) and n%group_size==0 and arm in ('nce_continued','conditional','conditional_wrong')
    i=np.arange(n);same=i[:,None]//group_size==i[None,:]//group_size
    def ce(matrix,labels,mask=None):
        z=matrix.copy()
        if mask is not None:z=np.where(mask,z,-np.inf)
        mx=z.max(1,keepdims=True);p=np.exp(z-mx);p/=p.sum(1,keepdims=True)
        loss=np.mean(mx[:,0]+np.log(np.exp(z-mx).sum(1))-matrix[i,labels])
        grad=p;grad[i,labels]-=1;return loss,grad/n
    row,rg=ce(s,i);col,cg=ce(s.T,i);global_loss=(row+col)/2;global_grad=(rg+cg.T)/2
    correct_row,fr=ce(s,i,same);correct_col,fc=ce(s.T,i,same)
    correct_fine=(correct_row+correct_col)/2
    shift=1 if arm=='conditional_wrong' else 0
    labels=i//group_size*group_size+(i%group_size+shift)%group_size
    inverse=i//group_size*group_size+(i%group_size-shift)%group_size
    row,rg=ce(s,labels,same);col,cg=ce(s.T,inverse,same);fine=(row+col)/2;fine_grad=(rg+cg.T)/2
    weight=0 if arm=='nce_continued' else 1
    return dict(loss=global_loss+weight*fine,gradient=global_grad+weight*fine_grad,
                global_loss=global_loss,correct_fine_loss=correct_fine,supervised_fine_loss=fine,coarse_loss=global_loss-correct_fine)

def torch_loss(logits,arm):
    import torch
    assert arm in ('nce_continued','conditional','conditional_wrong') and len(logits)%10==0
    n=len(logits);i=torch.arange(n,device=logits.device);same=i[:,None]//10==i[None,:]//10
    g=.5*((torch.logsumexp(logits,1)-logits[i,i]).mean()+(torch.logsumexp(logits,0)-logits[i,i]).mean())
    masked=logits.masked_fill(~same,-torch.inf)
    fine_correct=.5*((torch.logsumexp(masked,1)-logits[i,i]).mean()+(torch.logsumexp(masked,0)-logits[i,i]).mean())
    shift=1 if arm=='conditional_wrong' else 0
    target=i//10*10+(i%10+shift)%10;inverse=i//10*10+(i%10-shift)%10
    fine=.5*((torch.logsumexp(masked,1)-logits[i,target]).mean()+(torch.logsumexp(masked,0)-logits[inverse,i]).mean())
    loss=g+(0 if arm=='nce_continued' else 1)*fine
    return loss,dict(global_loss=g,correct_fine_loss=fine_correct,supervised_fine_loss=fine,coarse_loss=g-fine_correct)
