import math
import torch
import torch.nn as nn

def attention(q : torch.Tensor , k : torch.Tensor , v : torch.Tensor , mask=None, dropout=None):
    nbatch, nseq, dk = q.size()
    scale = 1 / math.sqrt(dk)    
    logit = torch.matmul(q, k.transpose(-2,-1)) * scale
    if mask is not None:
        logit = logit.masked_fill(mask == 0, -1e9)
    scores = logit.softmax(dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    attention = torch.matmul(scores, v)
    return attention

class MultiHeadedAttention(nn.Module):
    def __init__(self, nhead : int , dmodel : int , dropout : float ):
        super(MultiHeadedAttention, self).__init__()
        assert dmodel % nhead == 0
        self.nhead = nhead
        self.dmodel = dmodel
        self.linear_q = nn.Linear(dmodel, dmodel)
        self.linear_k = nn.Linear(dmodel, dmodel)
        self.linear_v = nn.Linear(dmodel, dmodel)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(dmodel, dmodel)
        self.attn = None

    def forward(self, q : torch.Tensor , k : torch.Tensor, v : torch.Tensor, mask = None ) -> torch.Tensor :
        '''
            computes dot-product multi-headed attention over (q,k,v)
            q : [nbatch, nseq, dk]
            k : [nbatch, nseq, dk]
            v : [nbatch, nseq, dk]
        '''
        nbatch, nseq, dmodel = q.size()
        dk = dmodel // self.nhead
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # reshape projections to [nbatch*nhead, nseq, dk]
        q = self._reshape_to_mh(q)
        k = self._reshape_to_mh(k)
        v = self._reshape_to_mh(v)

        # calculate dot-product attention
        y = attention(q, k, v, mask)
        y = self._reshape_from_mh(y)

        # proj layer
        out = self.linear_out(y)
        return out

    # helpers
    def _reshape_to_mh(self, x : torch.Tensor ) -> torch.Tensor :
        nbatch, nseq, dmodel = x.size()
        dk = dmodel // self.nhead
        x = x.view(nbatch, nseq, self.nhead, dk)
        x = x.permute(0,2,1,3)
        x = x.view(nbatch*self.nhead, nseq, dk)
        return x

if __name__ == '__main__':
    mh = MultiHeadedAttention(8, 512, 0.5)