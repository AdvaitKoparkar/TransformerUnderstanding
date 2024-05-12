import math
import torch
import torch.nn as nn
from copy import copy

def clones(x : nn.Module, N : int ) -> nn.ModuleList :
    xs = nn.ModuleList([copy(x) for _ in range(N)])
    return xs

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

def create_mask(size):
    mask = torch.triu((1,size,size), diagonal=1).type(torch.uint8)
    return mask

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

        if mask is not None:
            mask = mask.repeat(self.nhead,1,1)

        # calculate dot-product attention
        y = attention(q, k, v, mask, self.dropout)
        y = self._reshape_from_mh(y)

        # proj layer
        out = self.linear_out(y)
        return out

    # helpers
    def _reshape_to_mh(self, x : torch.Tensor ) -> torch.Tensor :
        nbatch, nseq, dmodel = x.size()
        dk = dmodel // self.nhead
        x = x.view(nbatch, nseq, self.nhead, dk)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(nbatch*self.nhead, nseq, dk)
        return x

    def _reshape_from_mh(self, x : torch.Tensor) -> torch.Tensor :
        _, nseq, dk = x.size()
        dmodel = self.dmodel
        nhead = self.nhead
        x = x.view(-1, nhead, nseq, dk)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(-1, nseq, nhead * dk)
        return x

class LayerNorm(nn.Module):
    def __init__(self, features : int , eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_1 = nn.Parameter(torch.zeros(features))
        self.a_2 = nn.Parameter(torch.ones(features))
        self.eps = eps
    
    def forward(self, x : torch.Tensor ) -> torch.Tensor :
        # nbatch, nseq, dmodel = x.size()
        return self.a_2 * ((x - x.mean(dim=-1, keepdims=True)) / (x.std(dim=-1, keepdims=True) + self.eps)) + self.a_1

class EncoderLayer(nn.Module):
    def __init__(self, nhead : int ,  dmodel : int , dropout : float ):
        super(EncoderLayer, self).__init__()
        self.nhead = nhead
        self.dmodel = dmodel
        self.self_attn = MultiHeadedAttention(nhead, dmodel, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(dmodel)
        self.feedforward = nn.Linear(dmodel, dmodel)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(dmodel)

    def forward(self, x : torch.Tensor , mask : torch.Tensor ):
        x = x + self.dropout1(self.norm1(self.self_attn(x,x,x,mask)))
        x = x + self.dropout2(self.norm2(self.feedforward(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, nhead : int, dmodel : int, dropout : float):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(nhead, dmodel, dropout)
        self.norm1 = LayerNorm(dmodel)
        self.dropout1 = nn.Dropout(dmodel)

        self.cross_attn = MultiHeadedAttention(nhead, dmodel, dropout)
        self.norm2 = LayerNorm(dmodel)
        self.dropout2 = nn.Dropout(dropout)

        self.feedforward = nn.Linear(dmodel, dmodel)
        self.norm3 = LayerNorm(dmodel)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x : torch.Tensor , memory : torch.Tensor, src_mask : torch.Tensor, tgt_mask : torch.Tensor ) -> torch.Tensor :
        x = x + self.dropout1(self.norm1(self.self_attn(x, x, x, tgt_mask)))
        x = x + self.dropout2(self.norm2(self.cross_attn(x, memory, memory, src_mask)))
        x = x + self.dropout3(self.norm3(self.feedforward(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, layer : nn.Module, N : int ):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.dmodel)
    
    def forward(self, x : torch.Tensor ) -> torch.Tensor :
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer : nn.Module, N : int ):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.dmodel)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

if __name__ == '__main__':
    nbatch, nseq, dmodel = 16, 48, 512
    nhead = 8
    dropout = 0.5

    # MH
    # mh = MultiHeadedAttention(nhead, dmodel, dropout=0.5)
    # q  = torch.randn(nbatch, nseq, dmodel)
    # k  = torch.randn_like(q)
    # v  = torch.randn_like(q)

    # Encoder layer
    # enc = EncoderLayer(nhead, dmodel, dropout)
    # x = torch.randn(nbatch, nseq, dmodel)
    # mask = torch.ones(nbatch, nseq, nseq)
    # print(enc(x, mask).shape)

    # Encoder
    # enc = Encoder(EncoderLayer(nhead, dmodel, dropout), 6)
    # print(enc)

    # Decoder layer
    dec = DecoderLayer(nhead, dmodel, dropout)
    nout = 16
    x = torch.randn(nbatch, nseq, dmodel)
    
    mem = torch.randn(nbatch, nseq, dmodel)
    src_mask = torch.ones(nbatch, dmodel, dmodel)

