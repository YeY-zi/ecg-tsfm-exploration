import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, context, horizon):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1,32,7,padding=3),nn.GELU(),
            nn.Conv1d(32,64,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),nn.Linear(64*16,horizon))
    def forward(self,x): return self.net(x[:,None,:])


class LSTM(nn.Module):
    def __init__(self, context, horizon):
        super().__init__()
        self.rnn = nn.LSTM(1,64,batch_first=True)
        self.head = nn.Linear(64,horizon)
    def forward(self,x): return self.head(self.rnn(x[:,:,None])[0][:,-1])


class PatchTST(nn.Module):
    """Univariate PatchTST-style forecasting adaptation, not official reproduction.

    Non-overlapping/overlapping patches, learned position embeddings, RevIN-like
    input normalization and flatten head. No pretrained weights.
    """
    def __init__(self, context, horizon, patch=16, stride=8):
        super().__init__()
        if context < patch: raise ValueError('context must be >= patch')
        self.patch,self.stride = patch,stride
        count = (context-patch)//stride+1
        self.embed = nn.Linear(patch,64)
        self.pos = nn.Parameter(torch.randn(1,count,64)*.02)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            64,4,128,dropout=.1,batch_first=True,activation='gelu'),2)
        self.head = nn.Linear(count*64,horizon)
    def forward(self,x):
        mean = x.mean(-1,keepdim=True).detach()
        std = (x.var(-1,keepdim=True,unbiased=False)+1e-5).sqrt().detach()
        z = ((x-mean)/std).unfold(-1,self.patch,self.stride)
        z = self.encoder(self.embed(z)+self.pos).flatten(1)
        return self.head(z)*std+mean


class Transformer(PatchTST):
    def __init__(self,context,horizon): super().__init__(context,horizon,patch=1,stride=1)


def build(name,context,horizon):
    return dict(cnn=CNN,lstm=LSTM,transformer=Transformer,patchtst=PatchTST)[name](context,horizon)
