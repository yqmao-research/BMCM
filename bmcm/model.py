import torch
import torch.nn as nn


def he_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)


class Encoder(nn.Module):
    def __init__(self, dim_in, dim_out, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim_out), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ClusterHead(nn.Module):
    def __init__(self, dim_in, n_clusters, hidden=100):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(dim_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(hidden, n_clusters), nn.Softmax(dim=1))

    def forward(self, z):
        h = self.hidden(z)
        return self.out(h), h


class IRE(nn.Module):
    def __init__(self, dim_in=4, hidden=14):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, r):
        return torch.sigmoid(self.net(r)).squeeze(-1)


class BMCMNet(nn.Module):
    def __init__(self, view_dims, n_clusters, fea_dim=256, ire_hidden=14, n_pairs=1):
        super().__init__()
        self.encoders = nn.ModuleList([Encoder(d, fea_dim) for d in view_dims])
        self.head = ClusterHead(fea_dim, n_clusters)
        self.apply(he_init)
        self.ire = IRE(4, ire_hidden)
        self.pair_logits = nn.Parameter(torch.zeros(n_pairs))

    def encode(self, xs):
        zs = [enc(x) for enc, x in zip(self.encoders, xs)]
        cs = [self.head(z)[0] for z in zs]
        return zs, cs

    def fuse(self, zs, coef):
        zf = sum(c * z for c, z in zip(coef, zs))
        cf, hf = self.head(zf)
        return zf, cf, hf
