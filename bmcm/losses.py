import math

import torch
import torch.nn.functional as F

EPS = 1e-9


def sq_cdist(x, y):
    d = (x * x).sum(1, keepdim=True) - 2 * x @ y.t() + (y * y).sum(1, keepdim=True).t()
    return F.relu(d)


def ddc_kernel(x, rel_sigma=0.15, min_sigma=EPS):
    d = sq_cdist(x, x)
    sigma2 = (rel_sigma * torch.median(d)).detach()
    sigma2 = torch.clamp(sigma2, min=min_sigma)
    return torch.exp(-d / (2 * sigma2))


def feature_contrast(zu, zv, fused_pred, tau, neg_ratio=0.5):
    b = zu.size(0)
    n = 2 * b
    z = torch.cat([zu, zv], 0)
    sim = z @ z.t() / tau
    pos = torch.cat([torch.diag(sim, b), torch.diag(sim, -b)]).reshape(n, 1)
    k = fused_pred.size(1)
    pl = fused_pred.detach().argmax(1).repeat(2)
    eye = torch.eye(k, device=z.device)
    weights = (1 - eye[pl])[:, pl].t()
    empty = weights.sum(1) == 0
    if bool(empty.any()):
        fallback = 1 - torch.eye(n, device=z.device)
        weights = torch.where(empty.unsqueeze(1), fallback, weights)
    neg_idx = torch.multinomial(weights, int(neg_ratio * n), replacement=True)
    neg = sim[torch.arange(n, device=z.device).reshape(-1, 1), neg_idx]
    logits = torch.cat([pos, neg], 1)
    target = torch.zeros(n, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, target, reduction="sum") / n


def marginal_entropy_gap(c):
    pi = c.mean(0)
    return math.log(c.size(1)) + (pi * torch.log(pi.clamp_min(1e-12))).sum()


def cluster_contrast(cu, cv, tau_c):
    k = cu.size(1)
    r = F.normalize(torch.cat([cu.t(), cv.t()], 0), dim=1)
    sim = r @ r.t() / tau_c
    sim = sim.masked_fill(torch.eye(2 * k, dtype=torch.bool, device=sim.device), float("-inf"))
    target = torch.cat([torch.arange(k, 2 * k), torch.arange(0, k)]).to(sim.device)
    l_con = F.cross_entropy(sim, target)
    return l_con + marginal_entropy_gap(cu) + marginal_entropy_gap(cv)


def _triu_sum(x):
    return torch.triu(x, diagonal=1).sum()


def _d_cs(kmat, a, n_clusters):
    nom = a.t() @ kmat @ a
    dnom2 = torch.diagonal(nom).unsqueeze(-1) @ torch.diagonal(nom).unsqueeze(0)
    nom = torch.where(nom < EPS, nom.new_tensor(EPS), nom)
    dnom2 = torch.where(dnom2 < EPS ** 2, dnom2.new_tensor(EPS ** 2), dnom2)
    return 2 / (n_clusters * (n_clusters - 1)) * _triu_sum(nom / torch.sqrt(dnom2))


def ddc_loss(hidden, out, sigma=0.15):
    n, k = out.shape
    kmat = ddc_kernel(hidden, sigma)
    l1 = _d_cs(kmat, out, k)
    l2 = 2 / (n * (n - 1)) * _triu_sum(out @ out.t())
    eye = torch.eye(k, device=out.device)
    mat = torch.exp(-sq_cdist(out, eye))
    l3 = _d_cs(kmat, mat, k)
    return l1 + l2 + l3
