import math

import torch
import torch.nn.functional as F

from .losses import ddc_kernel, sq_cdist


@torch.no_grad()
def separability(z, c, rel_sigma=0.15, mass_eps=1e-8):
    k = c.size(1)
    mass = c.sum(0)
    if bool((mass < mass_eps).any()):
        return z.new_tensor(0.0)
    a = c / mass
    kmat = ddc_kernel(z, rel_sigma)
    g = a.t() @ kmat @ a
    diag = torch.diagonal(g).clamp_min(1e-12)
    ratio = g / torch.sqrt(diag.unsqueeze(1) * diag.unsqueeze(0))
    overlap = 2.0 / (k * (k - 1)) * torch.triu(ratio, diagonal=1).sum()
    return (1.0 - overlap).clamp(0.0, 1.0)


@torch.no_grad()
def assignment_entropy(c):
    k = c.size(1)
    return -(c * torch.log(c.clamp_min(1e-12))).sum(1).mean() / math.log(k)


@torch.no_grad()
def intra_quality(z, c, eps_q=0.1, rel_sigma=0.15):
    d = separability(z, c, rel_sigma)
    h = assignment_entropy(c)
    w_intra = d / (h + eps_q)
    q = w_intra / (1.0 + w_intra)
    return {"D": d, "h": h, "w_intra": w_intra, "q": q}


def _expected_mi(a, b, n):
    nmax = int(torch.minimum(a.max(), b.max()).item())
    nij = torch.arange(1, nmax + 1, dtype=torch.float64, device=a.device)[None, None, :]
    A, Bc = a[:, None, None], b[None, :, None]
    valid = (nij >= torch.clamp(A + Bc - n, min=1)) & (nij <= torch.minimum(A, Bc))
    lg = torch.lgamma
    logp = (lg(A + 1) + lg(Bc + 1) + lg(n - A + 1) + lg(n - Bc + 1) - lg(torch.tensor(n + 1.0, dtype=torch.float64))
            - lg(nij + 1) - lg(torch.clamp(A - nij, min=0) + 1) - lg(torch.clamp(Bc - nij, min=0) + 1)
            - lg(torch.clamp(n - A - Bc + nij, min=0) + 1))
    term = nij / n * torch.log(n * nij / (A * Bc)) * torch.exp(logp)
    return torch.where(valid, term, torch.zeros_like(term)).sum()


@torch.no_grad()
def partition_ami(cu, cv):
    k = cu.size(1)
    yu = cu.argmax(1)
    yv = cv.argmax(1)
    n = yu.numel()
    cont = torch.bincount(yu * k + yv, minlength=k * k).reshape(k, k).double()
    a, b = cont.sum(1), cont.sum(0)
    pu, pv = a / n, b / n
    hu = -(pu[pu > 0] * torch.log(pu[pu > 0])).sum()
    hv = -(pv[pv > 0] * torch.log(pv[pv > 0])).sum()
    joint = cont / n
    nz = joint > 0
    mi = (joint[nz] * torch.log(joint[nz] / (pu.unsqueeze(1) * pv.unsqueeze(0))[nz])).sum()
    emi = _expected_mi(a[a > 0], b[b > 0], n)
    hbar = (hu + hv) / 2
    if float(hbar) <= float(emi):
        return cu.new_tensor(0.0)
    return ((mi - emi) / (hbar - emi)).clamp(0.0, 1.0).float()


@torch.no_grad()
def mmd_tilde(zu, zv, bw_scale=1.0):
    b = zu.size(0)
    zu, zv = F.normalize(zu, dim=1), F.normalize(zv, dim=1)
    zz = torch.cat([zu, zv], 0)
    d = sq_cdist(zz, zz)
    bw = torch.median(d[~torch.eye(2 * b, dtype=torch.bool, device=d.device)]).clamp_min(1e-12)
    if bw_scale != 1.0:
        bw = bw * bw_scale
    kmat = torch.exp(-d / bw)
    kuu = kmat[:b, :b].mean()
    kvv = kmat[b:, b:].mean()
    kuv = kmat[:b, b:].mean()
    mmd2 = (kuu + kvv - 2 * kuv).clamp_min(0.0)
    return (mmd2 / 2).clamp(0.0, 1.0)
