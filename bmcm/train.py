import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from .data import load_dataset
from .losses import cluster_contrast, ddc_loss, feature_contrast
from .metrics import evaluate
from .model import BMCMNet
from .quality import intra_quality, mmd_tilde, partition_ami

DEFAULT_EPOCHS = {"MS-COCO": 40}


def get_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    cfg, _ = pre.parse_known_args(argv)
    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--dataset", default=None)
    p.add_argument("--data_root", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--norm", default="minmax", choices=["none", "minmax", "zscore"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fea_dim", type=int, default=256)
    p.add_argument("--tau_f", type=float, default=0.1)
    p.add_argument("--tau_c", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--eps_q", type=float, default=0.1)
    p.add_argument("--eps_f", type=float, default=0.01)
    p.add_argument("--sigma_q", type=float, default=0.15)
    p.add_argument("--sigma_ddc", type=float, default=0.15)
    p.add_argument("--ire_hidden", type=int, default=14)
    p.add_argument("--mmd_bw_scale", type=float, default=1.0)
    p.add_argument("--w_rc", type=float, default=1.0)
    p.add_argument("--w_ire", type=float, default=1.0)
    p.add_argument("--weights", default="ire", choices=["ire", "none", "uniform", "learned", "fixed"])
    p.add_argument("--fixed_weights", default="")
    p.add_argument("--levels", default="fea,clu")
    p.add_argument("--alloc", default="original", choices=["original", "uniform_mass", "permute"])
    p.add_argument("--fusion", default="quality", choices=["quality", "uniform"])
    p.add_argument("--desc_drop", default="none", choices=["none", "sep", "ent", "agree", "mmd"])
    p.add_argument("--ire_terms", default="both", choices=["both", "qal", "cal"])
    p.add_argument("--qal_reduce", default="sum", choices=["sum", "mean"])
    p.add_argument("--corrupt_view", type=int, default=-1)
    p.add_argument("--corrupt_type", default="gauss", choices=["gauss", "shuffle"])
    p.add_argument("--corrupt_level", type=float, default=0.0)
    p.add_argument("--corrupt_seed", type=int, default=12345)
    p.add_argument("--save_embedding", action="store_true")
    p.add_argument("--save_embedding_epochs", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    if cfg.config:
        with open(cfg.config) as f:
            p.set_defaults(**json.load(f))
    a = p.parse_args(argv)
    if a.dataset is None:
        p.error("--dataset or --config is required")
    if a.epochs is None:
        a.epochs = DEFAULT_EPOCHS.get(a.dataset, 200)
    if a.out is None:
        a.out = os.path.join("runs", f"{a.dataset}_s{a.seed}")
    return a


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def corrupt(views, args):
    rs = np.random.RandomState(args.corrupt_seed)
    v, lev = args.corrupt_view, args.corrupt_level
    x = views[v].copy()
    n, d = x.shape
    if args.corrupt_type == "gauss":
        x = x + lev * rs.randn(n, d).astype(np.float32)
    else:
        m = int(round(lev * n))
        idx = rs.choice(n, m, replace=False)
        x[idx] = x[rs.permutation(idx)]
    views = list(views)
    views[v] = np.ascontiguousarray(x, dtype=np.float32)
    return views


@torch.no_grad()
def save_embedding(model, xs, coef, labels, path, chunk=8192):
    model.eval()
    zf_all, cf_all = [], []
    for st in range(0, xs[0].size(0), chunk):
        zs_b, _ = model.encode([x[st:st + chunk] for x in xs])
        zf_b, cf_b, _ = model.fuse(zs_b, coef)
        zf_all.append(zf_b.cpu())
        cf_all.append(cf_b.argmax(1).cpu())
    np.savez_compressed(path, fused=torch.cat(zf_all).numpy(), pred=torch.cat(cf_all).numpy(), labels=labels)
    model.train()


@torch.no_grad()
def predict(model, xs, coef, chunk=8192):
    model.eval()
    n = xs[0].size(0)
    fused, per_view = [], [[] for _ in xs]
    for s in range(0, n, chunk):
        b = [x[s:s + chunk] for x in xs]
        zs, cs = model.encode(b)
        _, cf, _ = model.fuse(zs, coef)
        fused.append(cf.argmax(1))
        for v, c in enumerate(cs):
            per_view[v].append(c.argmax(1))
    model.train()
    return torch.cat(fused).cpu().numpy(), [torch.cat(p).cpu().numpy() for p in per_view]


def main(argv=None):
    args = get_args(argv)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    set_seed(args.seed)
    dev = torch.device(args.device)

    views, labels = load_dataset(args.dataset, args.data_root, args.norm)
    if args.corrupt_view >= 0 and args.corrupt_level > 0:
        views = corrupt(views, args)
    n_views, n = len(views), len(labels)
    k = int(labels.max()) + 1
    xs = [torch.from_numpy(v).to(dev) for v in views]
    pairs = [(u, v) for u in range(n_views) for v in range(u + 1, n_views)]
    n_pairs = len(pairs)
    levels = set(args.levels.split(","))
    use_contrast = args.weights != "none"
    use_ire = args.weights == "ire"
    fixed_w = None
    if args.weights == "fixed":
        fixed_w = torch.tensor([float(s) for s in args.fixed_weights.split(",")], device=dev)
        if fixed_w.numel() != n_pairs:
            raise ValueError(f"--fixed_weights needs {n_pairs} values")
    perm = torch.tensor([(p + 1) % n_pairs for p in range(n_pairs)], device=dev)

    model = BMCMNet([v.shape[1] for v in views], k, args.fea_dim, args.ire_hidden, n_pairs).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    n_batches = n // args.batch_size
    coef_eval = torch.full((n_views,), 1.0 / n_views, device=dev)

    log_f = open(os.path.join(args.out, "log.jsonl"), "w")
    save_epochs = {int(e) for e in args.save_embedding_epochs.split(",") if e.strip()}
    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        idx_all = torch.randperm(n, generator=gen).to(dev)
        acc_terms = {"loss": 0.0, "rc": 0.0, "ire": 0.0, "qal": 0.0, "cal": 0.0, "ddc": 0.0,
                     "fea": 0.0, "clu": 0.0}
        w_sum = torch.zeros(n_pairs, device=dev)
        weff_sum = torch.zeros(n_pairs, device=dev)
        desc_sum = torch.zeros(n_pairs, 4, device=dev)
        q_sum = torch.zeros(n_views, device=dev)
        wi_sum = torch.zeros(n_views, device=dev)
        d_sum = torch.zeros(n_views, device=dev)
        h_sum = torch.zeros(n_views, device=dev)
        coef_sum = torch.zeros(n_views, device=dev)
        for b in range(n_batches):
            idx = idx_all[b * args.batch_size:(b + 1) * args.batch_size]
            xb = [x[idx] for x in xs]
            zs, cs = model.encode(xb)

            qual = [intra_quality(z.detach(), c.detach(), args.eps_q, args.sigma_q) for z, c in zip(zs, cs)]
            w_intra = torch.stack([q_["w_intra"] for q_ in qual])
            q = torch.stack([q_["q"] for q_ in qual])
            if args.desc_drop == "sep":
                w_intra = 1.0 / (torch.stack([q_["h"] for q_ in qual]) + args.eps_q)
                q = w_intra / (1.0 + w_intra)
            elif args.desc_drop == "ent":
                w_intra = torch.stack([q_["D"] for q_ in qual]) / args.eps_q
                q = w_intra / (1.0 + w_intra)

            if args.fusion == "quality":
                coef = ((w_intra + args.eps_f) / (w_intra.sum() + n_views * args.eps_f)).detach()
            else:
                coef = torch.full((n_views,), 1.0 / n_views, device=dev)
            _, cf, hf = model.fuse(zs, coef)

            loss_ire = zs[0].new_tensor(0.0)
            loss_qal = loss_cal = zs[0].new_tensor(0.0)
            w = None
            if use_contrast:
                with torch.no_grad():
                    a_uv = torch.stack([partition_ami(cs[u].detach(), cs[v].detach()) for u, v in pairs])
                    m_uv = torch.stack([mmd_tilde(zs[u].detach(), zs[v].detach(), args.mmd_bw_scale)
                                        for u, v in pairs])
                    alpha = args.alpha
                    if args.desc_drop == "agree":
                        a_uv, alpha = torch.zeros_like(a_uv), 0.0
                    elif args.desc_drop == "mmd":
                        m_uv, alpha = torch.zeros_like(m_uv), 1.0
                    qu = q[[u for u, _ in pairs]]
                    qv = q[[v for _, v in pairs]]
                    desc = torch.stack([(qu + qv) / 2, (qu - qv).abs(), a_uv, m_uv], 1)
                    q_ref = torch.minimum(qu, qv)
                    r_ref = alpha * a_uv + (1 - alpha) * (1 - m_uv)
                desc_sum += desc
                if use_ire:
                    w = model.ire(desc)
                    loss_qal = torch.relu(w - q_ref).sum()
                    if args.qal_reduce == "mean":
                        loss_qal = loss_qal / n_pairs
                    loss_cal = ((w - r_ref) ** 2).mean()
                    if args.ire_terms == "both":
                        loss_ire = loss_qal + loss_cal
                    elif args.ire_terms == "qal":
                        loss_ire = loss_qal
                    else:
                        loss_ire = loss_cal
                    w_rc = w.detach()
                    if args.alloc == "uniform_mass":
                        w_rc = w_rc.sum() / n_pairs * torch.ones_like(w_rc)
                    elif args.alloc == "permute":
                        w_rc = w_rc[perm]
                elif args.weights == "uniform":
                    w_rc = torch.ones(n_pairs, device=dev)
                elif args.weights == "learned":
                    w_rc = torch.sigmoid(model.pair_logits)
                else:
                    w_rc = fixed_w
                w_sum += (w if w is not None else w_rc).detach()
                weff_sum += w_rc.detach()

                loss_rc = zs[0].new_tensor(0.0)
                for p, (u, v) in enumerate(pairs):
                    term = 0.0
                    if "fea" in levels:
                        lf = feature_contrast(zs[u], zs[v], cf, args.tau_f)
                        term = term + lf
                        acc_terms["fea"] += lf.item() / n_pairs
                    if "clu" in levels:
                        lc = cluster_contrast(cs[u], cs[v], args.tau_c)
                        term = term + lc
                        acc_terms["clu"] += lc.item() / n_pairs
                    loss_rc = loss_rc + w_rc[p] * term
            else:
                loss_rc = zs[0].new_tensor(0.0)

            loss_ddc = ddc_loss(hf, cf, args.sigma_ddc)

            if args.w_ire != 1.0:
                loss_ire = args.w_ire * loss_ire
            if args.w_rc != 1.0:
                loss_rc = args.w_rc * loss_rc
            loss = loss_ire + loss_rc + loss_ddc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            acc_terms["loss"] += loss.item()
            acc_terms["rc"] += loss_rc.item()
            acc_terms["ire"] += loss_ire.item()
            acc_terms["qal"] += loss_qal.item()
            acc_terms["cal"] += loss_cal.item()
            acc_terms["ddc"] += loss_ddc.item()
            q_sum += q
            wi_sum += w_intra
            d_sum += torch.stack([q_["D"] for q_ in qual])
            h_sum += torch.stack([q_["h"] for q_ in qual])
            coef_sum += coef.detach()

        nb = max(n_batches, 1)
        coef_eval = coef_sum / nb
        rec = {"epoch": epoch, **{k_: v / nb for k_, v in acc_terms.items()},
               "w": (w_sum / nb).tolist(), "w_eff": (weff_sum / nb).tolist(),
               "desc": (desc_sum / nb).tolist(), "q": (q_sum / nb).tolist(),
               "w_intra": (wi_sum / nb).tolist(), "D": (d_sum / nb).tolist(),
               "h": (h_sum / nb).tolist(), "fusion_coef": coef_eval.tolist(),
               "epoch_time": time.time() - t0}
        pred, per_view = predict(model, xs, coef_eval)
        rec.update(evaluate(labels, pred))
        rec["view_metrics"] = [evaluate(labels, pv) for pv in per_view]
        rec["n_pred_clusters"] = int(len(np.unique(pred)))
        if epoch in save_epochs:
            save_embedding(model, xs, coef_eval, labels, os.path.join(args.out, f"embedding_ep{epoch:03d}.npz"))
        history.append(rec)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(f"ep {epoch:3d} loss {rec['loss']:.4f} acc {rec['acc']:.4f} nmi {rec['nmi']:.4f} "
              f"ari {rec['ari']:.4f} w {np.round(rec['w'], 3).tolist()} {rec['epoch_time']:.2f}s",
              flush=True)
        if not math.isfinite(rec["loss"]):
            print("non-finite loss, stopping", flush=True)
            break
    log_f.close()

    if args.save_embedding:
        save_embedding(model, xs, coef_eval, labels, os.path.join(args.out, "embedding.npz"))

    final = history[-1]
    last20 = history[-20:]
    summary = {
        "dataset": args.dataset, "seed": args.seed, "epochs_run": len(history),
        "final": {m: final[m] for m in ("acc", "nmi", "ari")},
        "last20_mean": {m: float(np.mean([h[m] for h in last20])) for m in ("acc", "nmi", "ari")},
        "final_w": final["w"], "final_q": final["q"], "final_fusion_coef": final["fusion_coef"],
        "wall_time_s": time.time() - t_start,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("FINAL", json.dumps(summary["final"]), flush=True)


if __name__ == "__main__":
    main()
