"""S2 semantic latent locator training + evaluation (PLAN §3.1 / §7 / §8).

Task: given native H3 latent z [1,24,32,48,84] and a class query q in
{0..31} ∪ {NULL}, predict a soft support volume M(t,y,x) over the 32x48x84
latent grid. NULL queries are supervised with empty support.

Supervision: the 4 keyframe instance masks (RGB 768x1344) are rasterized to
48x84 latent cells (16x16 mean coverage) and linearly interpolated in latent
time to 32 positions. Training never sees masks at deployment; they are the
paired supervision from the reviewed S0 labels.

Ablations / baselines (gate §8):
  --ablation query   : full model (class-conditioned)                 [main]
  --ablation none    : class-agnostic saliency head (query ignored)   [ablation]
Evaluation additionally reports a fixed center prior and random blobs,
plus empty-query AUC from the NULL token.

Usage:
  python s2_train_locator.py --ablation query --epochs 120
  python s2_train_locator.py --ablation none  --epochs 120
  python s2_train_locator.py --eval_only --ckpt .../best_query.pt
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from project_paths import DATA_ROOT

S32 = DATA_ROOT
LAB = S32 / "labels"
DATA = S32 / "data"
OUT = S32 / "s2_locator"
T, H, W = 32, 48, 84
KEY_FRAMES = [0, 35, 71, 106]      # RGB keyframes carrying masks
N_FRAMES = 107
NULL = 32                           # null query index


# ---------------------------------------------------------------- data ----
def rasterize(mask, gh=H, gw=W, cell=16):
    m = mask.astype(np.float32)
    m = m[: gh * cell, : gw * cell]
    return m.reshape(gh, cell, gw, cell).mean(axis=(1, 3))


def target_volume(sid):
    d = np.load(LAB / f"{sid}_masks.npz")
    km = {int(k[1:]): rasterize(d[k]) for k in d.files}
    # keyframe RGB frame -> continuous latent position, spanning [0, T-1]
    pos = [f * (T - 1) / (N_FRAMES - 1) for f in KEY_FRAMES]
    vol = np.zeros((T, H, W), np.float32)
    ts = np.arange(T, dtype=np.float32)
    for i in range(len(KEY_FRAMES) - 1):
        p0, p1 = pos[i], pos[i + 1]
        m0, m1 = km[KEY_FRAMES[i]], km[KEY_FRAMES[i + 1]]
        sel = (ts >= p0) & (ts <= p1) if i == len(KEY_FRAMES) - 2 else (ts >= p0) & (ts < p1)
        a = ((ts[sel] - p0) / max(p1 - p0, 1e-6))[:, None, None]
        vol[sel] = (1 - a) * m0 + a * m1
    vol[ts < pos[0]] = km[KEY_FRAMES[0]]
    vol[ts > pos[-1]] = km[KEY_FRAMES[-1]]
    return vol


def load_split(gold, split, cats):
    ids = gold["semantic_approved_by_split"][split]
    recs = json.loads((S32 / "splits.json").read_text())["rows"]
    split_of = {r["id"]: r["split"] for r in recs}
    zs, ys, cls = [], [], []
    for sid in ids:
        assert split_of.get(sid, split) == split, (sid, split_of.get(sid))
        z = torch.load(DATA / sid / "native_latent.pt", map_location="cpu")
        z = (z["video"] if isinstance(z, dict) else z)[0].float()  # [24,T,H,W]
        z = z / z.std().clamp_min(1e-6)
        vol = target_volume(sid)
        cat = json.loads((LAB / f"{sid}.json").read_text())["category"]
        zs.append(z.half())
        ys.append(torch.from_numpy(vol))
        cls.append(cats.index(cat))
    return torch.stack(zs), torch.stack(ys), torch.tensor(cls)


# --------------------------------------------------------------- model ----
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, tempo=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride=(tempo, stride, stride), padding=1)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)

    def forward(self, x):
        x = F.gelu(self.n1(self.c1(x)))
        return F.gelu(self.n2(self.c2(x)))


class Locator(nn.Module):
    """Small 3D U-Net conditioned on a class embedding via FiLM at bottleneck."""

    def __init__(self, n_cls=33, emb=96):
        super().__init__()
        e = emb
        self.emb = nn.Embedding(n_cls, e)
        self.inp = ConvBlock(24 + e, 64)                 # 32x48x84
        self.d1 = ConvBlock(64, 128, 2, 2)               # 16x24x42
        self.d2 = ConvBlock(128, 256, 2, 2)              # 8x12x21
        self.d3 = ConvBlock(256, 256, 2, 1)              # 8x6x10
        self.film = nn.Linear(e, 512)
        self.u2 = ConvBlock(256 + 256, 128, tempo=1)     # upsample manually
        self.u1 = ConvBlock(128 + 128, 64, tempo=1)
        self.u0 = ConvBlock(64 + 64, 64, tempo=1)
        self.head = nn.Conv3d(64, 1, 1)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, z, q):
        B = z.shape[0]
        e = self.emb(q)                                   # [B,e]
        z = torch.cat([z, e[:, :, None, None, None].expand(-1, -1, T, H, W)], 1)
        x0 = self.inp(z)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        g = self.pool(x3).flatten(1)
        s, t = self.film(e).chunk(2, dim=1)
        x3 = x3 * (1 + s[:, :, None, None, None]) + t[:, :, None, None, None]
        u = F.interpolate(x3, size=x2.shape[2:], mode="trilinear", align_corners=False)
        u = self.u2(torch.cat([u, x2], 1))
        u = F.interpolate(u, size=x1.shape[2:], mode="trilinear", align_corners=False)
        u = self.u1(torch.cat([u, x1], 1))
        u = F.interpolate(u, size=x0.shape[2:], mode="trilinear", align_corners=False)
        u = self.u0(torch.cat([u, x0], 1))
        return self.head(u)[:, 0]                          # [B,T,H,W] logits


# ---------------------------------------------------------------- loss ----
def loss_fn(logits, target, pos_weight=2.0):
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device))
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (denom + 1)
    return bce + dice.mean()


# ----------------------------------------------------------- evaluation ----
def iou_at(pred, target, thr=0.5):
    p = pred > thr
    t = target > 0.5
    inter = (p & t).sum(dim=(1, 2, 3)).float()
    union = (p | t).sum(dim=(1, 2, 3)).float()
    return (inter / union.clamp_min(1)).cpu()


def center_prior():
    yy = torch.arange(H)[:, None] - H / 2
    xx = torch.arange(W)[None, :] - W / 2
    g = torch.exp(-(yy ** 2 + xx ** 2) / (2 * 6.0 ** 2))
    return g[None].expand(T, -1, -1)


def random_prior(n=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        cy, cx = torch.randint(6, H - 6, (2,), generator=g).tolist()
        yy = torch.arange(H)[:, None] - cy
        xx = torch.arange(W)[None, :] - cx
        out.append(torch.exp(-(yy ** 2 + xx ** 2) / (2 * 6.0 ** 2)))
    return torch.stack(out)


def auc(scores_pos, scores_neg):
    s = np.concatenate([scores_pos, scores_neg])
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    order = np.argsort(s)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    from collections import defaultdict
    groups = defaultdict(list)
    for i, v in enumerate(s):
        groups[v].append(i)
    for v, idx in groups.items():
        if len(idx) > 1:
            m = np.mean([ranks[i] for i in idx])
            for i in idx:
                ranks[i] = m
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.inference_mode()
def evaluate(model, Z, Y, C, ablation, bs=8, want_auc=True):
    model.eval()
    ious, dices, s_pos, s_neg, wrong_max = [], [], [], [], []
    Zg, Yg = Z.float(), Y.float()
    ctr = center_prior()
    rnd = random_prior()
    base_ct, base_rnd = [], []
    for i in range(0, len(Zg), bs):
        zb = Zg[i:i + bs].cuda()
        q = C[i:i + bs].cuda() if ablation == "query" else torch.full(
            (min(bs, len(Zg) - i),), NULL, device="cuda", dtype=torch.long)
        p = torch.sigmoid(model(zb, q)).cpu()
        tb = Yg[i:i + bs]
        ious.append(iou_at(p, tb))
        inter = ((p > .5) & (tb > .5)).sum(dim=(1, 2, 3)).float()
        d = (2 * inter + 1) / ((p > .5).sum(dim=(1, 2, 3)) + (tb > .5).sum(dim=(1, 2, 3)) + 1)
        dices.append(d)
        base_ct.append(iou_at(ctr[None].expand_as(tb), tb))
        base_rnd.append(torch.stack([iou_at(r[None].expand_as(tb), tb) for r in rnd]).mean(0))
        if want_auc and ablation == "query":
            s_pos.append(p.flatten(1).max(1).values)
            pn = torch.sigmoid(model(zb, torch.full((zb.shape[0],), NULL,
                                                    device="cuda", dtype=torch.long))).cpu()
            s_neg.append(pn.flatten(1).max(1).values)
            # wrong-class queries: C6 rejection
            wq = (C[i:i + bs].cuda() + 1) % 32
            pw = torch.sigmoid(model(zb, wq)).cpu()
            wrong_max.append(pw.flatten(1).max(1).values)
    out = {"mIoU": float(torch.cat(ious).mean()),
           "mDice": float(torch.cat(dices).mean()),
           "iou_median": float(torch.cat(ious).median()),
           "center_prior_iou": float(torch.cat(base_ct).mean()),
           "random_prior_iou": float(torch.cat(base_rnd).mean())}
    if s_pos:
        out["empty_query_auc"] = auc(torch.cat(s_pos).numpy(), torch.cat(s_neg).numpy())
        if wrong_max:
            wm = torch.cat(wrong_max)
            # wrong-query suppression: fraction of items whose wrong-query peak
            # stays under the same item's correct-query peak times 0.5
            out["wrong_query_suppress"] = float(
                (wm < 0.5 * torch.cat(s_pos)).float().mean())
    return out


def per_bucket(Z, Y, C, model, ablation):
    """IoU by target size bucket (small/medium/large by GT cell count)."""
    with torch.inference_mode():
        p = torch.sigmoid(model(Z.float().cuda(),
                                C.cuda() if ablation == "query" else
                                torch.full((len(C),), NULL, device="cuda", dtype=torch.long))).cpu()
    iou = iou_at(p, Y.float())
    cells = Y.float().sum(dim=(1, 2, 3))
    qs = torch.quantile(cells, torch.tensor([1 / 3, 2 / 3]))
    out = {}
    for name, m in (("small", cells <= qs[0]), ("medium", (cells > qs[0]) & (cells <= qs[1])),
                    ("large", cells > qs[1])):
        out[name] = {"n": int(m.sum()), "mIoU": float(iou[m].mean()) if m.any() else None}
    return out


# ---------------------------------------------------------------- train ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", choices=["query", "none"], default="query")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--null_frac", type=float, default=0.25)
    ap.add_argument("--wrong_frac", type=float, default=0.0,
                    help="fraction of samples trained with a WRONG class query "
                         "whose target is empty (query-specificity negatives, C6)")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--split", choices=["validation", "test"], default="validation",
                    help="split used for evaluation (eval_only) and checkpoint selection")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    gold = json.loads((S32 / "gold.json").read_text())
    cats = sorted({json.loads((LAB / f"{s}.json").read_text())["category"]
                   for sp in ("train", "validation", "test")
                   for s in gold["semantic_approved_by_split"][sp]})
    assert len(cats) == 32, len(cats)

    print("loading train split ...", flush=True)
    Ztr, Ytr, Ctr = load_split(gold, "train", cats)
    print("loading validation split ...", flush=True)
    Zva, Yva, Cva = load_split(gold, args.split, cats)
    print(f"train {tuple(Ztr.shape)} {args.split} {tuple(Zva.shape)}", flush=True)

    model = Locator(n_cls=33).cuda()
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params {nparam:.2f}M", flush=True)

    if args.eval_only:
        assert args.ckpt
        model.load_state_dict(torch.load(args.ckpt))
        print(args.split.upper(),
              json.dumps(evaluate(model, Zva, Yva, Cva, args.ablation)), flush=True)
        print("BUCKETS", json.dumps(per_bucket(Zva, Yva, Cva, model, args.ablation)), flush=True)
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * math.ceil(len(Ztr) / args.bs),
        pct_start=0.1)
    best = -1
    logp = OUT / f"train_{args.ablation}.jsonl"
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Ztr))
        run, nb = 0.0, 0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            z = Ztr[idx].float().cuda()
            y = Ytr[idx].float().cuda()
            q = Ctr[idx].cuda()
            if args.ablation == "none":
                q[:] = NULL
            elif args.null_frac > 0:
                null = torch.rand(len(idx)) < args.null_frac
                q = q.clone()
                q[null] = NULL
                y = y.clone()
                y[null] = 0.0
            if args.ablation == "query" and args.wrong_frac > 0:
                wrong = torch.rand(len(idx)) < args.wrong_frac
                if wrong.any():
                    q = q.clone()
                    y = y.clone()
                    wr = torch.randint(0, 32, (int(wrong.sum()),), device=q.device)
                    # ensure the wrong class differs from the true one
                    wr = torch.where(wr == q[wrong], (wr + 1) % 32, wr)
                    q[wrong] = wr
                    y[wrong] = 0.0
            # horizontal flip consistency args
            flip = torch.rand(1).item() < 0.5
            if flip:
                z = torch.flip(z, dims=[-1])
                y = torch.flip(y, dims=[-1])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(z, q)
                loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            nb += 1
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            m = evaluate(model, Zva, Yva, Cva, args.ablation)
            rec = {"epoch": ep + 1, "loss": run / nb, "lr": sched.get_last_lr()[0],
                   "sec": round(time.time() - t0, 1), **m}
            with open(logp, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            # joint selection: localization accuracy + query specificity.
            # mIoU-only selection picks early epochs before class gating is learned
            # (v3: best-mIoU ep60 had suppress 0.14 while ep100 had 0.84 at mIoU 0.70).
            supp = 0.0 if args.ablation == "none" else m.get("wrong_query_suppress", 0.0)
            score = m["mIoU"] + 0.5 * supp
            if m["mIoU"] < 0.60:
                score = -1.0  # guard against degenerate low-IoU high-suppress models
            if score > best:
                best = score
                torch.save(model.state_dict(), OUT / f"best_{args.ablation}.pt")
            torch.save(model.state_dict(), OUT / f"last_{args.ablation}.pt")
    # final detailed report
    model.load_state_dict(torch.load(OUT / f"best_{args.ablation}.pt"))
    fin = {"best_val": evaluate(model, Zva, Yva, Cva, args.ablation),
           "buckets": per_bucket(Zva, Yva, Cva, model, args.ablation)}
    print("FINAL", json.dumps(fin, indent=1), flush=True)
    (OUT / f"final_{args.ablation}.json").write_text(json.dumps(fin, indent=1))


if __name__ == "__main__":
    main()
