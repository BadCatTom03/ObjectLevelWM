"""S3.1: self-trained blind RGB reader (no H3 VAE weights).

Input   isolated local RGB crop [F=107, 16*bh, 16*bw, 3]   (C1; no mask, no
        coordinates, no clean reference)
Output  two dense maps on the latent grid [T=32, bh, bw]:
          res  signed estimate of the written latent pattern (host-cancelled)
          evi  watermark-evidence logit (keyless "is this cell marked")
Message recovery = keyed correlator on `res` (same LocalPRC carrier / PRC code
as T0 and s3_build_data.py), so the network itself is key-agnostic: every
training draw uses a fresh key and a fresh random 32-bit message.

Training pairs: the SAME spatial window from the watermarked crop and from the
clean crop of the same decode batch (target 0 for clean), so the only
difference the network can exploit is the watermark itself. Windows smaller
than the canvas are placed on a gray canvas at a random offset.

Validation (full crops, held-out sources): chip/bit accuracy, exact 32-bit
rate, score separation vs paired clean crops + extra null keys, TPR at the
1% FPR threshold frozen on validation nulls, keyless evidence AUC.

Usage:
  python s3_train_reader.py --steps 6000
  python s3_train_reader.py --eval_only --ckpt .../best.pt --split validation --crf 23
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t0_g1g2_probe import N_CHIPS, LocalPRC, h264_roundtrip, prc_encode, prc_soft_decode  # noqa: E402

from project_paths import DATA_ROOT

S32 = DATA_ROOT
ROOT = S32 / "s3_reader"
T_LAT, MSG_BITS = 32, 32


# ---------------------------------------------------------------- data ----
class Windows(torch.utils.data.Dataset):
    def __init__(self, split_dir, win, length, crf_prob=0.0, crf_range=(20, 26), seed=0):
        self.items = []
        for sd in sorted(Path(split_dir).iterdir()):
            if (sd / "meta.json").exists():
                meta = json.loads((sd / "meta.json").read_text())
                self.items += [(sd, d["k"]) for d in meta["draws"]]
        assert self.items, split_dir
        self.win, self.length = win, length
        self.crf_prob, self.crf_range, self.seed = crf_prob, crf_range, seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        sd, k = self.items[int(rng.integers(len(self.items)))]
        w = np.load(sd / f"w{k}.npy", mmap_mode="r")
        c = np.load(sd / "clean.npy", mmap_mode="r")
        tgt = np.load(sd / f"target{k}.npy").astype(np.float32)
        msk = np.load(sd / "mask.npy").astype(np.float32)
        _, bh, bw = tgt.shape
        win = self.win
        hh, ww = min(win, bh), min(win, bw)
        # 80%: centre the window on a marked cell, else uniform over the box
        if rng.random() < 0.8:
            ys, xs = np.where(msk.max(0) >= 0.5)
            j = int(rng.integers(len(ys))) if len(ys) else 0
            cy, cx = (int(ys[j]), int(xs[j])) if len(ys) else (bh // 2, bw // 2)
            y0 = int(np.clip(cy - hh // 2 + rng.integers(-3, 4), 0, bh - hh))
            x0 = int(np.clip(cx - ww // 2 + rng.integers(-3, 4), 0, bw - ww))
        else:
            y0, x0 = int(rng.integers(0, bh - hh + 1)), int(rng.integers(0, bw - ww + 1))
        oy, ox = int(rng.integers(0, win - hh + 1)), int(rng.integers(0, win - ww + 1))
        F_ = w.shape[0]
        x = np.full((2, F_, win * 16, win * 16, 3), 127, np.uint8)
        sl = (slice(None), slice(y0 * 16, (y0 + hh) * 16), slice(x0 * 16, (x0 + ww) * 16))
        pw, pc = np.asarray(w[sl]), np.asarray(c[sl])
        if self.crf_prob > 0 and rng.random() < self.crf_prob:
            crf = int(rng.integers(self.crf_range[0], self.crf_range[1] + 1))
            pw = (h264_roundtrip(torch.from_numpy(pw).float() / 255, crf) * 255).round().clamp(0, 255).numpy().astype(np.uint8)
            pc = (h264_roundtrip(torch.from_numpy(pc).float() / 255, crf) * 255).round().clamp(0, 255).numpy().astype(np.uint8)
        x[0, :, oy * 16:(oy + hh) * 16, ox * 16:(ox + ww) * 16] = pw
        x[1, :, oy * 16:(oy + hh) * 16, ox * 16:(ox + ww) * 16] = pc
        t = np.zeros((2, T_LAT, win, win), np.float32)
        m = np.zeros((2, T_LAT, win, win), np.float32)
        t[0, :, oy:oy + hh, ox:ox + ww] = tgt[:, y0:y0 + hh, x0:x0 + ww]
        m[0, :, oy:oy + hh, ox:ox + ww] = msk[:, y0:y0 + hh, x0:x0 + ww]
        return torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(m)


def to_input(x_u8):
    """[B,F,H,W,3] uint8 -> [B,3,F,H,W] float in [-1,1]."""
    return x_u8.permute(0, 4, 1, 2, 3).float().div(127.5).sub(1.0)


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


class Reader(nn.Module):
    """RGB [B,3,F,16h,16w] -> res, evi  [B,T_LAT,h,w]."""

    def __init__(self, base=32):
        super().__init__()
        b = base
        # lossless 4x4 pixel-unshuffle (3 -> 48 ch) keeps fine texture while the
        # first 3D conv already runs at /4 resolution (full-res GroupNorm OOMs).
        self.stem = nn.Sequential(
            nn.Conv3d(48, 2 * b, 3, padding=1),
            nn.GroupNorm(8, 2 * b), nn.GELU())              # F,   /4
        self.s1 = ConvBlock(2 * b, 2 * b, 1, 2)             # F/2, /4
        self.s2 = ConvBlock(2 * b, 4 * b, 2, 1)             # F/2, /8
        self.s3 = ConvBlock(4 * b, 8 * b, 2, 1)             # F/2, /16  (latent grid)
        self.s4 = ConvBlock(8 * b, 8 * b, 2, 2)             # F/4, /32  (context)
        self.mid = ConvBlock(8 * b, 8 * b)
        self.up = ConvBlock(16 * b, 8 * b)
        self.post = nn.Sequential(ConvBlock(8 * b, 6 * b), ConvBlock(6 * b, 4 * b))
        self.res = nn.Conv3d(4 * b, 1, 1)
        self.evi = nn.Conv3d(4 * b, 1, 1)

    def forward(self, x):
        B, C, Fr, Hh, Ww = x.shape
        x = F.pixel_unshuffle(x.permute(0, 2, 1, 3, 4).reshape(B * Fr, C, Hh, Ww), 4)
        x = x.reshape(B, Fr, 48, Hh // 4, Ww // 4).permute(0, 2, 1, 3, 4)
        x = self.stem(x)
        x = self.s2(self.s1(x))
        x3 = self.s3(x)
        x4 = self.mid(self.s4(x3))
        x4 = F.interpolate(x4, size=x3.shape[2:], mode="trilinear", align_corners=False)
        y = self.up(torch.cat([x3, x4], 1))
        y = F.interpolate(y, size=(T_LAT, y.shape[3], y.shape[4]), mode="trilinear", align_corners=False)
        y = self.post(y)
        return self.res(y)[:, 0], self.evi(y)[:, 0]


# ---------------------------------------------------------------- eval ----
def bits_of(msg):
    return np.array([(msg >> (MSG_BITS - 1 - i)) & 1 for i in range(MSG_BITS)], np.uint8)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum())
                 / (len(pos) * len(neg)))


ORACLE_GAIN = 8.0


@torch.no_grad()
def infer(model, crop_u8, ref_u8=None):
    """ref_u8 given => oracle-diff diagnostic input (w - clean) * gain; else blind."""
    x = to_input(torch.from_numpy(np.ascontiguousarray(crop_u8))[None].cuda())
    if ref_u8 is not None:
        x = (x - to_input(torch.from_numpy(np.ascontiguousarray(ref_u8))[None].cuda())) * ORACLE_GAIN
    with torch.autocast("cuda", dtype=torch.bfloat16):
        r, e = model(x)
    return r[0].float().cpu().numpy(), torch.sigmoid(e[0].float()).cpu().numpy()


def codec(crop_u8, crf):
    y = h264_roundtrip(torch.from_numpy(np.ascontiguousarray(crop_u8)).float() / 255, crf)
    return (y * 255).round().clamp(0, 255).numpy().astype(np.uint8)


def evaluate(model, split_dir, crf=None, n_null=16, limit=0, dump=None, oracle=False):
    model.eval()
    dirs = [sd for sd in sorted(Path(split_dir).iterdir()) if (sd / "meta.json").exists()]
    if limit:
        dirs = dirs[:limit]
    rows, nulls, pos_scores, evi_pos, evi_neg, wrong = [], [], [], [], [], []
    for sd in dirs:
        meta = json.loads((sd / "meta.json").read_text())
        clean = np.load(sd / "clean.npy")
        if crf is not None:
            clean = codec(clean, crf)
        rc, ec = infer(model, clean, clean if oracle else None)
        Tn, bh, bw = rc.shape
        evi_neg.append(float(ec.mean()))
        for j in range(n_null):
            nk = f"s3null|{meta['sid']}|{j}".encode()
            nulls.append(prc_soft_decode(LocalPRC(nk, Tn, bh, bw).demodulate(rc), nk, N_CHIPS, MSG_BITS)[1])
        for d in meta["draws"]:
            k, key, msg = d["k"], d["key"].encode(), d["msg"]
            w = np.load(sd / f"w{k}.npy")
            if crf is not None:
                w = codec(w, crf)
            rw, ew = infer(model, w, clean if oracle else None)
            car = LocalPRC(key, Tn, bh, bw)
            cw = prc_encode(msg, key, N_CHIPS, MSG_BITS)
            soft = car.demodulate(rw)
            bits, score = prc_soft_decode(soft, key, N_CHIPS, MSG_BITS)
            tb = bits_of(msg)
            tgt = np.load(sd / f"target{k}.npy").astype(np.float32)
            msk = np.load(sd / "mask.npy").astype(np.float32) >= 0.5
            corr = float(np.corrcoef(rw[msk], tgt[msk])[0, 1]) if msk.sum() > 8 else float("nan")
            clean_score = prc_soft_decode(car.demodulate(rc), key, N_CHIPS, MSG_BITS)[1]
            nulls.append(clean_score)
            for j in range(4):
                wk = f"s3wrong|{meta['sid']}|{j}".encode()
                wrong.append(prc_soft_decode(LocalPRC(wk, Tn, bh, bw).demodulate(rw), wk, N_CHIPS, MSG_BITS)[1])
            pos_scores.append(score)
            evi_pos.append(float(ew.mean()))
            rows.append({"sid": meta["sid"], "k": k, "bh": bh, "bw": bw,
                         "mask_cells": float(msk.sum() / Tn),
                         "chip_acc": float(((soft > 0).astype(np.uint8) == cw).mean()),
                         "bit_acc": float((bits == tb).mean()), "exact": bool((bits == tb).all()),
                         "score": score, "clean_score": clean_score, "corr": corr,
                         "evi_w": float(ew.mean()), "evi_c": float(ec.mean()),
                         "evi_in": float(ew[msk].mean()) if msk.any() else float("nan")})
    thr = float(np.quantile(nulls, 0.99))
    chip = np.array([r["chip_acc"] for r in rows])
    out = {"n": len(rows), "crf": crf,
           "chip_acc_mean": float(chip.mean()), "chip_acc_median": float(np.median(chip)),
           "bit_acc_mean": float(np.mean([r["bit_acc"] for r in rows])),
           "exact_rate": float(np.mean([r["exact"] for r in rows])),
           "corr_median": float(np.nanmedian([r["corr"] for r in rows])),
           "score_pos_median": float(np.median(pos_scores)),
           "score_null_median": float(np.median(nulls)), "n_null": len(nulls),
           "thr_fpr1": thr, "tpr_at_fpr1": float((np.array(pos_scores) > thr).mean()),
           "exact_and_accept": float(np.mean([r["exact"] and r["score"] > thr for r in rows])),
           "score_auc": auc(pos_scores, nulls),
           "wrongkey_accept": float((np.array(wrong) > thr).mean()),
           "evi_auc": auc(evi_pos, evi_neg)}
    if dump:
        Path(dump).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return out


# ---------------------------------------------------------------- train ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--pairs", type=int, default=2, help="(watermarked, clean) window pairs per step")
    ap.add_argument("--win", type=int, default=24, help="training window in latent cells")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--crf_prob", type=float, default=0.0, help="H.264 augmentation probability")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--oracle_diff", action="store_true",
                    help="DIAGNOSTIC ONLY: feed (w - clean) so the host is removed; "
                         "checks target alignment / learnability, not a blind reader")
    ap.add_argument("--gamma0", type=float, default=1.0,
                    help="RGB residual amplification curriculum: watermarked input is "
                         "clean + g*(w-clean), g log-uniform in [1, gmax(step)], gmax "
                         "decays gamma0 -> 1 over the first --gamma_frac of training")
    ap.add_argument("--gamma_frac", type=float, default=0.6)
    ap.add_argument("--fade", action="store_true",
                    help="host fade-in curriculum: input = beta*clean + g*(w-clean); "
                         "beta 0->1 and g ORACLE_GAIN->1 between --fade_a and --fade_b "
                         "(fractions of training). Starts as the solvable pure-residual "
                         "problem and ends at the true blind condition (beta=1, g=1).")
    ap.add_argument("--fade_a", type=float, default=0.15)
    ap.add_argument("--fade_b", type=float, default=0.70)
    ap.add_argument("--init", default=None, help="warm-start weights")
    ap.add_argument("--data", default="data")
    ap.add_argument("--run", default="r1")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--crf", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    out = ROOT / f"run_{args.run}"
    out.mkdir(parents=True, exist_ok=True)
    droot = ROOT / args.data

    model = Reader(args.base).cuda()
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)

    if args.eval_only:
        model.load_state_dict(torch.load(args.ckpt))
        tag = f"{args.split}_crf{args.crf}" if args.crf is not None else args.split
        m = evaluate(model, droot / args.split, crf=args.crf, limit=args.val_limit,
                     dump=out / f"eval_{tag}_rows.jsonl", oracle=args.oracle_diff)
        print("EVAL", tag, json.dumps(m, indent=1), flush=True)
        (out / f"eval_{tag}.json").write_text(json.dumps(m, indent=1))
        return

    if args.init:
        model.load_state_dict(torch.load(args.init))
    ds = Windows(droot / "train", args.win, args.steps * args.pairs, crf_prob=args.crf_prob)
    print(f"train draws {len(ds.items)}", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.pairs, shuffle=False,
                                     num_workers=args.workers, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=max(0.05, 3.0 / args.steps))
    best, t0 = -1.0, time.time()
    run = {"mse": 0.0, "bce": 0.0, "corr": 0.0, "n": 0}
    logp = out / "train.jsonl"
    model.train()
    for step, (x, t, m) in enumerate(dl, 1):
        x = to_input(x.flatten(0, 1).cuda(non_blocking=True))        # [2P,3,F,H,W], rows = w,c,w,c..
        gmax = math.exp(math.log(args.gamma0) * max(0.0, 1.0 - step / (args.gamma_frac * args.steps)))
        q = 1.0
        if args.oracle_diff:
            x[0::2] = (x[0::2] - x[1::2]) * ORACLE_GAIN
            x[1::2] = 0.0
        elif args.fade:
            q = min(1.0, max(0.0, (step / args.steps - args.fade_a) / (args.fade_b - args.fade_a)))
            n = x.shape[0] // 2
            beta = torch.full((n, 1, 1, 1, 1), q, device=x.device)
            g = torch.full((n, 1, 1, 1, 1), ORACLE_GAIN ** (1.0 - q), device=x.device)
            if 0.0 < q < 1.0:          # keep some pairs at the true blind condition
                true = torch.rand(n, device=x.device) < 0.25
                beta[true], g[true] = 1.0, 1.0
            d = x[0::2] - x[1::2]
            x[1::2] = beta * x[1::2]
            x[0::2] = x[1::2] + g * d
        elif gmax > 1.0:
            g = torch.exp(torch.rand(x.shape[0] // 2, 1, 1, 1, 1, device=x.device) * math.log(gmax))
            x[0::2] = (x[1::2] + g * (x[0::2] - x[1::2])).clamp(-1, 1)
        t = t.flatten(0, 1).cuda()
        m = m.flatten(0, 1).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r, e = model(x)
        r, e = r.float(), e.float()
        mse = F.mse_loss(r, t)
        bce = F.binary_cross_entropy_with_logits(e, (m >= 0.5).float())
        loss = mse + 0.5 * bce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            sel = m >= 0.5
            if sel.sum() > 8:
                a, b = r[sel], t[sel]
                run["corr"] += float(((a - a.mean()) * (b - b.mean())).mean()
                                     / (a.std() * b.std() + 1e-8))
            run["mse"] += float(mse) / max(float((t ** 2).mean()), 1e-8)   # 1.0 == predicting zero
            run["bce"] += float(bce)
            run["n"] += 1
        if step % 50 == 0:
            n = run["n"]
            print(json.dumps({"step": step, "nmse": round(run["mse"] / n, 4), "bce": round(run["bce"] / n, 4),
                              "corr": round(run["corr"] / n, 4), "gmax": round(gmax, 2), "fade_q": round(q, 3), "sec": round(time.time() - t0, 1)}), flush=True)
            run = {"mse": 0.0, "bce": 0.0, "corr": 0.0, "n": 0}
        if step % args.eval_every == 0 or step == args.steps:
            mv = evaluate(model, droot / "validation", limit=args.val_limit, oracle=args.oracle_diff)
            model.train()
            rec = {"step": step, "sec": round(time.time() - t0, 1), **mv}
            with open(logp, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("VAL", json.dumps(rec), flush=True)
            # joint selection (lesson from S2): decoding AND accept-at-1%-FPR must both hold
            score = mv["bit_acc_mean"] + mv["exact_and_accept"]
            if score > best:
                best = score
                torch.save(model.state_dict(), out / "best.pt")
            torch.save(model.state_dict(), out / "last.pt")
        if step >= args.steps:
            break
    model.load_state_dict(torch.load(out / "best.pt"))
    fin = {"clean_channel": evaluate(model, droot / "validation", dump=out / "final_val_rows.jsonl",
                                     oracle=args.oracle_diff)}
    print("FINAL", json.dumps(fin, indent=1), flush=True)
    (out / "final_val.json").write_text(json.dumps(fin, indent=1))


if __name__ == "__main__":
    main()
