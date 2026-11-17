"""S3 data builder + blind-reading baseline (S3.0).

For each source video:
  1. take the semantic target volume (S0 masks -> latent grid, same as S2),
     union bbox over time = support box (grid aligned, cell units);
  2. write K keyed carriers  delta = alpha * zstd * (PRC carrier x soft mask),
     carrier keyed in *bbox-local* coordinates (T0 convention), one fresh key +
     random 32-bit message per draw;
  3. decode [clean, w_1..w_K] in ONE batch (fp16 common-mode error cancels,
     and clean negatives share the exact decode pipeline with positives);
  4. crop the support box from RGB (C1 isolated local), quantize to uint8,
     save crops + regression targets for the learned reader (S3.1).

With --baseline the isolated crops are additionally read back through the
frozen H3 VAE encoder and the keyed correlator, three ways:
  ref       (z_w - z_clean)            clean-reference oracle == T0 G1 protocol
  raw       z_w                        blind, no host suppression
  hp        z_w - local 3D mean        blind, high-pass host suppression
plus null scores (clean crop / wrong keys) for FPR calibration, and optionally
the same after an H.264 round trip (--crf).

Outputs  semantic32_v1/s3_reader/data/<split>/<sid>/{clean.npy,w<k>.npy,
target<k>.npy,mask.npy,meta.json}  and  s3_reader/baseline_<split>.jsonl
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t0_g1g2_probe import (  # noqa: E402
    N_CHIPS, LocalPRC, encode, h264_roundtrip, load_vae, prc_encode, prc_soft_decode,
)
from s2_train_locator import target_volume  # noqa: E402

from project_paths import DATA_ROOT

S32 = DATA_ROOT
DATA = S32 / "data"
OUT = S32 / "s3_reader"
MSG_BITS = 32
N_FRAMES = 107


def sample_key(sid, k, tag="s3"):
    return f"{tag}|{sid}|{k}".encode()


def sample_msg(key):
    h = hashlib.sha256(key + b"|msg").digest()
    return int.from_bytes(h[:4], "big")


def msg_bits(msg):
    return np.array([(msg >> (MSG_BITS - 1 - i)) & 1 for i in range(MSG_BITS)], np.uint8)


def support_box(vol, thr=0.25, min_cells=8):
    """vol [T,H,W] -> (y0,y1,x0,x1) in cells, union over time."""
    u = vol.max(0) >= thr
    ys, xs = np.where(u)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    H, W = u.shape

    def grow(a, b, lim):
        while b - a < min_cells:
            if a > 0:
                a -= 1
            if b - a < min_cells and b < lim:
                b += 1
        return a, b
    y0, y1 = grow(y0, y1, H)
    x0, x1 = grow(x0, x1, W)
    return y0, y1, x0, x1


def make_write(key, msg, mask):
    """mask [T,bh,bw] soft. Returns (unit-RMS-in-mask signed pattern, carrier, codeword)."""
    T, bh, bw = mask.shape
    car = LocalPRC(key, T, bh, bw)
    cw = prc_encode(msg, key, N_CHIPS, MSG_BITS)
    d = car.make_delta(cw) * mask
    rms = np.sqrt((d ** 2).sum() / max(float((mask ** 2).sum()), 1e-6))
    return (d / max(rms, 1e-12)).astype(np.float32), car, cw


def highpass(x, kt=3, ks=5):
    """x [T,h,w] numpy -> x minus local 3D box mean (edge-normalised)."""
    t = torch.from_numpy(x)[None, None].float()
    pad = (ks // 2, ks // 2, ks // 2, ks // 2, kt // 2, kt // 2)
    num = F.avg_pool3d(F.pad(t, pad), (kt, ks, ks), stride=1)
    den = F.avg_pool3d(F.pad(torch.ones_like(t), pad), (kt, ks, ks), stride=1)
    return (t - num / den)[0, 0].numpy()


def read_stats(est, car, key, cw, msg):
    soft = car.demodulate(est)
    bits, score = prc_soft_decode(soft, key, N_CHIPS, MSG_BITS)
    tb = msg_bits(msg)
    return {"chip_acc": float(((soft > 0).astype(np.uint8) == cw).mean()),
            "bit_acc": float((bits == tb).mean()),
            "exact": bool((bits == tb).all()), "score": score}


def null_score(est, key):
    T, bh, bw = est.shape
    car = LocalPRC(key, T, bh, bw)
    return prc_soft_decode(car.demodulate(est), key, N_CHIPS, MSG_BITS)[1]


def psnr(a, b, w=None):
    d2 = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    mse = float(d2.mean()) if w is None else float((d2.mean(-1) * w).sum() / max(w.sum(), 1.0))
    return round(float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9))), 2)


def lat_views(model, crop_u8, T, bh, bw):
    """uint8 crop [F,ph,pw,3] -> channel-mean latent [T,bh,bw] via frozen VAE encoder."""
    z = encode(model, torch.from_numpy(crop_u8).float() / 255.0)
    assert z.shape[2] >= T and z.shape[3] >= bh and z.shape[4] >= bw, z.shape
    return z[0, :, :T, :bh, :bw].mean(0).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--k", type=int, default=1, help="watermarked draws per source")
    ap.add_argument("--alpha", type=float, default=0.04)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--crf", type=int, default=None)
    ap.add_argument("--n_null", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ids", default="", help="comma list to rebuild only these sources")
    args = ap.parse_args()

    gold = json.loads((S32 / "gold.json").read_text())
    ids = gold["semantic_approved_by_split"][args.split]
    if args.ids:
        want = set(args.ids.split(","))
        ids = [i for i in ids if i in want]
    if args.limit:
        ids = ids[:: max(1, len(ids) // args.limit)][: args.limit]
    out_dir = OUT / f"data{args.tag}" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    blog = OUT / f"baseline{args.tag}_{args.split}.jsonl"
    model = load_vae()
    t_all = time.time()

    for n, sid in enumerate(ids):
        sd = out_dir / sid
        if (sd / "meta.json").exists():
            continue
        t0 = time.time()
        z = torch.load(DATA / sid / "native_latent.pt", map_location="cpu")
        z = (z["video"] if isinstance(z, dict) else z).float()          # [1,24,T,H,W]
        T = z.shape[2]
        zstd = float(z.std())
        vol = target_volume(sid)                                         # [T,H,W]
        y0, y1, x0, x1 = support_box(vol)
        bh, bw = y1 - y0, x1 - x0
        mask = vol[:, y0:y1, x0:x1].astype(np.float32)

        zb = z.repeat(1 + args.k, 1, 1, 1, 1)
        draws = []
        for k in range(args.k):
            key = sample_key(sid, k)
            msg = sample_msg(key)
            d_n, car, cw = make_write(key, msg, mask)
            zb[k + 1, :, :, y0:y1, x0:x1] += args.alpha * zstd * torch.from_numpy(d_n)
            draws.append((key, msg, d_n, car, cw))
        with torch.inference_mode():
            y = model.decode(zb.half().cuda())                           # [B,3,F,H,W]
        rgb = (y.permute(0, 2, 3, 4, 1).float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        del y
        torch.cuda.empty_cache()
        t_dec = time.time() - t0

        py0, py1, px0, px1 = y0 * 16, y1 * 16, x0 * 16, x1 * 16
        clean = np.ascontiguousarray(rgb[0][:, py0:py1, px0:px1])
        sd.mkdir(parents=True, exist_ok=True)
        np.save(sd / "clean.npy", clean)
        np.save(sd / "mask.npy", mask.astype(np.float16))
        # pixel weights for target-region PSNR: frame f <-> latent t
        tf = np.round(np.arange(N_FRAMES) * (T - 1) / (N_FRAMES - 1)).astype(int)
        wpx = np.repeat(np.repeat(mask[tf] >= 0.5, 16, axis=1), 16, axis=2).astype(np.float32)

        meta = {"sid": sid, "split": args.split, "alpha": args.alpha, "zstd": zstd,
                "box_cells": [y0, y1, x0, x1], "T": T, "draws": []}
        if args.baseline:
            c_raw = lat_views(model, clean, T, bh, bw)
            c_hp = highpass(c_raw)
            nulls = {"raw": [], "hp": []}
            for j in range(args.n_null):
                nk = sample_key(sid, j, "s3null")
                nulls["raw"].append(null_score(c_raw, nk))
                nulls["hp"].append(null_score(c_hp, nk))
        for k, (key, msg, d_n, car, cw) in enumerate(draws):
            crop = np.ascontiguousarray(rgb[k + 1][:, py0:py1, px0:px1])
            np.save(sd / f"w{k}.npy", crop)
            np.save(sd / f"target{k}.npy", d_n.astype(np.float16))
            dr = {"k": k, "key": key.decode(), "msg": msg,
                  "psnr_full": psnr(rgb[k + 1], rgb[0]),
                  "psnr_box": psnr(crop, clean),
                  "psnr_mask": psnr(crop, clean, wpx)}
            meta["draws"].append(dr)
            if args.baseline:
                w_raw = lat_views(model, crop, T, bh, bw)
                rec = {"sid": sid, **dr, "bh": bh, "bw": bw,
                       "mask_cells": float((mask >= 0.5).sum() / T),
                       "ref": read_stats(w_raw - c_raw, car, key, cw, msg),
                       "raw": read_stats(w_raw, car, key, cw, msg),
                       "hp": read_stats(highpass(w_raw), car, key, cw, msg),
                       # clean crop read with the true key (paired negative)
                       "clean_raw": read_stats(c_raw, car, key, cw, msg)["score"],
                       "clean_hp": read_stats(c_hp, car, key, cw, msg)["score"],
                       # watermarked crop read with wrong keys (C6 wrong-key)
                       "wrongkey_hp": [null_score(highpass(w_raw), sample_key(sid, j, "s3wrong"))
                                       for j in range(4)],
                       "null_raw": nulls["raw"], "null_hp": nulls["hp"]}
                if args.crf is not None:
                    cq = h264_roundtrip(torch.from_numpy(crop).float() / 255.0, args.crf)
                    cq = (cq.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
                    cc = h264_roundtrip(torch.from_numpy(clean).float() / 255.0, args.crf)
                    cc = (cc.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
                    q_raw = lat_views(model, cq, T, bh, bw)
                    qc_raw = lat_views(model, cc, T, bh, bw)
                    rec[f"crf{args.crf}"] = {
                        "ref": read_stats(q_raw - qc_raw, car, key, cw, msg),
                        "raw": read_stats(q_raw, car, key, cw, msg),
                        "hp": read_stats(highpass(q_raw), car, key, cw, msg),
                        "clean_hp": read_stats(highpass(qc_raw), car, key, cw, msg)["score"]}
                with blog.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print("BASE", sid, k, "ref", rec["ref"]["chip_acc"], rec["ref"]["exact"],
                      "| raw", round(rec["raw"]["chip_acc"], 3), rec["raw"]["exact"],
                      "| hp", round(rec["hp"]["chip_acc"], 3), rec["hp"]["exact"],
                      round(rec["hp"]["score"], 2), "clean", round(rec["clean_hp"], 2), flush=True)
        (sd / "meta.json").write_text(json.dumps(meta, indent=1))   # written last = done marker
        print(f"[{n + 1}/{len(ids)}] {sid} box {bh}x{bw} dec {t_dec:.1f}s total {time.time() - t0:.1f}s "
              f"psnr full/box/mask {meta['draws'][0]['psnr_full']}/{meta['draws'][0]['psnr_box']}/"
              f"{meta['draws'][0]['psnr_mask']} elapsed {(time.time() - t_all) / 60:.1f}m", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
