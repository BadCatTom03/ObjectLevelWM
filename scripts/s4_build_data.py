"""S4 closed-loop writer data builder: mask from the S2 locator on the terminal latent.

Mirrors s3_build_data.py exactly, except the write mask comes from the frozen S2
locator applied to the *terminal sampling latent itself* (z/z.std(), class query),
not from the gold annotation rasterisation. No decode->RGB->locate roundtrip:
the locator reads the same tensor the write happens on, which is the write point
a production graph node would use (sampler output latent -> locator -> write ->
decode).

Per source:
  1. z [1,24,T,H,W] terminal latent (native_latent.pt, never RGB-reencoded)
  2. pred = sigmoid(Locator(z/z.std(), class)) -> [T,H,W]; mask_vol = pred >= tau
  3. support box = union over time (support_box, same min-size rule as S3)
  4. write K keyed PRC carriers  delta = alpha * zstd * (carrier x pred in box)
  5. one batched decode [clean, w_1..w_K]; crop support box to uint8
  6. save S3-schema dir so s3_train_reader.evaluate() runs unchanged
meta.json records locator-vs-gold mask IoU and box IoU for the closed-loop claim.

Outputs semantic32_v1/s4_reader/data_s4/<split>/<sid>/{clean,w<k>,target<k>,mask}.npy
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t0_g1g2_probe import N_CHIPS, load_vae  # noqa: E402
from s2_train_locator import LAB, Locator, target_volume  # noqa: E402
from s3_build_data import (  # noqa: E402
    N_FRAMES, S32, highpass, lat_views, make_write, null_score, psnr, read_stats,
    sample_key, sample_msg, support_box,
)

OUT = S32 / "s4_reader"
LOCATOR_CKPT = S32 / "s2_locator/best_query.pt"


def load_locator(path, device="cuda"):
    model = Locator(n_cls=33).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def category_index(sid, cats):
    return cats.index(json.loads((LAB / f"{sid}.json").read_text())["category"])


@torch.inference_mode()
def locate(model, z, q):
    """z [1,24,T,H,W] raw terminal latent -> prob volume [T,H,W] float32."""
    zn = (z / z.std().clamp_min(1e-6)).float().to("cuda")
    p = torch.sigmoid(model(zn, torch.tensor([q], device="cuda", dtype=torch.long)))
    return p[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.04)
    ap.add_argument("--tau", type=float, default=0.5, help="locator prob threshold for the support mask")
    ap.add_argument("--locator", default=str(LOCATOR_CKPT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="_s4")
    ap.add_argument("--alpha_table", default="", help="json {sid:{alpha:...}} per-source strength")
    ap.add_argument("--tau_fallback", default="0.4,0.3,0.2", help="retry thresholds if mask is empty")
    ap.add_argument("--baseline", action="store_true", help="also log zero-training correlator reads")
    ap.add_argument("--n_null", type=int, default=16)
    args = ap.parse_args()

    gold = json.loads((S32 / "gold.json").read_text())
    ids = gold["semantic_approved_by_split"][args.split]
    cats = sorted({json.loads((LAB / f"{s}.json").read_text())["category"]
                   for sp in ("train", "validation", "test")
                   for s in gold["semantic_approved_by_split"][sp]})
    assert len(cats) == 32
    if args.limit:
        ids = ids[:: max(1, len(ids) // args.limit)][: args.limit]

    out_dir = OUT / f"data{args.tag}" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    blog = OUT / f"baseline{args.tag}_{args.split}.jsonl"
    alpha_table = {}
    if args.alpha_table:
        alpha_table = {k: float(v["alpha"]) for k, v in
                       json.loads(Path(args.alpha_table).read_text()).items()}
        print(f"alpha_table: {len(alpha_table)} sources", flush=True)
    taus = [args.tau] + [float(t) for t in args.tau_fallback.split(",") if t]
    loc = load_locator(args.locator)
    model = load_vae()
    t_all = time.time()

    for n, sid in enumerate(ids):
        sd = out_dir / sid
        if (sd / "meta.json").exists():
            continue
        t0 = time.time()
        z = torch.load(S32 / "data" / sid / "native_latent.pt", map_location="cpu")
        z = (z["video"] if isinstance(z, dict) else z).float()          # [1,24,T,H,W]
        T = z.shape[2]
        zstd = float(z.std())
        q = category_index(sid, cats)
        tau_used = None
        for tau_try in taus:
            prob = locate(loc, z, q)                                    # [T,H,W]
            vol = (prob >= tau_try).astype(np.float32)
            if vol.any():
                tau_used = tau_try
                break
        if tau_used is None:  # locator found nothing at any threshold
            print("SKIP_EMPTY_MASK", sid, flush=True)
            continue
        if tau_used != args.tau:
            print(f"TAU_FALLBACK {sid} {args.tau}->{tau_used}", flush=True)
        alpha = alpha_table.get(sid, args.alpha)
        y0, y1, x0, x1 = support_box(vol)
        bh, bw = y1 - y0, x1 - x0
        # write strength follows the soft probability, box/hard-support from the threshold
        mask = np.where(vol[:, y0:y1, x0:x1] > 0, prob[:, y0:y1, x0:x1], 0.0).astype(np.float32)

        gvol = target_volume(sid)
        gbox = support_box(gvol)
        gy0, gy1, gx0, gx1 = gbox
        box_iou = float(max(0.0, min(y1, gy1) - max(y0, gy0)) * max(0.0, min(x1, gx1) - max(x0, gx0))
                        / max((y1 - y0) * (x1 - x0) + (gy1 - gy0) * (gx1 - gx0)
                              - max(0.0, min(y1, gy1) - max(y0, gy0)) * max(0.0, min(x1, gx1) - max(x0, gx0)), 1e-9))
        gp = (gvol >= 0.5)
        pp = (vol > 0)
        mask_iou = float((gp & pp).sum() / max((gp | pp).sum(), 1))

        zb = z.repeat(1 + args.k, 1, 1, 1, 1)
        draws = []
        for k in range(args.k):
            key = sample_key(sid, k, "s4")
            msg = sample_msg(key)
            d_n, car, cw = make_write(key, msg, mask)
            zb[k + 1, :, :, y0:y1, x0:x1] += alpha * zstd * torch.from_numpy(d_n)
            draws.append((key, msg, d_n, car, cw))
        with torch.inference_mode():
            y = model.decode(zb.half().cuda())
        rgb = (y.permute(0, 2, 3, 4, 1).float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        del y
        torch.cuda.empty_cache()

        py0, py1, px0, px1 = y0 * 16, y1 * 16, x0 * 16, x1 * 16
        clean = np.ascontiguousarray(rgb[0][:, py0:py1, px0:px1])
        sd.mkdir(parents=True, exist_ok=True)
        np.save(sd / "clean.npy", clean)
        np.save(sd / "mask.npy", mask.astype(np.float16))
        tf = np.round(np.arange(N_FRAMES) * (T - 1) / (N_FRAMES - 1)).astype(int)
        wpx = np.repeat(np.repeat(mask[tf] >= 0.5, 16, axis=1), 16, axis=2).astype(np.float32)

        meta = {"sid": sid, "split": args.split, "alpha": alpha, "zstd": zstd,
                "box_cells": [y0, y1, x0, x1], "T": T, "tag": "s4_locator",
                "locator": str(args.locator), "tau": tau_used, "query_class": cats[q],
                "mask_iou_vs_gold": mask_iou, "box_iou_vs_gold": box_iou,
                "gold_box_cells": list(gbox), "draws": []}
        c_raw, nulls = None, {"raw": [], "hp": []}
        if args.baseline:
            c_raw = lat_views(model, clean, T, bh, bw)
            for j in range(args.n_null):
                nk = sample_key(sid, j, "s3null")
                nulls["raw"].append(null_score(c_raw, nk))
                nulls["hp"].append(null_score(highpass(c_raw), nk))
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
                       "raw": read_stats(w_raw, car, key, cw, msg),
                       "hp": read_stats(highpass(w_raw), car, key, cw, msg),
                       "null_raw": nulls["raw"], "null_hp": nulls["hp"]}
                with blog.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        (sd / "meta.json").write_text(json.dumps(meta, indent=1))   # written last = done marker
        print(f"[{n + 1}/{len(ids)}] {sid} box {bh}x{bw} iou mask/box {mask_iou:.3f}/{box_iou:.3f} "
              f"psnr full/box/mask {meta['draws'][0]['psnr_full']}/{meta['draws'][0]['psnr_box']}/"
              f"{meta['draws'][0]['psnr_mask']} elapsed {(time.time() - t_all) / 60:.1f}m", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
