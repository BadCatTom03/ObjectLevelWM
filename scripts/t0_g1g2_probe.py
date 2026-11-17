"""T0 round 2: G1 isolated-patch channel and G2 transplant channel.

Carrier is keyed in *support-bbox-local* coordinates, which is what a deployed
blind reader would do: locate the support (box), then demodulate in local
coordinates. So pasting to a new position is legitimate and the codeword
survives if the transform preserves the local cell grid.

G1  z_w -> decode -> crop patch -> encode(patch alone) -> demod (local coords)
G2  z_w -> decode -> crop patch -> paste into unseen host at new position ->
    encode(host) -> demod at pasted bbox (local coords)
Clean control: same G2 pipeline with no write (pipeline-inflation check).
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from project_paths import DATA_ROOT, VAE_CHECKPOINT, COMFY_ROOT

ROOT = DATA_ROOT.parent

N_CHIPS = 256
KEY = b"t0-probe-key-20261111"
BOX = (192, 640, 384, 944)  # y0,y1,x0,x1 RGB; 448x560 px


def _rng(key, label):
    h = hashlib.sha256(key + b"|" + label.encode()).digest()
    return np.random.Generator(np.random.PCG64(list(h[:16])))


def _pad_and_perm(key, n, msg_bits):
    pad = (_rng(key, f"prc|pad{n}").random(n) < 0.5).astype(np.uint8)
    r = n // msg_bits
    assign = np.repeat(np.arange(msg_bits), r).astype(np.int64)
    rest = n - r * msg_bits
    if rest:
        assign = np.concatenate([assign, np.arange(rest) % msg_bits])
    _rng(key, f"prc|perm{n}m{msg_bits}").shuffle(assign)
    return pad, assign


def prc_encode(message, key, n, msg_bits):
    mbits = np.array([(message >> (msg_bits - 1 - i)) & 1 for i in range(msg_bits)], np.uint8)
    pad, assign = _pad_and_perm(key, n, msg_bits)
    return (mbits[assign] ^ pad).astype(np.uint8)


def prc_soft_decode(soft, key, n, msg_bits):
    pad, assign = _pad_and_perm(key, n, msg_bits)
    depad = soft * np.where(pad == 0, 1.0, -1.0)
    llr = np.bincount(assign, weights=depad, minlength=msg_bits)
    bits = (llr > 0).astype(np.uint8)
    z = np.abs(llr) / ((float(soft.std()) + 1e-12) *
                       np.sqrt(np.maximum(np.bincount(assign, minlength=msg_bits), 1)))
    return bits, float(z.mean())


class LocalPRC:
    """Carrier defined on the support bbox sub-grid (T full, H/W cropped).

    Field/blocks are functions of local cell coordinates only, so the same
    codeword demodulates wherever the bbox lands.
    """

    def __init__(self, key, t_len, bh, bw, n_chips=N_CHIPS):
        self.t_len, self.bh, self.bw, self.n = t_len, bh, bw, n_chips
        dim = t_len * bh * bw
        perm = _rng(key, f"local|perm|{t_len}x{bh}x{bw}").permutation(dim)
        self.block = np.empty(dim, np.int64)
        self.block[perm] = (np.arange(dim) % n_chips).astype(np.int64)
        g = _rng(key, f"local|field|{t_len}x{bh}x{bw}").standard_normal(dim).astype(np.float32)
        norms = np.sqrt(np.bincount(self.block, weights=g.astype(np.float64) ** 2, minlength=n_chips))
        self.g = (g / np.maximum(norms[self.block], 1e-12)).astype(np.float32)

    def make_delta(self, codeword):
        s = np.asarray(codeword, np.float32) * 2 - 1
        return (self.g * s[self.block]).reshape(self.t_len, self.bh, self.bw)

    def demodulate(self, residual_bbox):
        r = residual_bbox.reshape(-1).astype(np.float64) * self.g
        return np.bincount(self.block, weights=r, minlength=self.n)


def load_vae():
    if COMFY_ROOT:
        sys.path.insert(0, str(COMFY_ROOT))
    from comfy.ldm.minimax.vae import (
        MiniMaxH3VideoVAE, LATENTS_MEAN, LATENTS_STD, IMAGENET_MEAN, IMAGENET_STD,
    )
    from safetensors.torch import load_file

    p = VAE_CHECKPOINT
    model = MiniMaxH3VideoVAE()
    sd = load_file(str(p), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert set(missing) == {"latents_mean", "latents_std"} and not unexpected
    model.latents_mean = torch.tensor(LATENTS_MEAN)
    model.latents_std = torch.tensor(LATENTS_STD)
    model.pixel_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
    model.pixel_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)
    return model.eval().to(device="cuda", dtype=torch.float16)


def decode(model, z):
    with torch.inference_mode():
        y = model.decode(z.half().cuda())
    return y[0].permute(1, 2, 3, 0).float().cpu().clamp(0, 1)


def encode(model, rgb01):
    x = (rgb01 * 2 - 1).permute(3, 0, 1, 2).unsqueeze(0)
    with torch.inference_mode():
        z = model.encode(x.half().cuda(), device="cuda")
    return z.float().cpu()


def resize_rgb(rgb, h, w):
    x = rgb.permute(3, 0, 1, 2)
    y = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return y.permute(1, 2, 3, 0)


def write_at(z_nat, car, cw, zstd, alpha=0.02):
    d = torch.from_numpy(car.make_delta(cw)).float()  # [T,bh,bw]
    d = d / d.std().clamp_min(1e-12) * (alpha * zstd)
    y0, y1, x0, x1 = BOX
    z_w = z_nat.clone()
    z_w[0, :, :, y0 // 16:y0 // 16 + car.bh, x0 // 16:x0 // 16 + car.bw] += d
    return z_w


def load_z(sid):
    return torch.load(DATA_ROOT / "data" / sid / "native_latent.pt",
                      map_location="cpu", weights_only=True)["video"].float()


def run_g1(model, src, msg, cap, out, alpha):
    z_nat = load_z(src)
    y0, y1, x0, x1 = BOX
    bh, bw = (y1 - y0) // 16, (x1 - x0) // 16
    car = LocalPRC(KEY, z_nat.shape[2], bh, bw)
    cw = prc_encode(msg, KEY, N_CHIPS, cap)
    z_w = write_at(z_nat, car, cw, float(z_nat.std()), alpha)
    rgb_w = decode(model, z_w)
    patch = resize_rgb(rgb_w[:, y0:y1, x0:x1], bh * 16, bw * 16)
    z_p = encode(model, patch)
    # patch encode yields [1,24,T',bh',bw']; crop support to actual size
    T = z_nat.shape[2]
    z_p = z_p[:, :, :T, :bh, :bw]
    rgb_c = decode(model, z_nat)
    patch_c = resize_rgb(rgb_c[:, y0:y1, x0:x1], bh * 16, bw * 16)
    z_c = encode(model, patch_c)[:, :, :T, :bh, :bw]
    soft_w = car.demodulate((z_p - z_c)[0].mean(0).numpy())
    soft_c = car.demodulate(np.zeros((T, bh, bw), np.float32))
    hard = (soft_w > 0).astype(np.uint8)
    bits, score = prc_soft_decode(soft_w, KEY, N_CHIPS, cap)
    tb = np.array([(msg >> (cap - 1 - i)) & 1 for i in range(cap)], np.uint8)
    rec = {"pipeline": "G1", "src": src, "cap": cap, "alpha": alpha,
           "chip_acc": float((hard == cw).mean()), "bit_acc": float((bits == tb).mean()),
           "exact": bool((bits == tb).all()), "score": score}
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def run_g2(model, src, host, scale, msg, cap, out, alpha, crf=None):
    z_nat = load_z(src)
    y0, y1, x0, x1 = BOX
    bh, bw = (y1 - y0) // 16, (x1 - x0) // 16
    car = LocalPRC(KEY, z_nat.shape[2], bh, bw)
    cw = prc_encode(msg, KEY, N_CHIPS, cap)
    z_w = write_at(z_nat, car, cw, float(z_nat.std()), alpha)
    rgb_w = decode(model, z_w)
    rgb_c = decode(model, z_nat)
    psnr_full = psnr(rgb_w.numpy(), rgb_c.numpy())
    psnr_box = psnr(rgb_w[:, y0:y1, x0:x1].numpy(), rgb_c[:, y0:y1, x0:x1].numpy())
    patch = rgb_w[:, y0:y1, x0:x1]
    if scale != 1.0:
        patch = resize_rgb(patch, max(16, int(round((y1 - y0) * scale)) // 16 * 16),
                           max(16, int(round((x1 - x0) * scale)) // 16 * 16))
    ph, pw = patch.shape[1], patch.shape[2]
    if ph > 768 or pw > 1344:  # patch must fit the host frame
        print(f"SKIP {src}->{host} scale={scale}: patch {ph}x{pw} exceeds frame", flush=True)
        return

    z_host = load_z(host)
    host_rgb = decode(model, z_host)
    hy = (768 - ph) // 2 // 16 * 16
    hx = (1344 - pw) // 2 // 16 * 16
    host_rgb = host_rgb.clone()
    host_rgb[:, hy:hy + ph, hx:hx + pw] = patch
    if crf is not None:
        host_rgb = h264_roundtrip(host_rgb, crf)
    z_new = encode(model, host_rgb)
    z_clean = encode(model, decode(model, z_host))

    T = car.t_len  # canonical temporal depth from the write side
    assert z_new.shape[2] >= T and z_clean.shape[2] >= T, (z_new.shape, z_clean.shape)
    bbh, bbw = ph // 16, pw // 16
    # canonical carrier (bbox size before transplant scaling); reader maps the
    # observed support back to canonical grid before demodulating.
    car_h = car
    rz = (z_new[:1, :, :T, hy // 16:hy // 16 + bbh, hx // 16:hx // 16 + bbw]
          - z_clean[:1, :, :T, hy // 16:hy // 16 + bbh, hx // 16:hx // 16 + bbw])
    if (bbh, bbw) != (bh, bw):
        # rz[0]: [24, T, bbh, bbw] -> [T*24, 1, bbh, bbw] for 2D resize
        r4 = rz[0].permute(1, 0, 2, 3).reshape(T * 24, 1, bbh, bbw)
        r4 = F.interpolate(r4, size=(bh, bw), mode="bilinear", align_corners=False)
        rz_canon = r4.reshape(T, 24, bh, bw).permute(1, 0, 2, 3).float()
    else:
        rz_canon = rz[0, :, :T]
    soft_w = car_h.demodulate(rz_canon.mean(0).numpy())

    host_rgb_c = decode(model, z_host)
    host_rgb_c = host_rgb_c.clone()
    host_rgb_c[:, hy:hy + ph, hx:hx + pw] = resize_rgb(rgb_c[:, y0:y1, x0:x1], ph, pw)
    z_cleanpaste = encode(model, host_rgb_c)
    rzc = (z_cleanpaste[:1, :, :T, hy // 16:hy // 16 + bbh, hx // 16:hx // 16 + bbw]
           - z_clean[:1, :, :T, hy // 16:hy // 16 + bbh, hx // 16:hx // 16 + bbw])
    if (bbh, bbw) != (bh, bw):
        rc = rzc[0].permute(1, 0, 2, 3).reshape(T * 24, 1, bbh, bbw)
        rc = F.interpolate(rc, size=(bh, bw), mode="bilinear", align_corners=False)
        rzc_canon = rc.reshape(T, 24, bh, bw).permute(1, 0, 2, 3).float()
    else:
        rzc_canon = rzc[0, :, :T]
    soft_c = car_h.demodulate(rzc_canon.mean(0).numpy())

    hard = (soft_w > 0).astype(np.uint8)
    bits, score = prc_soft_decode(soft_w, KEY, N_CHIPS, cap)
    tb = np.array([(msg >> (cap - 1 - i)) & 1 for i in range(cap)], np.uint8)
    rec = {"pipeline": "G2", "src": src, "host": host, "scale": scale, "cap": cap,
           "alpha": alpha, "chip_acc": float((hard == cw).mean()),
           "bit_acc": float((bits == tb).mean()), "exact": bool((bits == tb).all()),
           "score": score, "clean_score": float(prc_soft_decode(soft_c, KEY, N_CHIPS, cap)[1]),
           "psnr_full": round(psnr_full, 2), "psnr_box": round(psnr_box, 2),
           "crf": crf}
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def h264_roundtrip(rgb01, crf):
    """encode rgb01 [T,H,W,3] float 0-1 -> H.264 (yuv420p, libx264) -> decode back."""
    import av
    import io as _io
    u8 = (rgb01.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    T, h, w, _ = u8.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    u8 = np.ascontiguousarray(u8[:, :h2, :w2])
    bio = _io.BytesIO()
    out = av.open(bio, mode="w", format="mp4")
    st = out.add_stream("libx264", rate=24)
    st.width, st.height, st.pix_fmt = w2, h2, "yuv420p"
    st.options = {"crf": str(crf), "preset": "medium"}
    for f in u8:
        for pkt in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24").reformat(format="yuv420p")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    bio.seek(0)

    frames = []
    with av.open(bio, mode="r") as cin:
        for fr in cin.decode(video=0):
            frames.append(fr.to_ndarray(format="rgb24"))
    arr = np.stack(frames[:T])
    res = torch.from_numpy(arr.astype(np.float32) / 255.0)
    res = F.interpolate(res.permute(0, 3, 1, 2), size=(h, w), mode="bilinear",
                        align_corners=False).permute(0, 2, 3, 1).contiguous()
    return res


def done_keys(out):
    keys = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((r.get("pipeline"), r.get("src"), r.get("host"),
                      r.get("scale"), r.get("cap"), r.get("alpha"), r.get("crf")))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["car_02", "dog_01"])
    ap.add_argument("--hosts", nargs="+", default=["airplane_01", "boat_01"])
    ap.add_argument("--scales", default="1.0,0.5")
    ap.add_argument("--caps", default="16")
    ap.add_argument("--alphas", default=None, help="comma list; overrides --alpha")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--g2-only", action="store_true")
    ap.add_argument("--crf", type=int, default=None, help="H.264 CRF; if set, G2 runs G3 path")
    ap.add_argument("--out", default=str(ROOT / "t0_channel_probe" / "g1g2.jsonl"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = load_vae()
    caps = [int(c) for c in args.caps.split(",")]
    scales = [float(s) for s in args.scales.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else [args.alpha]
    done = done_keys(out)
    if not args.g2_only:
        for src in args.sources:
            for cap in caps:
                if ("G1", src, None, None, cap, args.alpha, None) in done:
                    continue
                run_g1(model, src, int(np.random.default_rng(7).integers(0, 2 ** cap - 1)), cap, out, args.alpha)
    for src in args.sources:
        for host in args.hosts:
            for scale in scales:
                for cap in caps:
                    for alpha in alphas:
                        if ("G2", src, host, scale, cap, alpha, args.crf) in done:
                            continue
                        run_g2(model, src, host, scale,
                               int(np.random.default_rng(11).integers(0, 2 ** cap - 1)),
                               cap, out, alpha, crf=args.crf)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
