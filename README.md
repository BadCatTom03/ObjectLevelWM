# ObjectLevelWM

Training code for **Object-Level Latent Watermarking for Video Diffusion Models with Blind Open-Scan Extraction**.

**Authors:** Haoyu Yang, Ke Niu, Shizhe Ru, and Qiang Huang.  
**Affiliation:** Engineering University of PAP, Xi'an, China.  
**Corresponding author:** Ke Niu — niuke@stu.xjtu.edu.cn.

ObjectLevelWM binds a 32-bit message to a semantic object's support in the terminal latent of MiniMax-H3. A class-conditioned locator predicts the support directly from the latent. A keyed carrier is added inside that support, and the frozen native VAE renders the watermarked video. A separately trained RGB reader predicts a carrier residual and an evidence map.

This repository provides the locator and reader training pipelines, paired-data preparation, the box-local carrier and message codec, and the source-level dataset split utility. Training sources come from the experiment server; the release centralizes filesystem configuration and loads native VAE dependencies when preparing data. Source provenance is recorded in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

## Repository layout

| File | Role |
| --- | --- |
| `scripts/project_paths.py` | Data, ComfyUI, and VAE checkpoint paths. |
| `scripts/s2_train_locator.py` | Class-conditioned 3-D latent locator and its training/validation loop. |
| `scripts/s3_build_data.py` | Clean/watermarked crop pairs from annotated semantic support. |
| `scripts/s3_train_reader.py` | RGB reader, host fade-in curriculum, and crop-level validation. |
| `scripts/s4_build_data.py` | Paired crops written under the trained locator's support. |
| `scripts/t0_g1g2_probe.py` | Shared box-local carrier, 256-chip codec, codec round trips, and native VAE helpers. |
| `scripts/freeze_splits.py` | Source-disjoint 6/2/2 split for each of 32 categories. |
| `tests/test_training_components.py` | CPU checks for carrier decoding, network outputs, and data-pair loading. |

The complete RGB open-scan receiver and S5 evaluation runners are maintained separately from this training release. The evaluation built into `s3_train_reader.py` reads isolated crops at their known box size; its detection thresholds belong to that crop-level protocol. The full-canvas open-scan threshold reported in the paper is calibrated over its own search family.

## Installation

Use Python 3.10 or newer. Training runs on a CUDA GPU; the component checks run on CPU.

```bash
git clone https://github.com/BadCatTom03/ObjectLevelWM.git
cd ObjectLevelWM
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. Install the CUDA build of PyTorch appropriate for your system using the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

Paired-data generation uses a MiniMax-H3-enabled ComfyUI installation exposing `comfy.ldm.minimax.vae.MiniMaxH3VideoVAE`, plus `MiniMax-H3-video_vae.safetensors`. Configure their locations before preparing crops. Training with prepared crops uses PyTorch and the shared carrier helpers.

## Paths

The following variables are read when an entry point starts:

| Variable | Meaning | Default |
| --- | --- | --- |
| `OBJECTLEVELWM_DATA` | Root containing `gold.json`, source latents, labels, and training crops. | `data/semantic32_v1` inside this repository. |
| `H3_COMFY` | ComfyUI directory containing the `comfy/` package. | Python's existing module search path. |
| `H3_VAE_CHECKPOINT` | Native MiniMax-H3 VAE checkpoint. | `checkpoints/MiniMax-H3-video_vae.safetensors`. |

For example, on Linux:

```bash
export OBJECTLEVELWM_DATA=/path/to/semantic32_v1
export H3_COMFY=/path/to/ComfyUI
export H3_VAE_CHECKPOINT=/path/to/MiniMax-H3-video_vae.safetensors
```

In PowerShell, use `$env:OBJECTLEVELWM_DATA = 'D:/datasets/semantic32_v1'` and set the other two variables in the same way.

## Data format

The study uses 320 generated clips from 32 categories, with 10 clips per category and source-disjoint train/validation/test splits of 192/64/64. Each source has 107 RGB frames at 1344×768; its terminal native latent has shape `[1,24,32,48,84]`.

Place prepared inputs under `OBJECTLEVELWM_DATA`:

```text
semantic32_v1/
  gold.json
  splits.json
  labels/
    <source_id>.json
    <source_id>_masks.npz
  data/
    <source_id>/native_latent.pt
  s3_reader/data/
    train/<source_id>/...
    validation/<source_id>/...
    test/<source_id>/...
```

- `gold.json` contains `semantic_approved_by_split`, with `train`, `validation`, and `test` lists of source IDs.
- `splits.json` contains a `rows` array; each row identifies a source with `id` and `split`.
- `labels/<source_id>.json` contains its `category`. Category IDs follow the sorted category names across the split lists.
- `<source_id>_masks.npz` contains RGB-space masks named `f0`, `f35`, `f71`, and `f106`, each with shape `[768,1344]`. The locator averages each 16×16 cell and interpolates the four masks to 32 latent-time positions.
- `native_latent.pt` stores the native terminal tensor directly, or in a dictionary under `video`.

For a new dataset, `python scripts/freeze_splits.py` creates `splits.json` from a `manifest.json` array. Each manifest row supplies `id`, `category`, `category_index`, `clip_index`, `position_request`, and `seed`. The utility uses the experiment's five-entry camera schedule (`static`, `tracking`, `static`, `pan`, `static`) and assigns one static and one moving-camera clip per category to each held-out split. `gold.json` records the reviewed source IDs assigned to those splits.

The paired-data builder writes these files for each source:

| File | Contents |
| --- | --- |
| `clean.npy` | Clean crop, uint8 RGB `[107,16*bh,16*bw,3]`. |
| `w<k>.npy` | Corresponding watermarked crop in the same format. |
| `target<k>.npy` | Signed carrier regression target `[32,bh,bw]`. |
| `mask.npy` | Semantic support weights `[32,bh,bw]`. |
| `meta.json` | Source ID, split, write box, strength, and each draw's index, key, and 32-bit message. |

Dataset tensors and pretrained checkpoints are supplied through these local paths. The GitHub release contains the training source and format documentation.

## Train the semantic locator

Run commands from the repository root:

```bash
python scripts/s2_train_locator.py \
  --ablation query --epochs 150 --bs 4 --lr 2e-4 \
  --null_frac 0.25 --wrong_frac 0.1
```

The locator is a 10.33M-parameter 3-D U-Net with a 96-dimensional class embedding and FiLM conditioning. Training combines soft Dice and weighted BCE, uses AdamW and a OneCycle schedule, and includes horizontal flips. NULL queries supply empty-support supervision; wrong-class queries teach suppression. Checkpoint selection combines validation mIoU and wrong-query suppression with an mIoU floor of 0.60. Outputs are written to `s2_locator/` under the data root.

```bash
python scripts/s2_train_locator.py \
  --eval_only --split test \
  --ckpt "$OBJECTLEVELWM_DATA/s2_locator/best_query.pt"
```

Use `--ablation none` to train the class-agnostic saliency comparison.

## Prepare reader-training pairs

Prepare two keyed message draws per training source and one per validation/test source:

```bash
python scripts/s3_build_data.py --split train --k 2 --alpha 0.04
python scripts/s3_build_data.py --split validation --k 1 --alpha 0.04
python scripts/s3_build_data.py --split test --k 1 --alpha 0.04
```

Each draw encodes 32 payload bits into 256 keyed, interleaved chips. A carrier is generated in box-local integer coordinates and normalized under the semantic mask. The builder writes into the original native latent, decodes the clean and marked examples together, and saves matching crops. `--alpha` sets the preparation strength; the commands above use the script's 0.04 setting.

## Train the RGB reader

```bash
python scripts/s3_train_reader.py \
  --fade --steps 6000 --pairs 2 --win 24 --base 32 \
  --lr 3e-4 --workers 8 --run full
```

The reader has 19.45M parameters at `--base 32`. A 4×4 pixel-unshuffle stem feeds a 3-D convolutional encoder-decoder. It predicts a signed residual and evidence logits on the latent spatial grid, with 32 temporal positions. Its loss is residual MSE plus 0.5 times evidence BCE. AdamW, OneCycle scheduling, bfloat16 autocast, and gradient clipping are used for optimization.

With `--fade`, each positive input is `beta * clean + gain * (watermarked - clean)`. Between 15% and 70% of training, `beta` increases from 0 to 1 and `gain` decreases from 8 to 1. During that transition, 25% of pairs use the final blind condition. The clean member of each pair follows the same host scaling. By the end, the network reads the marked RGB crop directly. The training log records `fade_q`, loss, and residual correlation.

Checkpoints and logs are written to `s3_reader/run_full/`. Crop-level validation is available through:

```bash
python scripts/s3_train_reader.py \
  --eval_only --run full --split validation \
  --ckpt "$OBJECTLEVELWM_DATA/s3_reader/run_full/best.pt"

python scripts/s3_train_reader.py \
  --eval_only --run full --split validation --crf 23 \
  --ckpt "$OBJECTLEVELWM_DATA/s3_reader/run_full/best.pt"
```

On Windows, replace the shell variable in these checkpoint arguments with its absolute path.

## Prepare locator-based writes

After training the locator, generate paired examples using its predicted support:

```bash
python scripts/s4_build_data.py \
  --split validation --k 1 --alpha 0.04 \
  --locator "$OBJECTLEVELWM_DATA/s2_locator/best_query.pt"
```

These examples are written to `s4_reader/data_s4/validation/`. The same command accepts `--split train` and `--split test`. `--alpha_table` accepts a JSON mapping from source IDs to objects containing an `alpha` value for per-source strengths.

## Component checks

```bash
python -m unittest discover -s tests -v
```

The checks exercise the 32-bit carrier round trip, locator query inputs at the native grid size, reader output geometry, paired-crop loading, and path configuration. They use synthetic tensors on CPU. The paired-data builders additionally need the external VAE runtime and checkpoint described above.

## Paper and license

**Object-Level Latent Watermarking for Video Diffusion Models with Blind Open-Scan Extraction**  
Haoyu Yang, Ke Niu, Shizhe Ru, and Qiang Huang.

This project's source is distributed under the [MIT License](LICENSE). MiniMax-H3 weights and the ComfyUI runtime are obtained from their respective distributions.
