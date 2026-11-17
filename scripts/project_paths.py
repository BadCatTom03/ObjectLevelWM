"""Filesystem configuration shared by the training entry points."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("OBJECTLEVELWM_DATA", PROJECT_ROOT / "data" / "semantic32_v1")).expanduser()
COMFY_ROOT = Path(os.environ["H3_COMFY"]).expanduser() if os.environ.get("H3_COMFY") else None
VAE_CHECKPOINT = Path(os.environ.get(
    "H3_VAE_CHECKPOINT", PROJECT_ROOT / "checkpoints" / "MiniMax-H3-video_vae.safetensors"
)).expanduser()
