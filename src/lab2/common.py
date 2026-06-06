from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def auto_detect_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def create_timestamped_output_dir(base_dir: str, subdir: str, name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / subdir / f"{timestamp}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def prepare_decoder_images(images: torch.Tensor) -> torch.Tensor:
    """
    Convert renderer outputs to decoder input shape [K, H, W].

    Mitsuba rendering may return RGB images [K, H, W, 3], while decoders expect
    grayscale observations [K, H, W]. This helper keeps both Mitsuba forward
    rendering and Mitsuba autodiff rendering on the same decoder interface.
    """
    if images.ndim == 3:
        return images
    if images.ndim == 4 and images.shape[-1] == 3:
        return images.mean(dim=-1)
    raise ValueError(f"Unsupported image tensor shape for decoder: {tuple(images.shape)}")
