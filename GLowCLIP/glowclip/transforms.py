from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def decode_rgb(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation, convert to RGB, and drop attached metadata."""
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    return Image.fromarray(np.asarray(oriented, dtype=np.uint8).copy(), "RGB")


def letterbox(
    image: Image.Image, size: int, fill: tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")
    scale = min(size / width, size / height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    canvas = Image.new("RGB", (size, size), fill)
    left = (size - resized.width) // 2
    top = (size - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


class CLIPImageTransform:
    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size
        self.mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(CLIP_STD, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = letterbox(decode_rgb(image), self.image_size)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        tensor = torch.from_numpy(array)
        return (tensor - self.mean) / self.std
