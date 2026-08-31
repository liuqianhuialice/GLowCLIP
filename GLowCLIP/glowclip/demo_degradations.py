from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .transforms import decode_rgb

JPEG_COMPRESSION = "JPEG Compression"
GAUSSIAN_BLUR = "Gaussian Blur"
RESIZE = "Resize"
GAUSSIAN_NOISE = "Gaussian Noise"
COLOR_JITTER = "Color Jitter"
CENTER_CROP = "Center Crop"

DEGRADATION_ORDER = (
    JPEG_COMPRESSION,
    GAUSSIAN_BLUR,
    RESIZE,
    GAUSSIAN_NOISE,
    COLOR_JITTER,
    CENTER_CROP,
)

JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)
COLOR_JITTER_AMOUNT = 0.20
CENTER_CROP_FRACTION = 0.80


@dataclass(frozen=True)
class DemoDegradationConfig:
    jpeg_quality: int = 70
    blur_sigma: float = 1.0
    resize_scale: float = 0.5
    noise_sigma: float = 0.05
    color_brightness: float = COLOR_JITTER_AMOUNT
    color_contrast: float = COLOR_JITTER_AMOUNT
    seed: int = 42

    def validate(self) -> None:
        if self.jpeg_quality not in JPEG_QUALITIES:
            raise ValueError(f"JPEG quality must be one of {JPEG_QUALITIES}")
        if self.blur_sigma not in BLUR_SIGMAS:
            raise ValueError(f"Blur sigma must be one of {BLUR_SIGMAS}")
        if self.resize_scale not in RESIZE_SCALES:
            raise ValueError(f"Resize scale must be one of {RESIZE_SCALES}")
        if self.noise_sigma not in NOISE_SIGMAS:
            raise ValueError(f"Noise sigma must be one of {NOISE_SIGMAS}")
        if not -COLOR_JITTER_AMOUNT <= self.color_brightness <= COLOR_JITTER_AMOUNT:
            raise ValueError("Color brightness must be between -0.20 and 0.20")
        if not -COLOR_JITTER_AMOUNT <= self.color_contrast <= COLOR_JITTER_AMOUNT:
            raise ValueError("Color contrast must be between -0.20 and 0.20")


@dataclass(frozen=True)
class DemoDegradationResult:
    image: Image.Image
    operations: tuple[str, ...]


def apply_demo_degradations(
    image: Image.Image,
    selected: Iterable[str],
    config: DemoDegradationConfig | None = None,
) -> DemoDegradationResult:
    """Apply the user-selected demo degradations in a fixed, documented order."""
    config = config or DemoDegradationConfig()
    config.validate()
    selected_set = set(selected)
    unknown = sorted(selected_set.difference(DEGRADATION_ORDER))
    if unknown:
        raise ValueError(f"Unsupported degradation(s): {', '.join(unknown)}")

    output = decode_rgb(image)
    operations: list[str] = []

    if JPEG_COMPRESSION in selected_set:
        output = _jpeg_compression(output, config.jpeg_quality)
        operations.append(f"JPEG Compression (quality={config.jpeg_quality})")

    if GAUSSIAN_BLUR in selected_set:
        output = output.filter(ImageFilter.GaussianBlur(radius=config.blur_sigma))
        operations.append(f"Gaussian Blur (sigma={config.blur_sigma:g})")

    if RESIZE in selected_set:
        output = _downscale_then_upscale(output, config.resize_scale)
        operations.append(f"Resize ({config.resize_scale:g}x downscale, then upscale)")

    if GAUSSIAN_NOISE in selected_set:
        output = _gaussian_noise(output, config.noise_sigma, config.seed)
        operations.append(f"Gaussian Noise (sigma={config.noise_sigma:.2f})")

    if COLOR_JITTER in selected_set:
        output, description = _color_jitter(
            output,
            config.color_brightness,
            config.color_contrast,
        )
        operations.append(description)

    if CENTER_CROP in selected_set:
        output = _center_crop(output, CENTER_CROP_FRACTION)
        operations.append("Center Crop (80% of width and height)")

    if not operations:
        operations.append("No degradation (original image)")
    return DemoDegradationResult(output.convert("RGB"), tuple(operations))


def _jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _downscale_then_upscale(image: Image.Image, scale: float) -> Image.Image:
    original_size = image.size
    reduced_size = (
        max(1, round(original_size[0] * scale)),
        max(1, round(original_size[1] * scale)),
    )
    reduced = image.resize(reduced_size, Image.Resampling.LANCZOS)
    return reduced.resize(original_size, Image.Resampling.BICUBIC)


def _seeded_rng(seed: int, stream: int) -> np.random.Generator:
    normalized_seed = int(seed) % (2**32)
    return np.random.default_rng(np.random.SeedSequence([normalized_seed, stream]))


def _gaussian_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    array = np.asarray(image, dtype=np.float32) / 255.0
    noise = _seeded_rng(seed, stream=1).standard_normal(
        size=array.shape, dtype=np.float32
    )
    array = np.clip(array + noise * sigma, 0.0, 1.0)
    return Image.fromarray((array * 255.0).round().astype(np.uint8), "RGB")


def _color_jitter(
    image: Image.Image,
    brightness_delta: float,
    contrast_delta: float,
) -> tuple[Image.Image, str]:
    output = ImageEnhance.Brightness(image).enhance(1.0 + brightness_delta)
    output = ImageEnhance.Contrast(output).enhance(1.0 + contrast_delta)
    description = (
        f"Color Jitter (brightness={brightness_delta * 100:+.0f}%, "
        f"contrast={contrast_delta * 100:+.0f}%)"
    )
    return output, description


def _center_crop(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, round(width * fraction))
    crop_height = max(1, round(height * fraction))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))
