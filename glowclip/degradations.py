from __future__ import annotations

import io
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class DegradationResult:
    image: Image.Image
    operations: tuple[str, ...]


class CompoundDegradation:
    """Sample a 0--5 operation degradation chain using label-agnostic randomness."""

    chain_lengths = (0, 1, 2, 3, 4, 5)
    chain_probabilities = (0.15, 0.20, 0.25, 0.20, 0.12, 0.08)
    categories = ("geometry", "blur", "noise", "color", "compression")

    def __init__(self) -> None:
        self._operations: dict[
            str, Callable[[Image.Image, str], tuple[Image.Image, str]]
        ] = {
            "geometry": self._geometry,
            "blur": self._blur,
            "noise": self._noise,
            "color": self._color,
            "compression": self._compression,
        }

    @staticmethod
    def _severity() -> str:
        return random.choices(
            ("mild", "medium", "heavy"), weights=(0.4, 0.4, 0.2), k=1
        )[0]

    def __call__(self, image: Image.Image) -> DegradationResult:
        image = image.convert("RGB")
        chain_length = random.choices(
            self.chain_lengths, weights=self.chain_probabilities, k=1
        )[0]
        if chain_length == 0:
            return DegradationResult(image=image.copy(), operations=("identity",))

        chosen = random.sample(self.categories, k=chain_length)
        operations: list[str] = []
        output = image.copy()
        for category in chosen:
            output, description = self._operations[category](output, self._severity())
            operations.append(description)
        return DegradationResult(
            image=output.convert("RGB"), operations=tuple(operations)
        )

    @staticmethod
    def _geometry(image: Image.Image, severity: str) -> tuple[Image.Image, str]:
        choice = random.choice(("crop", "rotation", "downscale", "pixelate"))
        width, height = image.size
        strength = {"mild": 0, "medium": 1, "heavy": 2}[severity]

        if choice == "crop":
            min_scale = (0.88, 0.72, 0.52)[strength]
            scale = random.uniform(min_scale, min(1.0, min_scale + 0.12))
            aspect = random.uniform(
                (0.9, 0.78, 0.65)[strength], (1.1, 1.28, 1.5)[strength]
            )
            crop_w = min(width, max(2, int(width * (scale * aspect) ** 0.5)))
            crop_h = min(height, max(2, int(height * (scale / aspect) ** 0.5)))
            left = random.randint(0, max(0, width - crop_w))
            top = random.randint(0, max(0, height - crop_h))
            output = image.crop((left, top, left + crop_w, top + crop_h)).resize(
                (width, height), Image.Resampling.BICUBIC
            )
            return output, f"random_crop_{severity}"

        if choice == "rotation":
            max_angle = (3.0, 8.0, 15.0)[strength]
            angle = random.uniform(-max_angle, max_angle)
            output = image.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(0, 0, 0),
            )
            return output, f"rotation_{angle:.1f}deg"

        if choice == "downscale":
            factor = random.uniform(
                *(((0.72, 0.9), (0.45, 0.72), (0.20, 0.45))[strength])
            )
            small = (max(8, int(width * factor)), max(8, int(height * factor)))
            output = image.resize(small, Image.Resampling.LANCZOS).resize(
                (width, height), Image.Resampling.BICUBIC
            )
            return output, f"downscale_{factor:.2f}"

        factor = random.uniform(*(((0.65, 0.85), (0.35, 0.65), (0.12, 0.35))[strength]))
        small = (max(4, int(width * factor)), max(4, int(height * factor)))
        output = image.resize(small, Image.Resampling.BILINEAR).resize(
            (width, height), Image.Resampling.NEAREST
        )
        return output, f"pixelate_{factor:.2f}"

    @staticmethod
    def _blur(image: Image.Image, severity: str) -> tuple[Image.Image, str]:
        strength = {"mild": 0, "medium": 1, "heavy": 2}[severity]
        choice = random.choice(("gaussian", "box", "median", "motion"))
        if choice == "gaussian":
            radius = random.uniform(*(((0.2, 0.7), (0.7, 1.6), (1.6, 3.0))[strength]))
            return image.filter(
                ImageFilter.GaussianBlur(radius)
            ), f"gaussian_blur_{radius:.2f}"
        if choice == "box":
            radius = random.uniform(*(((0.2, 0.6), (0.6, 1.3), (1.3, 2.5))[strength]))
            return image.filter(ImageFilter.BoxBlur(radius)), f"box_blur_{radius:.2f}"
        if choice == "median":
            size = (3, 3, 5)[strength]
            return image.filter(
                ImageFilter.MedianFilter(size=size)
            ), f"median_blur_{size}"

        size = (3, 5, 5)[strength]
        kernel = [0.0] * (size * size)
        if random.random() < 0.5:
            row = size // 2
            for x in range(size):
                kernel[row * size + x] = 1.0
        else:
            for index in range(size):
                kernel[index * size + index] = 1.0
        output = image.filter(
            ImageFilter.Kernel((size, size), kernel, scale=float(size))
        )
        return output, f"motion_blur_{size}"

    @staticmethod
    def _noise(image: Image.Image, severity: str) -> tuple[Image.Image, str]:
        array = np.asarray(image, dtype=np.float32) / 255.0
        rng = np.random.default_rng(random.getrandbits(64))
        strength = {"mild": 0, "medium": 1, "heavy": 2}[severity]
        choice = random.choice(("gaussian", "poisson", "speckle", "impulse"))

        if choice == "gaussian":
            sigma = (
                random.uniform(*(((1.0, 4.0), (4.0, 9.0), (9.0, 18.0))[strength]))
                / 255.0
            )
            array += rng.normal(0.0, sigma, size=array.shape)
            description = f"gaussian_noise_{sigma * 255.0:.1f}"
        elif choice == "poisson":
            levels = (96.0, 48.0, 20.0)[strength]
            array = rng.poisson(np.clip(array, 0.0, 1.0) * levels) / levels
            description = f"poisson_noise_{int(levels)}"
        elif choice == "speckle":
            sigma = (0.015, 0.04, 0.08)[strength]
            array += array * rng.normal(0.0, sigma, size=array.shape)
            description = f"speckle_noise_{sigma:.3f}"
        else:
            fraction = (0.002, 0.008, 0.02)[strength]
            mask = rng.random(array.shape[:2])
            array[mask < fraction / 2.0] = 0.0
            array[(mask >= fraction / 2.0) & (mask < fraction)] = 1.0
            description = f"impulse_noise_{fraction:.3f}"

        output = Image.fromarray(
            (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB"
        )
        return output, description

    @staticmethod
    def _color(image: Image.Image, severity: str) -> tuple[Image.Image, str]:
        strength = {"mild": 0, "medium": 1, "heavy": 2}[severity]
        choice = random.choice(
            ("brightness", "contrast", "saturation", "hue", "gamma", "posterize")
        )
        ranges = ((0.88, 1.12), (0.72, 1.28), (0.50, 1.50))
        if choice in {"brightness", "contrast", "saturation"}:
            factor = random.uniform(*ranges[strength])
            enhancer = {
                "brightness": ImageEnhance.Brightness,
                "contrast": ImageEnhance.Contrast,
                "saturation": ImageEnhance.Color,
            }[choice]
            return enhancer(image).enhance(factor), f"{choice}_{factor:.2f}"
        if choice == "hue":
            max_shift = (5, 12, 24)[strength]
            shift = random.randint(-max_shift, max_shift)
            hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
            return Image.fromarray(hsv, "HSV").convert("RGB"), f"hue_shift_{shift}"
        if choice == "gamma":
            gamma = random.uniform(*ranges[strength])
            lookup = [int(255.0 * ((value / 255.0) ** gamma)) for value in range(256)]
            return image.point(lookup * 3), f"gamma_{gamma:.2f}"
        bits = (7, 6, 4)[strength]
        return ImageOps.posterize(image, bits), f"posterize_{bits}bit"

    @staticmethod
    def _compression(image: Image.Image, severity: str) -> tuple[Image.Image, str]:
        strength = {"mild": 0, "medium": 1, "heavy": 2}[severity]
        quality_range = ((82, 95), (55, 81), (25, 54))[strength]
        codec = random.choice(("JPEG", "JPEG", "WEBP"))
        passes = 2 if random.random() < 0.25 else 1
        output = image
        qualities: list[int] = []
        for _ in range(passes):
            quality = random.randint(*quality_range)
            qualities.append(quality)
            buffer = io.BytesIO()
            try:
                output.save(buffer, format=codec, quality=quality)
            except OSError:
                codec = "JPEG"
                output.save(buffer, format=codec, quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                output = decoded.convert("RGB").copy()
        return output, f"{codec.lower()}_q{'-'.join(map(str, qualities))}"
