from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from glowclip.demo_degradations import (
    CENTER_CROP,
    COLOR_JITTER,
    GAUSSIAN_BLUR,
    GAUSSIAN_NOISE,
    JPEG_COMPRESSION,
    RESIZE,
    DemoDegradationConfig,
    apply_demo_degradations,
)


def gradient_image(width: int = 80, height: int = 60) -> Image.Image:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    array = np.empty((height, width, 3), dtype=np.uint8)
    array[..., 0] = x
    array[..., 1] = y
    array[..., 2] = 127
    return Image.fromarray(array, "RGB")


def test_multiple_degradations_use_documented_order_and_values() -> None:
    image = gradient_image()
    config = DemoDegradationConfig(
        jpeg_quality=30,
        blur_sigma=2.0,
        resize_scale=0.25,
        noise_sigma=0.10,
        seed=123,
    )
    result = apply_demo_degradations(
        image,
        [
            CENTER_CROP,
            COLOR_JITTER,
            GAUSSIAN_NOISE,
            RESIZE,
            GAUSSIAN_BLUR,
            JPEG_COMPRESSION,
        ],
        config,
    )

    assert result.image.mode == "RGB"
    assert result.image.size == (64, 48)
    assert [operation.split(" (")[0] for operation in result.operations] == [
        JPEG_COMPRESSION,
        GAUSSIAN_BLUR,
        RESIZE,
        GAUSSIAN_NOISE,
        COLOR_JITTER,
        CENTER_CROP,
    ]


def test_noise_is_seed_reproducible_with_user_selected_color_jitter() -> None:
    image = gradient_image(31, 23)
    config = DemoDegradationConfig(
        color_brightness=-0.12,
        color_contrast=0.08,
        seed=987,
    )
    first = apply_demo_degradations(image, [GAUSSIAN_NOISE, COLOR_JITTER], config)
    second = apply_demo_degradations(image, [GAUSSIAN_NOISE, COLOR_JITTER], config)

    assert first.operations == second.operations
    assert "brightness=-12%" in first.operations[1]
    assert "contrast=+8%" in first.operations[1]
    np.testing.assert_array_equal(np.asarray(first.image), np.asarray(second.image))


def test_no_selection_returns_rgb_copy_at_original_size() -> None:
    image = Image.new("RGBA", (17, 29), (40, 80, 120, 100))
    result = apply_demo_degradations(image, [])
    assert result.image.mode == "RGB"
    assert result.image.size == image.size
    assert result.operations == ("No degradation (original image)",)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jpeg_quality", 80),
        ("blur_sigma", 1.5),
        ("resize_scale", 0.75),
        ("noise_sigma", 0.03),
        ("color_brightness", 0.21),
        ("color_contrast", -0.21),
    ],
)
def test_unlisted_parameter_values_are_rejected(field: str, value: float) -> None:
    values = {field: value}
    with pytest.raises(ValueError):
        apply_demo_degradations(
            gradient_image(), [JPEG_COMPRESSION], DemoDegradationConfig(**values)
        )
