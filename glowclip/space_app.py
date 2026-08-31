from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from .demo_degradations import (
    BLUR_SIGMAS,
    CENTER_CROP,
    COLOR_JITTER,
    DEGRADATION_ORDER,
    GAUSSIAN_BLUR,
    GAUSSIAN_NOISE,
    JPEG_COMPRESSION,
    JPEG_QUALITIES,
    NOISE_SIGMAS,
    RESIZE,
    RESIZE_SCALES,
    DemoDegradationConfig,
    apply_demo_degradations,
)
from .inference import GLowCLIPPredictor
from .transforms import decode_rgb

DEFAULT_CHECKPOINT_FILENAME = "final_handoff_checkpoint_glow_dataset.pt"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_space_checkpoint(explicit_path: str | Path | None = None) -> Path:
    """Resolve a local checkpoint or download one from an optional Hub model repo."""
    configured = explicit_path or os.getenv("GLOWCLIP_CHECKPOINT")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(PROJECT_ROOT / "checkpoints" / DEFAULT_CHECKPOINT_FILENAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    model_repo_id = os.getenv("GLOWCLIP_MODEL_REPO_ID")
    if model_repo_id:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "huggingface_hub is required to download the configured checkpoint"
            ) from error
        filename = os.getenv(
            "GLOWCLIP_CHECKPOINT_FILENAME", DEFAULT_CHECKPOINT_FILENAME
        )
        revision = os.getenv("GLOWCLIP_MODEL_REVISION")
        token = os.getenv("HF_TOKEN")
        downloaded = hf_hub_download(
            repo_id=model_repo_id,
            filename=filename,
            revision=revision,
            token=token or None,
        )
        return Path(downloaded)

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "GLowCLIP checkpoint not found. Searched: "
        f"{searched}. Set GLOWCLIP_CHECKPOINT or add the checkpoint locally."
    )


class SpaceModelService:
    """Lazily load one predictor and reuse it across queued Space requests."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        precision: str | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device or os.getenv("GLOWCLIP_DEVICE", "auto")
        self.precision = precision or os.getenv("GLOWCLIP_PRECISION")
        self._predictor: GLowCLIPPredictor | None = None
        self._load_lock = Lock()

    def predictor(self) -> GLowCLIPPredictor:
        if self._predictor is None:
            with self._load_lock:
                if self._predictor is None:
                    path = resolve_space_checkpoint(self.checkpoint_path)
                    self._predictor = GLowCLIPPredictor(
                        path,
                        device=self.device,
                        precision=self.precision,
                    )
        return self._predictor


def prepare_upload_for_ui(
    image: Image.Image | None,
) -> tuple[Image.Image | None, str, None, None, None]:
    if image is None:
        return None, "No image", None, None, None
    original = decode_rgb(image)
    width, height = original.size
    return original, f"**{width} × {height}** · Original", None, None, None


def _compact_degradation_summary(
    selected: list[str], config: DemoDegradationConfig
) -> str:
    labels = {
        JPEG_COMPRESSION: f"JPEG q{config.jpeg_quality}",
        GAUSSIAN_BLUR: f"Blur σ{config.blur_sigma:g}",
        RESIZE: f"Resize {config.resize_scale:g}×",
        GAUSSIAN_NOISE: f"Noise σ{config.noise_sigma:.2f}",
        COLOR_JITTER: (
            f"Color B{config.color_brightness * 100:+.0f}% "
            f"C{config.color_contrast * 100:+.0f}%"
        ),
        CENTER_CROP: "Crop 80%",
    }
    selected_set = set(selected)
    return " → ".join(
        labels[name] for name in DEGRADATION_ORDER if name in selected_set
    )


def apply_degradations_for_ui(
    image: Image.Image | None,
    selected: list[str] | None,
    jpeg_quality: int,
    blur_sigma: float,
    resize_scale: float,
    noise_sigma: float,
    color_brightness: float,
    color_contrast: float,
) -> tuple[Image.Image, str, None, None, None]:
    if image is None:
        raise ValueError("Upload an image before applying degradations.")
    config = DemoDegradationConfig(
        jpeg_quality=int(jpeg_quality),
        blur_sigma=float(blur_sigma),
        resize_scale=float(resize_scale),
        noise_sigma=float(noise_sigma),
        color_brightness=float(color_brightness) / 100.0,
        color_contrast=float(color_contrast) / 100.0,
    )
    selected = selected or []
    result = apply_demo_degradations(image, selected, config)
    operations = _compact_degradation_summary(selected, config) or "Original"
    width, height = result.image.size
    status = f"**{width} × {height}** · {operations}"
    return result.image, status, None, None, None


def undo_degradations_for_ui(
    image: Image.Image | None,
) -> tuple[list[str], Image.Image | None, str, None, None, None]:
    if image is None:
        return [], None, "No image", None, None, None
    original = decode_rgb(image)
    width, height = original.size
    return (
        [],
        original,
        f"**{width} × {height}** · Original",
        None,
        None,
        None,
    )


def prediction_for_ui(
    image: Image.Image | None,
    service: SpaceModelService,
) -> tuple[str, dict[str, float], Image.Image]:
    if image is None:
        raise ValueError("Upload an image before running GLowCLIP.")
    predictor = service.predictor()
    prediction = predictor.predict(image)
    model_input = predictor.model_input_preview(image)
    score = f"## {prediction.summary}"
    class_scores = {
        "Real": prediction.real_probability,
        "AI-generated": prediction.fake_probability,
    }
    return score, class_scores, model_input


def _build_parameter_controls(gr: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    with gr.Accordion("Settings", open=True):
        with gr.Row():
            jpeg_quality = gr.Radio(
                choices=list(JPEG_QUALITIES),
                value=70,
                label="JPEG q",
            )
            blur_sigma = gr.Radio(
                choices=list(BLUR_SIGMAS),
                value=1.0,
                label="Blur σ",
            )
        with gr.Row():
            resize_scale = gr.Radio(
                choices=list(RESIZE_SCALES),
                value=0.5,
                label="Resize",
            )
            noise_sigma = gr.Radio(
                choices=list(NOISE_SIGMAS),
                value=0.05,
                label="Noise σ",
            )
        with gr.Row():
            color_brightness = gr.Slider(
                minimum=-20,
                maximum=20,
                value=20,
                step=1,
                label="Brightness %",
            )
            color_contrast = gr.Slider(
                minimum=-20,
                maximum=20,
                value=20,
                step=1,
                label="Contrast %",
            )
    return (
        jpeg_quality,
        blur_sigma,
        resize_scale,
        noise_sigma,
        color_brightness,
        color_contrast,
    )


def build_demo(
    service: SpaceModelService | None = None,
    prediction_callback: Callable[[Image.Image | None], Any] | None = None,
):
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError(
            "Gradio is required for the web demo; install with pip install -e '.[space]'"
        ) from error

    service = service or SpaceModelService()
    css = """
    .gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
    .glow-hero {
      padding: 1.2rem 1.5rem;
      border-radius: 18px;
      background: linear-gradient(125deg, #111827, #4c1d95 58%, #7e22ce);
      color: white;
      margin-bottom: 1rem;
      box-shadow: 0 14px 38px rgba(76, 29, 149, .22);
    }
    .glow-hero h1 {
      display: inline-block;
      margin: 0;
      padding: .12rem .62rem .2rem;
      border: 1px solid rgba(255, 255, 255, .42);
      border-radius: 12px;
      background: rgba(15, 23, 42, .52);
      color: #fff !important;
      -webkit-text-fill-color: #fff !important;
      font-size: 2rem;
      letter-spacing: -.03em;
      text-shadow: 0 2px 10px rgba(0, 0, 0, .72);
    }
    .glow-hero p {
      margin: .35rem 0 0;
      color: #f5f3ff !important;
      opacity: 1;
      text-shadow: 0 1px 5px rgba(0, 0, 0, .65);
    }
    .glow-panel { border-radius: 16px !important; }
    .glow-status { min-height: 1.6rem; opacity: .82; }
    .glow-score h2 { margin: .35rem 0; }
    """
    with gr.Blocks(title="GLowCLIP", fill_width=True) as demo:
        gr.HTML(
            f"<style>{css}</style>"
            "<div class='glow-hero'><h1>GLowCLIP</h1>"
            "<p>Real vs AI · test robustness under degradation</p></div>"
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, variant="panel", elem_classes="glow-panel"):
                source_image = gr.Image(
                    type="pil",
                    image_mode="RGB",
                    format="png",
                    sources=["upload"],
                    label="1 · Upload",
                    height=330,
                )
                degradations = gr.CheckboxGroup(
                    choices=list(DEGRADATION_ORDER),
                    label="2 · Degradations",
                    info="JPEG → blur → resize → noise → color → crop",
                )
                parameter_controls = _build_parameter_controls(gr)
                with gr.Row():
                    apply_button = gr.Button("Apply", variant="primary")
                    undo_button = gr.Button("Undo all", variant="secondary")

            with gr.Column(scale=1, variant="panel", elem_classes="glow-panel"):
                processed_image = gr.Image(
                    type="pil",
                    image_mode="RGB",
                    format="png",
                    label="3 · Preview",
                    interactive=False,
                    height=330,
                )
                processing_status = gr.Markdown("No image", elem_classes="glow-status")
                analyze_button = gr.Button("Analyze", variant="primary")
                score = gr.Markdown("## —", elem_classes="glow-score")
                class_scores = gr.Label(
                    label="Scores",
                    num_top_classes=2,
                )
                with gr.Accordion("Model input · 224 × 224", open=False):
                    model_input = gr.Image(
                        type="pil",
                        image_mode="RGB",
                        format="png",
                        show_label=False,
                        interactive=False,
                        height=240,
                    )

        analysis_outputs = [score, class_scores, model_input]
        source_image.change(
            fn=prepare_upload_for_ui,
            inputs=source_image,
            outputs=[processed_image, processing_status, *analysis_outputs],
            queue=False,
        )
        apply_button.click(
            fn=apply_degradations_for_ui,
            inputs=[source_image, degradations, *parameter_controls],
            outputs=[processed_image, processing_status, *analysis_outputs],
        )
        undo_button.click(
            fn=undo_degradations_for_ui,
            inputs=source_image,
            outputs=[
                degradations,
                processed_image,
                processing_status,
                *analysis_outputs,
            ],
            queue=False,
        )

        if prediction_callback is None:

            def run_prediction(image: Image.Image | None):
                try:
                    return prediction_for_ui(image, service)
                except Exception as error:
                    raise gr.Error(str(error)) from error

        else:
            run_prediction = prediction_callback

        analyze_button.click(
            fn=run_prediction,
            inputs=processed_image,
            outputs=analysis_outputs,
            concurrency_limit=1,
            concurrency_id="glowclip-model",
        )
        gr.Markdown("Model estimate—not proof.")
    return demo.queue(max_size=16, default_concurrency_limit=2)


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
