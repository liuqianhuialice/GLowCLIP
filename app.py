"""GLowCLIP interactive demo entrypoint."""

# ruff: noqa: I001 -- ZeroGPU's import must precede the package's torch imports.

# The optional dynamic-GPU shim must be imported before anything that imports
# torch. Local and standard-GPU deployments use the fallback path.
try:
    import spaces
except ImportError:  # pragma: no cover - the normal local path
    spaces = None

from glowclip.space_app import SpaceModelService, build_demo, prediction_for_ui


if spaces is not None:
    model_service = SpaceModelService(device="cuda")
    # ZeroGPU's CUDA emulation supports module-scope placement. This avoids
    # reloading CLIP and the adapter on every dynamically allocated GPU call.
    model_service.predictor()

    @spaces.GPU(duration=60)
    def run_prediction(image):
        return prediction_for_ui(image, model_service)

else:
    model_service = SpaceModelService()

    def run_prediction(image):
        return prediction_for_ui(image, model_service)


demo = build_demo(service=model_service, prediction_callback=run_prediction)


if __name__ == "__main__":
    demo.launch()
