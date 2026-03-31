from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("torchvision")
pytest.importorskip("transformers")
pytest.importorskip("rich")
pytest.importorskip("pandas")

from q_lab.onnx_utils import export_to_onnx
from q_lab.types import BenchmarkConfig, InvocationMode, LoadedModel, ModelFamily
from tests.helpers import ThirdPartyTextNet, ThirdPartyVisionNet, TinyTextNet, TinyVisionNet


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    torch.manual_seed(7)


@pytest.fixture()
def benchmark_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        warmup_iterations=1,
        benchmark_iterations=2,
        calibration_iterations=1,
        batch_size=1,
        device="cpu",
    )


@pytest.fixture()
def tiny_vision_model() -> TinyVisionNet:
    return TinyVisionNet().eval()


@pytest.fixture()
def tiny_text_model() -> TinyTextNet:
    return TinyTextNet().eval()


@pytest.fixture()
def vision_loaded_model(tiny_vision_model: TinyVisionNet) -> LoadedModel:
    example_input = torch.randn(1, 3, 8, 8)
    return LoadedModel(
        identifier="toyvision",
        display_name="Toy Vision",
        source="unit-test",
        model=tiny_vision_model,
        family=ModelFamily.VISION,
        invocation_mode=InvocationMode.POSITIONAL,
        input_names=("input",),
        example_inputs=(example_input,),
    )


@pytest.fixture()
def text_loaded_model(tiny_text_model: TinyTextNet) -> LoadedModel:
    return LoadedModel(
        identifier="toytext",
        display_name="Toy Text",
        source="unit-test",
        model=tiny_text_model,
        family=ModelFamily.TEXT,
        invocation_mode=InvocationMode.KEYWORD,
        input_names=("input_ids", "attention_mask", "token_type_ids"),
        example_inputs=(
            torch.randint(0, 32, (1, 6), dtype=torch.long),
            torch.ones((1, 6), dtype=torch.long),
            torch.zeros((1, 6), dtype=torch.long),
        ),
    )


@pytest.fixture()
def scripted_vision_model_path(tmp_path: Path, tiny_vision_model: TinyVisionNet) -> Path:
    example_input = torch.randn(1, 3, 8, 8)
    scripted_model = torch.jit.trace(tiny_vision_model, example_input)
    destination = tmp_path / "tiny_vision.pt"
    scripted_model.save(destination.as_posix())
    return destination


@pytest.fixture()
def scripted_third_party_vision_model_path(
    tmp_path: Path,
) -> Path:
    model = ThirdPartyVisionNet().eval()
    example_input = torch.randn(1, 3, 8, 8)
    scripted_model = torch.jit.trace(model, example_input)
    destination = tmp_path / "external_backbone_aux.pt"
    scripted_model.save(destination.as_posix())
    return destination


@pytest.fixture()
def scripted_third_party_text_model_path(
    tmp_path: Path,
) -> Path:
    model = ThirdPartyTextNet().eval()
    scripted_model = torch.jit.script(model)
    destination = tmp_path / "external_bert_encoder.pt"
    scripted_model.save(destination.as_posix())
    return destination


@pytest.fixture()
def onnx_vision_model_path(tmp_path: Path, tiny_vision_model: TinyVisionNet) -> Path:
    example_input = torch.randn(1, 3, 8, 8)
    loaded_model = LoadedModel(
        identifier="toyvision",
        display_name="Toy Vision",
        source="unit-test",
        model=tiny_vision_model,
        family=ModelFamily.VISION,
        invocation_mode=InvocationMode.POSITIONAL,
        input_names=("input",),
        example_inputs=(example_input,),
    )
    return export_to_onnx(
        loaded_model=loaded_model,
        model=tiny_vision_model,
        destination=tmp_path / "toy_vision.onnx",
        opset_version=17,
    )


@pytest.fixture()
def onnx_text_model_path(tmp_path: Path) -> Path:
    model = ThirdPartyTextNet().eval()
    input_ids = torch.randint(0, 256, (1, 6), dtype=torch.long)
    attention_mask = torch.ones((1, 6), dtype=torch.long)
    token_type_ids = torch.zeros((1, 6), dtype=torch.long)
    loaded_model = LoadedModel(
        identifier="toytext",
        display_name="Toy Text",
        source="unit-test",
        model=model,
        family=ModelFamily.TEXT,
        invocation_mode=InvocationMode.KEYWORD,
        input_names=("input_ids", "attention_mask", "token_type_ids"),
        example_inputs=(input_ids, attention_mask, token_type_ids),
    )
    return export_to_onnx(
        loaded_model=loaded_model,
        model=model,
        destination=tmp_path / "toy_text.onnx",
        opset_version=17,
    )
