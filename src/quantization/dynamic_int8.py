from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.quantization.fake_quant import FakeQuantLinear

try:
    from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear
except ImportError:  # pragma: no cover - compatibility fallback for older torch builds
    DynamicQuantizedLinear = ()


def get_available_quantized_engine() -> str | None:
    """Return a usable PyTorch quantized backend, if one is available.

    Some local builds, especially on macOS, start with
    torch.backends.quantized.engine == "none". In that state dynamic int8
    conversion fails with NoQEngine. This helper selects a real backend from
    the engines compiled into the current PyTorch build.
    """
    supported = list(torch.backends.quantized.supported_engines)
    for engine in ("x86", "fbgemm", "qnnpack", "onednn"):
        if engine in supported:
            return engine
    return None


def ensure_quantized_engine() -> str:
    """Set and return a usable quantized backend for dynamic int8 Linear."""
    current = torch.backends.quantized.engine
    if current != "none":
        return current

    engine = get_available_quantized_engine()
    if engine is None:
        raise RuntimeError(
            "Dynamic int8 quantization is unavailable: this PyTorch build has "
            f"no usable quantized backend. Supported engines: "
            f"{torch.backends.quantized.supported_engines}."
        )

    torch.backends.quantized.engine = engine
    return engine


def fake_quant_linear_to_linear(layer: FakeQuantLinear) -> nn.Linear:
    """Convert a FakeQuantLinear layer back to a regular nn.Linear layer.

    FakeQuantLinear keeps trainable floating-point weights. This helper copies
    those weights into nn.Linear so that PyTorch dynamic quantization can later
    recognize and quantize the layer.
    """
    linear = nn.Linear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,
    )
    linear.weight = nn.Parameter(layer.weight.detach().clone())
    linear.weight.requires_grad = layer.weight.requires_grad

    if layer.bias is not None:
        linear.bias = nn.Parameter(layer.bias.detach().clone())
        linear.bias.requires_grad = layer.bias.requires_grad

    return linear


def convert_fake_quant_linear_to_linear(
    module: nn.Module,
    inplace: bool = True,
) -> nn.Module:
    """Recursively replace FakeQuantLinear layers with nn.Linear layers."""
    if not inplace:
        module = copy.deepcopy(module)

    def _replace(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, FakeQuantLinear):
                setattr(parent, name, fake_quant_linear_to_linear(child))
            else:
                _replace(child)

    _replace(module)
    return module


def apply_dynamic_int8_quantization(
    model: nn.Module,
    inplace: bool = False,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """Apply PyTorch dynamic int8 quantization to nn.Linear layers.

    Dynamic quantization is intended for CPU inference. It quantizes Linear
    weights to int8 and dynamically quantizes activations during inference.
    The returned model is moved to CPU and switched to eval mode.

    If the model contains FakeQuantLinear layers, call
    convert_fake_quant_linear_to_linear first.
    """
    if dtype != torch.qint8:
        raise ValueError(f"Only torch.qint8 is supported, got {dtype}")

    if not inplace:
        model = copy.deepcopy(model)

    model.to("cpu")
    model.eval()
    ensure_quantized_engine()

    return torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=dtype,
        inplace=True,
    )


def prepare_model_for_dynamic_int8(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """Convert FakeQuantLinear layers to nn.Linear and apply dynamic int8.

    This is the expected export path after fake-quant training:

        FakeQuantLinear training model
            -> regular nn.Linear model with trained float weights
            -> PyTorch dynamic int8 model for CPU inference
    """
    if not inplace:
        model = copy.deepcopy(model)

    model = convert_fake_quant_linear_to_linear(model, inplace=True)
    model = apply_dynamic_int8_quantization(model, inplace=True)
    return model


def count_dynamic_quantized_linear_layers(module: nn.Module) -> int:
    """Count dynamically quantized Linear layers in a model."""
    if DynamicQuantizedLinear == ():
        return 0

    return sum(1 for child in module.modules() if isinstance(child, DynamicQuantizedLinear))
