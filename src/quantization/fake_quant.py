from __future__ import annotations

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


ModuleFilter = Callable[[str, nn.Linear], bool]


def fake_quantize_int8_ste(
    x: torch.Tensor,
    num_bits: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply symmetric int8-like fake quantization with STE gradients.

    Forward pass:
        float tensor -> int8 grid -> dequantized float tensor.

    Backward pass:
        straight-through estimator, so gradients flow as if quantization was
        the identity function.
    """
    if not torch.is_floating_point(x):
        raise TypeError(f"fake_quantize_int8_ste expects a floating tensor, got {x.dtype}")

    if x.numel() == 0:
        return x

    if num_bits < 2:
        raise ValueError("num_bits must be at least 2")

    qmax = float(2 ** (num_bits - 1) - 1)
    qmin = -qmax

    max_abs = x.detach().abs().amax()
    scale = torch.clamp(max_abs / qmax, min=eps)

    q = torch.round(x / scale)
    q = torch.clamp(q, qmin, qmax)
    x_dequant = q * scale

    return x + (x_dequant - x).detach()


class FakeQuantLinear(nn.Module):
    """Linear layer with fake int8 quantization in the forward pass.

    The layer keeps trainable float32/fp16 parameters. During forward it uses
    fake-quantized activations and/or weights. Gradients are propagated through
    the fake quantizer with the straight-through estimator.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantize_weight: bool = True,
        quantize_activation: bool = True,
        num_bits: int = 8,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_weight = quantize_weight
        self.quantize_activation = quantize_activation
        self.num_bits = num_bits

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        quantize_weight: bool = True,
        quantize_activation: bool = True,
        num_bits: int = 8,
    ) -> "FakeQuantLinear":
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            quantize_weight=quantize_weight,
            quantize_activation=quantize_activation,
            num_bits=num_bits,
        )
        layer.weight = nn.Parameter(linear.weight.detach().clone())
        layer.weight.requires_grad = linear.weight.requires_grad

        if linear.bias is not None:
            layer.bias = nn.Parameter(linear.bias.detach().clone())
            layer.bias.requires_grad = linear.bias.requires_grad

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_activation:
            x = fake_quantize_int8_ste(x, num_bits=self.num_bits)

        weight = self.weight
        if self.quantize_weight:
            weight = fake_quantize_int8_ste(weight, num_bits=self.num_bits)

        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quantize_weight={self.quantize_weight}, "
            f"quantize_activation={self.quantize_activation}, num_bits={self.num_bits}"
        )


def replace_linear_with_fake_quant(
    module: nn.Module,
    quantize_weight: bool = True,
    quantize_activation: bool = True,
    num_bits: int = 8,
    inplace: bool = True,
    module_filter: ModuleFilter | None = None,
) -> nn.Module:
    """Recursively replace nn.Linear layers with FakeQuantLinear layers."""
    if not inplace:
        module = copy.deepcopy(module)

    def _replace(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, FakeQuantLinear):
                continue

            if isinstance(child, nn.Linear):
                should_replace = module_filter is None or module_filter(full_name, child)
                if should_replace:
                    setattr(
                        parent,
                        name,
                        FakeQuantLinear.from_linear(
                            child,
                            quantize_weight=quantize_weight,
                            quantize_activation=quantize_activation,
                            num_bits=num_bits,
                        ),
                    )
            else:
                _replace(child, full_name)

    _replace(module)
    return module


def count_linear_layers(module: nn.Module) -> int:
    return sum(1 for child in module.modules() if isinstance(child, nn.Linear))


def count_fake_quant_linear_layers(module: nn.Module) -> int:
    return sum(1 for child in module.modules() if isinstance(child, FakeQuantLinear))
