from .fake_quant import (
    FakeQuantLinear,
    count_fake_quant_linear_layers,
    count_linear_layers,
    fake_quantize_int8_ste,
    replace_linear_with_fake_quant,
)

__all__ = ["FakeQuantLinear", "count_fake_quant_linear_layers", "count_linear_layers", "fake_quantize_int8_ste", "replace_linear_with_fake_quant"]
