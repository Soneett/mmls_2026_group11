import torch
import torch.nn as nn

from src.quantization.dynamic_int8 import (
    apply_dynamic_int8_quantization,
    convert_fake_quant_linear_to_linear,
    count_dynamic_quantized_linear_layers,
    prepare_model_for_dynamic_int8,
)
from src.quantization.fake_quant import (
    FakeQuantLinear,
    count_fake_quant_linear_layers,
    count_linear_layers,
    replace_linear_with_fake_quant,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


def test_convert_fake_quant_linear_to_linear_recovers_regular_linear_layers():
    torch.manual_seed(0)

    model = ToyModel()
    fake_quant_model = replace_linear_with_fake_quant(model, inplace=False)

    assert count_linear_layers(fake_quant_model) == 0
    assert count_fake_quant_linear_layers(fake_quant_model) == 2

    recovered_model = convert_fake_quant_linear_to_linear(fake_quant_model, inplace=False)

    assert count_linear_layers(recovered_model) == 2
    assert count_fake_quant_linear_layers(recovered_model) == 0

    x = torch.randn(3, 4)
    y = recovered_model(x)

    assert y.shape == (3, 2)


def test_convert_fake_quant_linear_to_linear_preserves_weights():
    torch.manual_seed(1)

    linear = nn.Linear(4, 3)
    fake_quant_linear = FakeQuantLinear.from_linear(linear)
    recovered_linear = convert_fake_quant_linear_to_linear(
        nn.Sequential(fake_quant_linear),
        inplace=False,
    )[0]

    assert isinstance(recovered_linear, nn.Linear)
    assert torch.allclose(recovered_linear.weight, linear.weight)
    assert recovered_linear.bias is not None
    assert linear.bias is not None
    assert torch.allclose(recovered_linear.bias, linear.bias)


def test_apply_dynamic_int8_quantization_replaces_linear_layers_on_cpu():
    torch.manual_seed(2)

    model = ToyModel()
    quantized_model = apply_dynamic_int8_quantization(model, inplace=False)

    assert count_linear_layers(model) == 2
    assert count_linear_layers(quantized_model) == 0
    assert count_dynamic_quantized_linear_layers(quantized_model) == 2

    x = torch.randn(5, 4)
    y = quantized_model(x)
    assert y.shape == (5, 2)
    assert next(quantized_model.parameters(), torch.empty(0)).device.type == "cpu"
    assert not quantized_model.training


def test_apply_dynamic_int8_quantization_can_work_inplace():
    torch.manual_seed(3)

    model = ToyModel()
    quantized_model = apply_dynamic_int8_quantization(model, inplace=True)

    assert quantized_model is model
    assert count_linear_layers(model) == 0
    assert count_dynamic_quantized_linear_layers(model) == 2


def test_prepare_model_for_dynamic_int8_exports_fake_quant_model():
    torch.manual_seed(4)

    model = ToyModel()
    fake_quant_model = replace_linear_with_fake_quant(model, inplace=False)

    assert count_fake_quant_linear_layers(fake_quant_model) == 2

    int8_model = prepare_model_for_dynamic_int8(fake_quant_model, inplace=False)

    assert count_fake_quant_linear_layers(int8_model) == 0
    assert count_linear_layers(int8_model) == 0
    assert count_dynamic_quantized_linear_layers(int8_model) == 2

    x = torch.randn(6, 4)
    y = int8_model(x)
    assert y.shape == (6, 2)


def test_prepare_model_for_dynamic_int8_does_not_modify_original_when_out_of_place():
    model = ToyModel()
    fake_quant_model = replace_linear_with_fake_quant(model, inplace=False)

    _ = prepare_model_for_dynamic_int8(fake_quant_model, inplace=False)

    assert count_fake_quant_linear_layers(fake_quant_model) == 2
    assert count_dynamic_quantized_linear_layers(fake_quant_model) == 0
