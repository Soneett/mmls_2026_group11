import torch
import torch.nn as nn

from models.compressor import Compressor
from models.encoder import SimpleGCNEncoder
from src.quantization.fake_quant import (
    FakeQuantLinear,
    count_fake_quant_linear_layers,
    count_linear_layers,
    fake_quantize_int8_ste,
    replace_linear_with_fake_quant,
)


def test_fake_quant_forward_uses_int8_grid_and_backward_is_ste():
    x = torch.tensor([-2.0, -0.7, 0.0, 0.9, 2.0], requires_grad=True)

    y = fake_quantize_int8_ste(x)

    scale = x.detach().abs().amax() / 127.0
    q = y.detach() / scale

    assert torch.allclose(q, torch.round(q), atol=1e-5)

    y.sum().backward()

    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_fake_quant_linear_preserves_shape_and_allows_backward():
    torch.manual_seed(0)

    linear = nn.Linear(4, 3)
    fq_linear = FakeQuantLinear.from_linear(linear)

    x = torch.randn(5, 4, requires_grad=True)
    y = fq_linear(x)

    assert y.shape == (5, 3)

    loss = y.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert fq_linear.weight.grad is not None
    assert fq_linear.bias is not None
    assert fq_linear.bias.grad is not None


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Sequential(
                nn.Linear(8, 2),
            ),
        )
        self.norm = nn.LayerNorm(2)

    def forward(self, x):
        return self.norm(self.net(x))


def test_replace_linear_with_fake_quant_can_work_out_of_place():
    model = ToyModel()

    quantized_model = replace_linear_with_fake_quant(model, inplace=False)

    assert count_linear_layers(model) == 2
    assert count_fake_quant_linear_layers(model) == 0

    assert count_linear_layers(quantized_model) == 0
    assert count_fake_quant_linear_layers(quantized_model) == 2

    x = torch.randn(3, 4, requires_grad=True)
    y = quantized_model(x)
    assert y.shape == (3, 2)

    y.sum().backward()
    assert x.grad is not None


def test_replace_linear_with_fake_quant_supports_module_filter():
    model = ToyModel()

    quantized_model = replace_linear_with_fake_quant(
        model,
        inplace=False,
        module_filter=lambda name, layer: name == "net.0",
    )

    assert count_linear_layers(quantized_model) == 1
    assert count_fake_quant_linear_layers(quantized_model) == 1


def test_project_encoder_and_compressor_linear_layers_can_be_replaced():
    encoder = SimpleGCNEncoder(
        in_dim=8,
        hid_dim=8,
        out_dim=8,
        n_layers=2,
        dropout=0.1,
    )
    compressor = Compressor(d_in=8, d_out=4)
    model = nn.ModuleDict({"encoder": encoder, "compressor": compressor})

    assert count_linear_layers(model) == 4

    quantized_model = replace_linear_with_fake_quant(model, inplace=False)

    assert count_linear_layers(quantized_model) == 0
    assert count_fake_quant_linear_layers(quantized_model) == 4
