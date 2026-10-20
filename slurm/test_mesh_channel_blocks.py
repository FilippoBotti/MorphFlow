"""CPU numerical checks for the opt-in sparse convolution channel splitting."""

import enum
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:
    torch = None


SPEC = importlib.util.spec_from_file_location(
    "mesh_channel_blocks", Path(__file__).with_name("mesh_channel_blocks.py")
)
blocks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(blocks)


class ConvAlgo(enum.Enum):
    Native = 0
    MaskImplicitGemm = 1


class Activation(enum.Enum):
    None_ = 0


class SparseInput:
    def __init__(self, features, indices=None, cache=None):
        self.features = features
        self.indices = indices
        self.indice_dict = {} if cache is None else cache

    def replace_feature(self, features):
        return SparseInput(features, self.indices, self.indice_dict)


class LinearSparseConv:
    """Emulate KRSC convolution summation, including bias and sparse metadata."""

    def __init__(self, inputs=5, outputs=8):
        self.in_channels = inputs
        self.out_channels = outputs
        self.kernel_size = [3, 3, 3]
        self.stride = [1, 1, 1]
        self.padding = [1, 1, 1]
        self.dilation = [1, 1, 1]
        self.groups = 1
        self.subm = True
        self.inverse = False
        self.transposed = False
        self.training = False
        self.conv1x1 = False
        self.algo = ConvAlgo.Native
        self.act_type = Activation.None_
        self.weight = torch.linspace(-.2, .3, outputs * 27 * inputs).reshape(outputs, 3, 3, 3, inputs)
        self.bias = torch.linspace(.1, .8, outputs)
        self.calls = []
        self.fail_on = None

    def _conv_forward(self, training, input, weight, bias):
        self.calls.append((input.features.shape, weight.shape, bias))
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("synthetic convolution failure")
        kernel = weight.sum(dim=(1, 2, 3))
        result = input.features @ kernel.T
        if bias is not None:
            result = result + bias
        input.indice_dict["reused_native_indices"] = "cached"
        return input.replace_feature(result)

    def forward(self, sparse_input):
        return self._conv_forward(False, sparse_input, self.weight, self.bias)


@unittest.skipIf(torch is None, "Torch is needed for CPU numerical tests")
class ChannelBlockTests(unittest.TestCase):
    def setUp(self):
        constants = types.ModuleType("spconv.constants")
        constants.ALL_WEIGHT_IS_KRSC = True
        spconv = types.ModuleType("spconv")
        spconv.__path__ = []
        spconv.constants = constants
        core = types.ModuleType("spconv.core")
        core.ConvAlgo = ConvAlgo
        self.modules = patch.dict(sys.modules, {
            "spconv": spconv, "spconv.constants": constants, "spconv.core": core,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.layer = LinearSparseConv()
        self.input = SparseInput(
            torch.arange(35, dtype=torch.float32).reshape(7, 5) / 35,
            torch.arange(28, dtype=torch.int32).reshape(7, 4),
            {"original_cache": "retained"},
        )

    def reference(self):
        return self.input.features @ self.layer.weight.sum(dim=(1, 2, 3)).T + self.layer.bias

    def test_small_convolution_uses_original_forward_once(self):
        expected = self.reference()
        original = Mock(wraps=self.layer.forward)
        result = blocks.channel_block_subm(
            self.layer, self.input, max_bytes=1024, original_forward=original
        )
        original.assert_called_once_with(self.input)
        self.assertEqual(len(self.layer.calls), 1)
        torch.testing.assert_close(result.features, expected)

    def test_both_channel_axes_split_preserve_result_bias_coords_and_cache(self):
        expected = self.reference()
        before = dict(vars(self.layer))
        event = Mock()
        result = blocks.channel_block_subm(self.layer, self.input, max_bytes=64, on_split=event)
        self.assertGreater(len(self.layer.calls), 1)
        self.assertTrue(all(shape[1] * shape[0] * 4 <= 64 for shape, _, _ in self.layer.calls))
        self.assertTrue(all(weight[0] * shape[0] * 4 <= 64 for shape, weight, _ in self.layer.calls))
        self.assertTrue(all(bias is None for _, _, bias in self.layer.calls))
        torch.testing.assert_close(result.features, expected, rtol=2e-6, atol=2e-6)
        self.assertIs(result.indices, self.input.indices)
        self.assertEqual(result.indice_dict["original_cache"], "retained")
        self.assertEqual(result.indice_dict["reused_native_indices"], "cached")
        event.assert_called_once()
        for name in ("weight", "bias", "algo", "in_channels", "out_channels"):
            self.assertIs(getattr(self.layer, name), before[name])
        torch.testing.assert_close(self.input.features, torch.arange(35).reshape(7, 5) / 35)

    def test_partial_failure_restores_layer_and_input(self):
        original_features = self.input.features
        original_weight = self.layer.weight
        original_bias = self.layer.bias
        self.layer.fail_on = 2
        with self.assertRaisesRegex(RuntimeError, "synthetic convolution failure"):
            blocks.channel_block_subm(self.layer, self.input, max_bytes=64)
        self.assertIs(self.layer.weight, original_weight)
        self.assertIs(self.layer.bias, original_bias)
        self.assertEqual(self.layer.in_channels, 5)
        self.assertEqual(self.layer.out_channels, 8)
        self.assertEqual(self.layer.algo, ConvAlgo.Native)
        self.assertIs(self.input.features, original_features)


if __name__ == "__main__":
    unittest.main()
