"""Bound the per-kernel channel buffers of inference-only sparse convolutions.

Some cumm kernels use signed 32-bit *byte* offsets even when the GPU has enough
memory. A channel decomposition keeps each kernel's input and output below that
limit without dropping voxels, changing the weights, or reducing precision.
Only the order in which FP32 partial sums are added changes.

This module deliberately has no Torch/spconv import at import time: preparation
of a frozen-run repair must remain usable from a CPU login node.
"""

import inspect
import operator


INT32_MAX = (1 << 31) - 1
DEFAULT_MAX_BYTES = INT32_MAX - (16 << 20)


def _private_forward(module, sparse_input, weight, bias):
    """Use the explicit-weight inference API; never replace model parameters."""
    return module._conv_forward(False, sparse_input, weight, bias)


def _check_split_compatibility(module, sparse_input, torch):
    import spconv.constants as constants
    from spconv.core import ConvAlgo

    if not getattr(constants, "ALL_WEIGHT_IS_KRSC", False):
        raise RuntimeError("Channel blocks require spconv ALL_WEIGHT_IS_KRSC=True")
    if not getattr(module, "subm", False):
        raise RuntimeError("Channel blocks support only submanifold convolutions")
    if any(int(value) != 1 for value in module.stride):
        raise RuntimeError("Channel blocks require stride=1")
    if getattr(module, "groups", None) != 1:
        raise RuntimeError("Channel blocks require groups=1")
    if getattr(module, "inverse", False) or getattr(module, "transposed", False):
        raise RuntimeError("Channel blocks do not support inverse/transposed convolutions")
    activation = getattr(module, "act_type", None)
    if activation is not None and getattr(activation, "value", activation) != 0:
        raise RuntimeError("Channel blocks do not support fused activation")
    if module.algo not in (ConvAlgo.Native, ConvAlgo.MaskImplicitGemm):
        raise RuntimeError("Channel blocks support Native or MaskImplicitGemm only")
    if getattr(module, "conv1x1", False):
        raise RuntimeError("Pointwise convolutions should keep their original Torch path")

    features = sparse_input.features
    weight = module.weight
    if features.dtype != torch.float32 or weight.dtype != torch.float32:
        raise RuntimeError("Channel blocks require FP32 features and weights")
    if features.device != weight.device:
        raise RuntimeError("Channel blocks require features and weights on the same device")
    if weight.ndim != len(module.kernel_size) + 2:
        raise RuntimeError("Unexpected sparse-convolution weight rank")
    if (int(weight.shape[0]) != int(module.out_channels)
            or int(weight.shape[-1]) != int(module.in_channels)):
        raise RuntimeError("Sparse-convolution weights are not in the expected KRSC layout")
    if tuple(int(v) for v in weight.shape[1:-1]) != tuple(module.kernel_size):
        raise RuntimeError("Sparse-convolution weight kernel shape is inconsistent")
    bias = module.bias
    if bias is not None:
        if bias.dtype != torch.float32 or bias.device != features.device:
            raise RuntimeError("Channel blocks require FP32 bias on the features device")
        if tuple(bias.shape) != (int(module.out_channels),):
            raise RuntimeError("Sparse-convolution bias shape is inconsistent")

    # This private API exists in the installed spconv 2.3 runtimes. Refuse a
    # different signature instead of accidentally treating a weight as a bias.
    parameters = list(inspect.signature(module._conv_forward).parameters)
    if parameters[:4] != ["training", "input", "weight", "bias"]:
        raise RuntimeError("Unsupported spconv _conv_forward signature for channel blocks")


def channel_block_subm(module, input_tensor, max_bytes=DEFAULT_MAX_BYTES,
                       original_forward=None, on_split=None):
    """Run a SubM convolution, splitting input/output channels when necessary.

    ``original_forward`` is the *bound*, unwrapped ``module.forward``. Passing it
    preserves the exact original fast path when installing a forward wrapper.
    Without it, the ordinary path uses spconv's explicit-weight inference API.
    ``on_split`` optionally receives one JSON-compatible diagnostic dictionary.

    A module is used by one inference worker at a time. Its channel-count
    attributes are temporarily adjusted for the explicit-weight calls and are
    restored on both success and failure; its Parameters are never modified.
    Every coordinate, kernel offset, input channel and output channel is kept.
    """
    import torch

    max_bytes = operator.index(max_bytes)
    if max_bytes <= 0 or max_bytes > INT32_MAX:
        raise ValueError("max_bytes must be positive and no larger than INT32_MAX")
    if getattr(module, "training", True):
        raise RuntimeError("Channel-block convolution is restricted to eval mode")

    features = input_tensor.features
    if features.ndim != 2:
        raise ValueError("Sparse-convolution features must have shape [rows, channels]")
    rows, input_channels = (int(value) for value in features.shape)
    output_channels = int(module.out_channels)
    if input_channels != int(module.in_channels):
        raise ValueError("Sparse-convolution input channel count is inconsistent")
    if input_channels <= 0 or output_channels <= 0:
        raise ValueError("Sparse-convolution channel counts must be positive")
    element_bytes = int(features.element_size())
    if rows * max(input_channels, output_channels) * element_bytes < max_bytes:
        if original_forward is not None:
            return original_forward(input_tensor)
        return _private_forward(module, input_tensor, module.weight, module.bias)

    _check_split_compatibility(module, input_tensor, torch)
    # Strictly smaller than the budget, matching cumm's strict '< int_max'
    # assertion. Splitting only K would leave a too-large N*C input untouched.
    block_limit = (max_bytes - 1) // (rows * element_bytes)
    if block_limit < 1:
        raise RuntimeError("Even a single channel exceeds the sparse-kernel byte budget")
    input_block = min(input_channels, block_limit)
    output_block = min(output_channels, block_limit)
    if on_split is not None:
        on_split({
            "rows": rows,
            "in_channels": input_channels,
            "out_channels": output_channels,
            "input_block": input_block,
            "output_block": output_block,
            "input_blocks": (input_channels + input_block - 1) // input_block,
            "output_blocks": (output_channels + output_block - 1) // output_block,
            "max_bytes": max_bytes,
            "algorithm": getattr(module.algo, "name", str(module.algo)),
            "precision_changed": False,
        })

    weights = module.weight
    bias = module.bias
    original_input_channels = module.in_channels
    original_output_channels = module.out_channels
    cached_indices = input_tensor.indice_dict
    template = None
    with torch.no_grad():
        full_features = torch.empty((rows, output_channels), dtype=torch.float32,
                                    device=features.device)
        try:
            for out_start in range(0, output_channels, output_block):
                out_stop = min(out_start + output_block, output_channels)
                for in_start in range(0, input_channels, input_block):
                    in_stop = min(in_start + input_block, input_channels)
                    # A view retains the original row stride: materialization
                    # is essential to bound the address range seen by cumm.
                    sparse_chunk = input_tensor.replace_feature(
                        features[:, in_start:in_stop].contiguous())
                    sparse_chunk.indice_dict = cached_indices
                    weight_chunk = weights[out_start:out_stop, ..., in_start:in_stop].contiguous()
                    module.in_channels = in_stop - in_start
                    module.out_channels = out_stop - out_start
                    partial = _private_forward(module, sparse_chunk, weight_chunk, None)
                    if tuple(partial.features.shape) != (rows, out_stop - out_start):
                        raise RuntimeError("Channel-block SubM convolution changed the active row count")
                    if (partial.indices is not input_tensor.indices
                            and not torch.equal(partial.indices, input_tensor.indices)):
                        raise RuntimeError("Channel-block SubM convolution changed coordinates/order")
                    if template is None:
                        # Keep spatial state without holding an additional
                        # potentially multi-gigabyte first-partial feature array.
                        template = partial.replace_feature(
                            torch.empty((rows, 0), dtype=torch.float32, device=features.device))
                    cached_indices = partial.indice_dict
                    destination = full_features[:, out_start:out_stop]
                    if in_start == 0:
                        destination.copy_(partial.features)
                    else:
                        destination.add_(partial.features)
                    del partial, sparse_chunk, weight_chunk, destination
            # Adding bias inside each input block would multiply it by the
            # number of blocks. Add it once after the full FP32 channel sum.
            if bias is not None:
                full_features.add_(bias)
            result = template.replace_feature(full_features)
            result.indice_dict = cached_indices
            return result
        finally:
            module.in_channels = original_input_channels
            module.out_channels = original_output_channels


def smoke_gpu():
    """Compare forced channel splitting with real, unsplit FP32 CUDA kernels."""
    import torch
    import spconv.pytorch as spconv
    from spconv.core import ConvAlgo

    if not torch.cuda.is_available():
        raise RuntimeError("The channel-block smoke test requires the allocated GPU")
    torch.manual_seed(173)
    grid = torch.cartesian_prod(*(torch.arange(4, device="cuda", dtype=torch.int32) for _ in range(3)))
    indices = torch.cat((torch.zeros_like(grid[:, :1]), grid), dim=1)
    rows = int(indices.shape[0])
    source = spconv.SparseConvTensor(torch.randn(rows, 16, device="cuda"), indices, [4, 4, 4], 1)
    first = spconv.SubMConv3d(16, 24, 3, padding=1, bias=True,
                             indice_key="channel_smoke", algo=ConvAlgo.Native).cuda().eval()
    second = spconv.SubMConv3d(24, 16, 3, padding=1, bias=True,
                              indice_key="channel_smoke", algo=ConvAlgo.Native).cuda().eval()
    # A budget of eight channels forces splitting BOTH input and output; the
    # second convolution also verifies reuse of the first layer's indice cache.
    budget = rows * 4 * 8 + 1
    splits = []
    with torch.no_grad():
        first.bias.uniform_(0.2, 0.8)
        second.bias.uniform_(0.2, 0.8)
        expected = first(source)
        actual = channel_block_subm(first, source, budget, on_split=splits.append)
        torch.testing.assert_close(actual.features, expected.features, rtol=2e-4, atol=2e-5)
        if not torch.equal(actual.indices, expected.indices):
            raise AssertionError("Channel splitting changed sparse coordinates")
        expected_second = second(expected)
        actual_second = channel_block_subm(second, actual, budget, on_split=splits.append)
        torch.testing.assert_close(actual_second.features, expected_second.features, rtol=2e-4, atol=2e-5)
        if not torch.equal(actual_second.indices, expected_second.indices):
            raise AssertionError("Channel splitting changed reused sparse coordinates")
        fast = channel_block_subm(first, source, original_forward=first.forward)
        torch.testing.assert_close(fast.features, expected.features, rtol=0, atol=0)
    if (first.in_channels, first.out_channels, second.in_channels, second.out_channels) != (16, 24, 24, 16):
        raise AssertionError("Channel splitting left changed layer dimensions")
    if len(splits) != 2 or any(row["input_blocks"] <= 1 or row["output_blocks"] <= 1 for row in splits):
        raise AssertionError("GPU smoke test did not exercise both channel dimensions")
    torch.cuda.synchronize()
    print("CHANNEL_BLOCK_GPU_SMOKE_OK: " + torch.cuda.get_device_name(0), flush=True)


if __name__ == "__main__":
    smoke_gpu()
