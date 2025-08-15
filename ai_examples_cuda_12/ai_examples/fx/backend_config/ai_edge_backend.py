'''

Supported per tensor affine for activation
Supported per tensor symmetric, per channel symmetric for weights
No support for bias
Reference:

https://ai.google.dev/edge/litert/models/quantization_spec
and
https://github.com/google-ai-edge/ai-edge-torch/blob/f7585fedbc8d3386b0b9e2f5d147fbb1d3fd2d37/ai_edge_torch/quantize/pt2e_quantizer.py#L50C1-L68C44
def _supported_symmetric_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
  supported_operators: Dict[str, List[OperatorPatternType]] = {
      # Both conv and linear should be able to handle relu + hardtanh fusion since
      # those are clamp ops
      "conv2d": [
          [torch.nn.Conv2d, torch.nn.ReLU],
          [torch.nn.Conv2d, F.relu],
          [F.conv2d, torch.nn.ReLU],
          [F.conv2d, F.relu],
      ],
      "linear": [[torch.nn.Linear], [F.linear]],
      "add": [[torch.add]],
      "max_pool2d": [[torch.nn.MaxPool2d], [F.max_pool2d]],
      "adaptive_avg_pool2d": [
          [torch.nn.AdaptiveAvgPool2d],
          [F.adaptive_avg_pool2d],
      ],
  }
  return copy.deepcopy(supported_operators)

and 
https://github.com/google-ai-edge/ai-edge-torch/blob/f7585fedbc8d3386b0b9e2f5d147fbb1d3fd2d37/ai_edge_torch/quantize/pt2e_quantizer.py#L103C3-L135C9
act_quantization_spec = QuantizationSpec(
      dtype=torch.int8,
      quant_min=-128,
      quant_max=127,
      qscheme=torch.per_tensor_affine,
      is_dynamic=is_dynamic,
      observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(eps=2**-12),
  )
  qscheme = (
      torch.per_channel_symmetric if is_per_channel else torch.per_tensor_symmetric
  )
  weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = MinMaxObserver
  if is_qat:
    weight_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize
  elif is_per_channel:
    weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver

  extra_args: Dict[str, Any] = {"eps": 2**-12}
  if is_qat:
    if qscheme == torch.per_tensor_symmetric:
      extra_args["observer"] = MovingAverageMinMaxObserver
    else:
      extra_args["observer"] = MovingAveragePerChannelMinMaxObserver  # type: ignore[dict-item]
  weight_quantization_spec = QuantizationSpec(
      dtype=torch.int8,
      quant_min=-127,
      quant_max=127,
      qscheme=qscheme,
      ch_axis=0,
      is_dynamic=False,
      observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
          **extra_args
      ),

https://github.com/google-ai-edge/ai-edge-torch/blob/f7585fedbc8d3386b0b9e2f5d147fbb1d3fd2d37/ai_edge_torch/quantize/pt2e_quantizer.py#L245C3-L272C4
STATIC_QAT_ONLY_OPS = [
    "conv_bn_relu",
    "conv_bn",
]

# static quantization ops (both PTQ and QAT)
STATIC_OPS = [
    "linear",
    "addmm",
    "conv_relu",
    "conv",
    "adaptive_avg_pool2d",
    "gru_io_only",
    "max_pool2d",
    "add_relu",
    "add",
    "mul_relu",
    "mul",
    "cat",
    "fixed_qparams",
]

DYNAMIC_OPS = [
    "linear",
    "addmm",
    "conv",
    "conv_relu",
]

Notes: Linear+ReLU, Linear+BN+ReLU, can be supported with post_process_for_deploy integration of relu and clamp to quantization parameters.

Fixed QParams ops
Reference:
https://github.com/google-ai-edge/ai-edge-torch/blob/f7585fedbc8d3386b0b9e2f5d147fbb1d3fd2d37/ai_edge_torch/quantize/pt2e_quantizer_utils.py#L711C1-L723C15
- Sigmoid
- Softmax

Shared QParams ops
https://github.com/google-ai-edge/ai-edge-torch/blob/f7585fedbc8d3386b0b9e2f5d147fbb1d3fd2d37/ai_edge_torch/quantize/pt2e_quantizer_utils.py#L963C1-L978C4
def _is_share_obs_or_fq_op(op: Callable) -> bool:
  return op in [
      torch.ops.aten.hardtanh.default,
      torch.ops.aten.hardtanh_.default,
      torch.ops.aten.mean.default,
      torch.ops.aten.mean.dim,
      torch.ops.aten.permute.default,
      torch.ops.aten.permute_copy.default,
      torch.ops.aten.squeeze.dim,
      torch.ops.aten.squeeze_copy.dim,
      torch.ops.aten.adaptive_avg_pool2d.default,
      torch.ops.aten.view_copy.default,
      torch.ops.aten.view.default,
      torch.ops.aten.slice_copy.Tensor,
      torch.ops.aten.flatten.using_ints,
  ]

'''

# mypy: allow-untyped-defs

from typing import Union, Callable
import torch

from torch.ao.quantization.backend_config._common_operator_config_utils import (
    _get_linear_configs,
    _get_conv_configs,
    _get_binary_op_configs,
    _get_cat_config,
    # _get_tensor_info_op_configs, # not sure if necessary
    _add_fixed_qparams_to_dtype_configs,
    # _get_share_qparams_op_configs, # redefine here
    # _get_fixed_qparams_op_configs, # redefine here
    # _get_rnn_op_configs,           # redefine here
)

from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
    )


__all__ = [
    "get_ai_edge_torch_backend_config",
    "get_ai_edge_torch_backend_config_dict"
]


# ===================
# |  DTYPE CONFIGS  |
# ===================


# weighted op int8 dtype config
# this is config for ops that has quantized weights, like linear, conv
weighted_op_qint8_dtype_config = DTypeConfig(
    input_dtype=torch.qint8,
    output_dtype=torch.qint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

non_weighted_op_qint8_dtype_config = DTypeConfig(
    input_dtype=torch.qint8,
    output_dtype=torch.qint8,
)

weighted_dynamic_qint8_dtype_config = DTypeConfig(
    input_dtype=torch.qint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    # currently the dtype check is not yet enabled, so we provided the dtype_configs but
    # it is not really used yet,
    # we will enable it a bit later after we moved everything to backend_config_dict
    is_dynamic=True,
)


# =====================
# |  BACKEND CONFIGS  |
# =====================


def _get_addmm_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    addmm_configs = [
        BackendPatternConfig(torch.addmm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index(
            {
                "bias": 0,
                "input": 1,
                "weight": 2,
            }
        )
    ]
    return addmm_configs


def _get_cat_config(dtype_configs: list[DTypeConfig]) -> BackendPatternConfig:
    cat_config = (
        BackendPatternConfig(torch.cat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs(dtype_configs)
    )
    return cat_config


# Add constraints for fixed qparams ops like sigmoid and tanh to ensure values
# fall within the proper ranges, e.g. [0, 1] for sigmoid, [-1, 1] for tanh
_FIXED_QPARAM_OP_0TO1_CONSTRAINTS = DTypeWithConstraints(
    dtype=torch.qint8,
    quant_min_lower_bound=-128,
    quant_max_upper_bound=127,
    scale_exact_match=1.0 / 256.0,
    zero_point_exact_match=-128,
)


# _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS = DTypeWithConstraints(
#     dtype=torch.qint8,
#     quant_min_lower_bound=-128,
#     quant_max_upper_bound=127,
#     scale_exact_match=2.0 / 256.0,
#     zero_point_exact_match=0,
# )


_FIXED_QPARAMS_OP_TO_CONSTRAINTS: dict[Union[Callable, str], DTypeWithConstraints] = {
    torch.nn.Sigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    torch.sigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "sigmoid": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "sigmoid_": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    torch.nn.Softmax: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
}

def _get_fixed_qparams_op_configs(
    dtype_configs: list[DTypeConfig],
) -> list[BackendPatternConfig]:
    fixed_qparams_op_configs = []
    for fixed_qparam_op, constraints in _FIXED_QPARAMS_OP_TO_CONSTRAINTS.items():
        new_dtype_configs = _add_fixed_qparams_to_dtype_configs(
            dtype_configs, constraints
        )
        fixed_qparams_op_configs.append(
            BackendPatternConfig(fixed_qparam_op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )  # noqa: E131
            .set_dtype_configs(new_dtype_configs)
        )
    return fixed_qparams_op_configs


def _get_share_qparams_op_configs(dtype_configs):
    """Get the operator config for the operators that works for both float and quantized input
    if input is quantized, the output Tensor shares the same quantization parameter
    with input.
    Example operator: adaptive_avgpool2d, reshape, transpose, maxpool2d
    Example observed operator:
    observer_0 - adaptive_avgpool2d - observer_0 (same observer instance as input)
    """

    def _get_share_qprams_op_backend_config(op):
        return (
            BackendPatternConfig(op)
            .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
            .set_dtype_configs(dtype_configs)
        )

    share_qparams_ops = [
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.MaxPool2d,
        torch.nn.functional.adaptive_avg_pool2d,
        torch.nn.functional.max_pool2d,
        torch.flatten,
        torch.repeat_interleave,
        torch.transpose,
        torch.squeeze,
        torch.stack,
        torch.unsqueeze,
        "permute",
        "repeat",
        "repeat_interleave",
        "reshape",
        "resize_",
        "squeeze",
        "squeeze_",
        "transpose",
        "unsqueeze",
        "unsqueeze_",
        "view",
    ]
    return [_get_share_qprams_op_backend_config(op) for op in share_qparams_ops]


def _get_rnn_op_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    rnn_op_configs = []
    for rnn_op, ref_rnn_op in [
        (torch.nn.GRU, torch.ao.nn.quantized.reference.GRU),
    ]:
        rnn_op_configs.append(
            BackendPatternConfig(rnn_op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(rnn_op)
            .set_reference_quantized_module(ref_rnn_op)
        )
    return rnn_op_configs


def get_ai_edge_torch_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) with various additional fp16 ops.
    """
    linear_dtype_configs = [
        weighted_op_qint8_dtype_config,
        weighted_dynamic_qint8_dtype_config,
        ]
    conv_dtype_configs = [
        weighted_op_qint8_dtype_config,
        weighted_dynamic_qint8_dtype_config
        ]
    addmm_dtype_configs = [
        weighted_op_qint8_dtype_config,
        weighted_dynamic_qint8_dtype_config,
        ]
    cat_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
        ]
    binary_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
        ]
    fixed_qparams_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
        ]
    share_qparams_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
        ]
    # tensor_info_op_dtype_configs = [
    #     non_weighted_op_qint8_dtype_config,
    #     ]
    rnn_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
        ]
    return (
        BackendConfig("ai_edge_torch")
            .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
            .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
            .set_backend_pattern_configs(_get_addmm_configs(addmm_dtype_configs))
            .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
            .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs))
            .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs))
            # .set_backend_pattern_configs(_get_tensor_info_op_configs(tensor_info_op_dtype_configs))
            .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs))
            .set_backend_pattern_config(_get_cat_config(cat_dtype_configs))
        )


def get_ai_edge_torch_backend_config_dict():
    """
    Return the `BackendConfig` for ai_edge_torch backend in dictionary form.
    """
    return get_ai_edge_torch_backend_config().to_dict()
