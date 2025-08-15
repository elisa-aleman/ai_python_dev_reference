'''
Based on the outdated pytorch version (<2.0) implementation of openvino backend config from mmrazor:

https://github.com/open-mmlab/mmrazor/blob/main/mmrazor/structures/quantization/backend_config/openvino.py
'''

import torch

from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    )

from torch.ao.quantization.backend_config._common_operator_config_utils import (
    _get_linear_configs,
    _get_conv_configs,
    _get_binary_op_configs,
    _get_share_qparams_op_configs,
    )

__all__ = [
    'get_openvino_backend_config',
    'get_openvino_backend_config_dict',
]

# ===================
# |  DTYPE CONFIGS  |
# ===================
weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.quint8,
        output_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float,
    )
non_weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.quint8,
        output_dtype=torch.quint8,
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


def get_openvino_backend_config() -> BackendConfig:
    """Return the `BackendConfig` for the OpenVINO backend.

    Note:
        Learn more about BackendConfig, please refer to:
        https://github.com/pytorch/pytorch/tree/master/torch/ao/quantization/backend_config # noqa: E501
    """
    linear_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    conv_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    addmm_dtype_configs = [
        weighted_op_qint8_dtype_config,
        ]
    cat_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
        ]
    binary_op_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    # there might be things not supported in fx2trt, but it will error out
    # during fx2trt conversion and can support them after that
    return (
        BackendConfig('openvino')
            .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
            .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
            .set_backend_pattern_configs(_get_addmm_configs(addmm_dtype_configs))
            .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
            .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs))
            .set_backend_pattern_config(_get_cat_config(cat_dtype_configs))
        )

def get_openvino_backend_config_dict():
    """Return the `BackendConfig` for the OpenVINO backend in dictionary
    form."""
    return get_openvino_backend_config().to_dict()
