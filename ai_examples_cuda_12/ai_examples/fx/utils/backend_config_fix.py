import torch
from torch.ao.quantization.backend_config import (
    ObservationType,
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    )

def relu_clamp_backend_config_unshare_observers(
        backend_config:BackendConfig
    ):
    clamp_ops = [
        torch.nn.ReLU,
        torch.nn.ReLU6,
        torch.nn.functional.relu,
        torch.nn.functional.relu6,
        torch.clamp,
        "clamp",
        'relu',
        'relu_'
        ]
    clamp_ops_dtype_configs_list = [
        pattern_cfg.dtype_configs
        for pattern_cfg in backend_config.configs
        if pattern_cfg.pattern in clamp_ops
        ]
    clamp_op_to_dtype = dict(zip(clamp_ops, clamp_ops_dtype_configs_list))
    def _get_clamp_op_config(
            op,
            dtype_configs: list[DTypeConfig],
        ) -> BackendPatternConfig:
        return BackendPatternConfig(op) \
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
            .set_dtype_configs(dtype_configs)
    clamp_ops_configs = [
        _get_clamp_op_config(k,v)
        for k,v in clamp_op_to_dtype.items()
    ]
    backend_config.set_backend_pattern_configs(clamp_ops_configs)
    return backend_config