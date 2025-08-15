import torch
from torch.ao.quantization.backend_config import (
    ObservationType,
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    )

from torch.ao.quantization.fx.graph_module import ObservedGraphModule

def propagate_split_share_qparams_pre_process(
        observed_module: ObservedGraphModule,
        backend_config: BackendConfig,
        ):
    share_qparams_ops = [
        pattern_cfg.pattern
        for pattern_cfg in backend_config.configs
        if pattern_cfg.observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
        ]
    # First share qparams for all split children
    for current_node in observed_module.graph.nodes:
        if current_node.target in [torch.split, torch.chunk]:
            previous_nodes = current_node.all_input_nodes
            if len(previous_nodes) == 1:
                previous_node = previous_nodes[0]
                # if the previous node before split is fake_quantize:
                if 'activation_post_process_' in previous_node.name:
                    child_nodes = [user for user in current_node.users.keys()]
                    for child_node in child_nodes:
                        child_node_users = [user for user in child_node.users.keys()]
                        if len(child_node_users)==1:
                            child_node_next = child_node_users[0]
                            # if the nodes after split are fake_quantize:
                            if 'activation_post_process_' in child_node_next.name:
                                child_node_next.target = previous_node.target
    # do another pass for all share qparams nodes in case they were acidentally altered.
    for current_node in observed_module.graph.nodes:
        if current_node.target in share_qparams_ops:
            previous_nodes = current_node.all_input_nodes
            if len(previous_nodes) == 1:
                previous_node = previous_nodes[0]
                # if the previous node before split is fake_quantize:
                if 'activation_post_process_' in previous_node.name:
                    user_nodes = [user for user in current_node.users.keys()]
                    if len(user_nodes) == 1:
                        user_node = user_nodes[0]
                        # if the nodes after the op are fake_quantize:
                        if 'activation_post_process_' in user_node.name:
                            user_node.target = previous_node.target
    observed_module.delete_all_unused_submodules()
    observed_module.recompile()
    

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