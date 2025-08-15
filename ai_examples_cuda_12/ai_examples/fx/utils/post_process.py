# Modified from outdated `mmrazor.models.quantizers.native_quantizer.TorchNativeQuantizer.post_process_for_deploy
# reference:
# https://github.com/open-mmlab/mmrazor/blob/main/mmrazor/models/quantizers/native_quantizer.py#L253

import torch
from torch.ao.quantization.fx.graph_module import ObservedGraphModule
from torch.ao.quantization import (
    disable_observer,
    enable_fake_quant,
    enable_observer,
    )
from torch.ao.nn.intrinsic import _FusedModule

SUPPORT_QAT_MODULES: tuple = (
    torch.nn.intrinsic.qat.modules.ConvBn1d,
    torch.nn.intrinsic.qat.modules.ConvBn2d,
    torch.nn.intrinsic.qat.modules.ConvBn3d,
    torch.nn.intrinsic.qat.modules.ConvBnReLU1d,
    torch.nn.intrinsic.qat.modules.ConvBnReLU2d,
    torch.nn.intrinsic.qat.modules.ConvBnReLU3d,
    torch.nn.intrinsic.qat.modules.ConvReLU1d,
    torch.nn.intrinsic.qat.modules.ConvReLU2d,
    torch.nn.intrinsic.qat.modules.ConvReLU3d,
    torch.nn.intrinsic.qat.modules.LinearBn1d,
    torch.nn.intrinsic.qat.modules.LinearReLU,
    torch.nn.qat.modules.Conv1d,
    torch.nn.qat.modules.Conv2d,
    torch.nn.qat.modules.Conv3d,
    torch.nn.qat.modules.Linear,
    )

MERGE_BN_MAPPINGS: dict = {
    torch.nn.intrinsic.qat.modules.ConvBn1d: torch.nn.qat.modules.Conv1d,
    torch.nn.intrinsic.qat.modules.ConvBn2d: torch.nn.qat.modules.Conv2d,
    torch.nn.intrinsic.qat.modules.ConvBn3d: torch.nn.qat.modules.Conv3d,
    torch.nn.intrinsic.qat.modules.ConvBnReLU1d: torch.nn.intrinsic.qat.modules.ConvReLU1d,
    torch.nn.intrinsic.qat.modules.ConvBnReLU2d: torch.nn.intrinsic.qat.modules.ConvReLU2d,
    torch.nn.intrinsic.qat.modules.ConvBnReLU3d: torch.nn.intrinsic.qat.modules.ConvReLU3d,
    torch.nn.intrinsic.qat.modules.LinearBn1d: torch.nn.qat.modules.Linear,
}


def fuse_qat_bn_post_process(
        observed_module: ObservedGraphModule,
        qconfig,
        device: str = 'cpu',
        update_weight_with_fakequant: bool = False,
        keep_w_fake_quant: bool = False,
    ):
    """
    `SUPPORT_QAT_MODULES` will be convert to normal modules,
    and BN will be merged and consolidated into conv layers.

    Args:
        observed_module (ObservedGraphModule): Modules after fused and
            observed.
        keep_w_fake_quant (bool, optional): Bool to determine whether to
            keep weight fake-quant op, depending on the backend. Defaults
            to False.                
    """
    def traverse(module):
        for name, child in module.named_children():
            # Trace `SUPPORT_QAT_MODULES` recursively.
            if isinstance(child, SUPPORT_QAT_MODULES):
                # We add w_fakequant once in case some ptq methods have
                # specific operations such as Adaround. So we do Quantize
                # to perform these operations and do dequantize to
                # introduce quantization loss in advance.
                weight_fakequant = child.weight_fake_quant
                # `to_float()` function fuse BN into conv or conv_relu, and
                # also convert a qat module to a normal module.
                # source url: https://github.com/pytorch/pytorch/blob/master/torch/nn/intrinsic/qat/modules/conv_fused.py # noqa: E501
                float_child = child.to_float()
                # Only necessary for cases where the fake quant operation
                # is removed from the graph
                if update_weight_with_fakequant:
                    if issubclass(type(float_child), _FusedModule):
                        float_child[0].weight = weight_fakequant(
                            float_child[0].weight.detach().clone())
                    else:
                        float_child.weight = weight_fakequant(
                            float_child.weight.detach().clone())
                # This is decided by backend type, some backend need
                # explicitly keep the fake quant structure, others don't.
                # ONNXRuntime uses it
                if keep_w_fake_quant:
                    for m in float_child.modules():
                        setattr(m, 'qconfig', qconfig)
                    if type(child) in MERGE_BN_MAPPINGS:
                        new_class = MERGE_BN_MAPPINGS[type(child)]
                        new_child = new_class.from_float(float_child).to(device)
                    else:
                        new_child = type(child).from_float(float_child).to(
                            device)
                    # because weight fakequants and observers are replaced
                    # with base fakequants and base observers, some
                    # initialized args need to be update by running
                    # weight_fake_quant.
                    enable_observer(new_child)
                    new_child.weight_fake_quant(new_child.weight)
                    disable_observer(new_child)
                else:
                    new_child = float_child.to(device)
                setattr(module, name, new_child)
            else:
                traverse(child)
    observed_module.apply(enable_fake_quant)
    observed_module.apply(disable_observer)
    traverse(observed_module)



def merge_relu_clamp_to_qparams_post_process(
        observed_module: ObservedGraphModule,
    ):
    clamp_ops = [
        torch.nn.functional.relu,
        torch.nn.functional.relu6,
        torch.clamp,
        'clamp',
        'relu',
        'relu_'
        ]
    clamp_modules = (
        torch.nn.ReLU,
        torch.nn.ReLU6,
        )
    # delete clamp where
    # op -> clamp -> observer
    for current_node in observed_module.graph.nodes:
        is_clamp = False
        if current_node.op == "call_function":
            is_clamp = current_node.target in clamp_ops
        if current_node.op == "call_module":
            current_op = getattr(observed_module, current_node.target)
            is_clamp = isinstance(current_op, clamp_modules)
        if is_clamp:
            previous_nodes = current_node.all_input_nodes
            if len(previous_nodes) == 1:
                previous_node = previous_nodes[0]
                # if the previous node before clamp is a regular op
                if 'activation_post_process_' not in previous_node.name:
                    next_nodes = [user for user in current_node.users.keys()]
                    if len(next_nodes) == 1:
                        next_node = next_nodes[0]
                        # if the node after the current node is fake_quantize:
                        if 'activation_post_process_' in next_node.name:
                            next_node.replace_input_with(current_node, previous_node)
                            observed_module.graph.erase_node(current_node)
    observed_module.delete_all_unused_submodules()
    observed_module.recompile()
    # delete observer1 and clamp 
    # where op -> observer1 -> clamp -> observer2
    # if observer1!=observer2
    for current_node in observed_module.graph.nodes:
        is_clamp = False
        if current_node.op == "call_function":
            is_clamp = current_node.target in clamp_ops
        if current_node.op == "call_module":
            current_op = getattr(observed_module, current_node.target)
            is_clamp = isinstance(current_op, clamp_modules)
        if is_clamp:
            previous_nodes = current_node.all_input_nodes
            if len(previous_nodes) == 1:
                previous_node = previous_nodes[0]
                # if the previous node before split is fake_quantize:
                if 'activation_post_process_' in previous_node.name:
                    previous_observer = getattr(observed_module, previous_node.name)
                    previous_op_nodes = previous_node.all_input_nodes
                    if len(previous_op_nodes) == 1:
                        previous_op_node = previous_op_nodes[0]
                        next_nodes = [user for user in current_node.users.keys()]
                        if len(next_nodes) == 1:
                            next_node = next_nodes[0]
                            # if the node after the current node is fake_quantize:
                            if 'activation_post_process_' in next_node.name:
                                next_observer = getattr(observed_module, next_node.name)
                                previous_scale, previous_z = previous_observer.calculate_qparams()
                                next_scale, next_z = next_observer.calculate_qparams()
                                same_observer = (
                                    previous_scale == next_scale and
                                    previous_z == next_z
                                    )
                                if not same_observer:
                                    next_node.replace_input_with(current_node, previous_op_node)
                                    observed_module.graph.erase_node(current_node)
                                    observed_module.graph.erase_node(previous_node)
    observed_module.graph.eliminate_dead_code()
    observed_module.delete_all_unused_submodules()
    observed_module.recompile()

