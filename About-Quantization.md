# About Quantization

This guide is under construction

- [ ] Definition
- [ ] Qantization Affine Transformations, quantization parameters, and 8-bit computations
    - [ ] Symmetric vs Asymmetric quantization parameters
    - [ ] Quantization Parameter definition
    - [ ] Quantization Parameter calculation
    - [ ] 16-bit quantization
    - [ ] Per-Tensor and Per-Channel Quantization Schemes
- [ ] Static vs Dynamic Quantization
    - [ ] Static vs Dynamic Quantization differences 
    - [ ] Choosing a method
    - [ ] Static vs. Dynamic Quantization operators (layers)
- [ ] Quantization Aware Training
    - [ ] How recent is QAT
- [ ] Quantization Code implementation 
    - [ ] Pytorch Quantization
        - [ ] Quantization Configuration
            - [ ] FakeQuantize and Observer classes
            - [ ] eager fx and pt2e examples for configuration wrapping classes
        - [ ] Quantization Methods
            - [ ] Eager Mode
                - [ ] Eager mode refactoring for execution
                - [ ] Eager mode PTQ
                    - [ ] Eager mode static
                    - [ ] Eager mode dynamic
                - [ ] Eager mode QAT
            - [ ] FX Symbolic Tracing
                - [ ] FX Symbolic Tracing Refactoring for traceability
            - [ ] PT2E (python 2 export)
                - [ ] PT2E refactoring
                - [ ] PT2E PTQ
                    - [ ] PT2E static
                    - [ ] PT2E dynamic
                - [ ] PT2E QAT
        - [ ] Export method conversion
            - [ ] Eager mode observed to FX 
            - [ ] FX float to PT2E float
            - [ ] FX quantized to PT2E quantized
                - [ ] FX PTQ to PT2E PTQ
                - [ ] FX QAT to PT2E QAT
    - [ ] ONNX
    - [ ] Tensorflow / Keras /TFLite
        - [ ] TF float to PTQ
        - [ ] TF Keras to QAT (tfmot)
            - [ ] Sequential
            - [ ] Functional
- [ ] Backend conversion
    - [ ] Python float to ONNX float
    - [ ] ONNX float to Keras float
    - [ ] Tensorflow / Keras float to Quantized TFLite
    - [ ] Python quantized to ONNX quantized
    - [ ] Python observed QAT to ONNX quantized
    - [ ] Python float to TFLite quantized (ai_edge_torch)
    - [ ] Python quantized to TFLite quantized (ai_edge_torch)
    - [ ] 


```python


pt2e_prepared_model = capture_pre_autograd_graph(fx_prepared_model, example_inputs)
torch.ao.quantization.move_exported_model_to_eval(pt2e_prepared_model)


fx_prepared_model_weight_fake_quants = dict([(m[0].replace('.','_'),m[1]) for m in fx_prepared_model.named_modules() if 'weight_fake_quant' in m[0] and '.activation_post_process' not in m[0]])
fake_quant_counter=0
for current_node in pt2e_prepared_model.graph.nodes:
    if current_node.op == 'call_function' and 'fake_quant' in current_node.target.__name__:
        input_nodes = [node for node in current_node.all_input_nodes]
        input_arg_name = input_nodes[-1].target.replace('self.','').replace('_zero_point','')
        if 'activation_post_process_' in input_arg_name:
            old_fakequant = getattr(fx_prepared_model, input_arg_name)
        if 'weight_fake_quant' in input_arg_name:
            old_fakequant = fx_prepared_model_weight_fake_quants[input_arg_name]
        new_activation_name = f'activation_post_process_{fake_quant_counter}'
        setattr(pt2e_prepared_model, new_activation_name, old_fakequant)
        with pt2e_prepared_model.graph.inserting_after(current_node):
            new_fakequant_node = pt2e_prepared_model.graph.create_node(
                'call_module', new_activation_name, (input_nodes[0],), {}
                )
        for use_node in list(current_node.users):
            use_node.replace_input_with(
                old_input=current_node,
                new_input=new_fakequant_node,
                )
        fake_quant_counter += 1
        pt2e_prepared_model.graph.erase_node(current_node)
        for inp in input_nodes[1:]:
            pt2e_prepared_model.delete_submodule(inp.target)
            pt2e_prepared_model.graph.erase_node(inp)


pt2e_prepared_model.recompile()
pt2e_prepared_model.meta = fx_prepared_model.meta


```