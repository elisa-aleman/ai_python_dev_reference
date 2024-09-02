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