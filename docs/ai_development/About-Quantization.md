# About Quantization

This guide is under construction

Sources:

- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [Pytorch blog: quantization in practice](https://pytorch.org/blog/quantization-in-practice/)
- [ONNX Quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [Tensorflow Quantization docs](https://ai.google.dev/edge/litert/models/model_optimization?hl=en)
- [Huggingface Quantization docs](https://huggingface.co/docs/transformers/main/en/main_classes/quantization)

TODO:

- [ ] ReLU merge and split share qparams on PT2E

## Style Guide

This guide uses a particular style for code blocks for conveying relevant information before entering commands.

For code blocks intended for showing commands entered in an interactive environment:
````
```language
<comment> @ <environment>::<sub environment>::path

<commands>
```
````

Similarly, for files, I like to at least add the file location before showing the contents:
````
```language
<comment> @ path

<contents>
```
````


## Definition

In computation, quantization is the method to convert calculations from a higher precision but memory and processing heavy format to a lower memory and processing format in exchange of the precision that the higher format has. In most cases when talking about quantization of AI models, machine learning and neural networks, the process refers to converting a model's weights and computations from floating point numbers in 32 bit memory, to the 4 times compressed 8 bit format. This is done not only to save memory of the model itself, but to make it possible to calculate the results faster in lesser equipped hardware. For example, while the model might be trained for hours or days on a GPU with faster processing, the final model might be more usable in smaller or mobile CPU devices if it is quantized in this way. This presents many difficulties and technical knowledge so I prepared this guide for myself and others to use when needed.

## Quantization Affine Transformations, quantization parameters, and 8-bit computations

The process of quantizing a tensor (for example, the weights or the input and output of a neural network node) is done via an [affine transformation](https://en.wikipedia.org/wiki/Affine_transformation), in this case specifically, a scaling and translation transformation. Scaling is done to reduce the range of possible 32 bit numbers (`2^32=4294967296` values) to the 8 bit range (`2^8=256` values). Scaling the whole range, however, would be not only inefficient but most of the information would be lost. There needs to be a relevant minimum and maximum value that can be mapped to the 8 bit range and calculate a scaling factor and then the translation can be given to map the value corresponding to zero to correctly handle positive and negative values during calculations. These values are often referred to as **quantization parameters**, which are often *scale* and *zero point*. There is also *minimum value*, *maximum value* for the observed floating point values to be used in the scaling, and *quantization maximum* and *quantization minimum* to refer to the range of 8 bit values to be used. 

Below I show a visual representation of this:

```python
x = tensor(dtype=float32)
quant_min = 0   # or -128 or -127
quant_max = 255 # or 127
z = zero_point

min_value-----------0-------------max_value
       ??            ??             ??
       quant_min----z----quant_max

```

### Symmetric vs Asymmetric quantization parameters

Depending on the calculation being performed, or the backend specifications ([Pytorch](https://pytorch.org/docs/stable/quantization.html), [ONNXRuntime](https://onnxruntime.ai/docs/), [TFLite](https://ai.google.dev/edge/litert/models/model_optimization?hl=en), [TensorRT](https://developer.nvidia.com/tensorrt), [OpenVINO](https://github.com/openvinotoolkit/openvino), etc.) the quantization range of values and zero point mapping might be preferred to be symmetric or asymmetric. 

For symmetric quantization, the absolute values of the input are used to calculate a maximum value, and then mirrored from 0 to represent the minimum value. This is slightly less efficient in terms of mapping range, since some of the range will be unused, but it will increase calculation speed since the zero point does not need to be determined or considered in the calculations, as it will always map to 0.

Visually represented, the symmetric quantization would be like this:

```python
x = tensor(dtype=float32)
quant_min = -128 # asymmetric range or -127 for symmetric range
quant_max = 127
zero_point = 0

-max(abs(x))-----------0-------------max(abs(x))
          ??            ??             ??
          quant_min----0----quant_max
```

This representation is particularly useful in weighted and biased operations, since the multiplication of quantized values introduces the need to calculate an intermediate zero point for the product before adding any other values.

### Quantization Parameter calculation

To calculate the quantization parameters, as well as the quantized tensor with asymmetric quantization, we use the following formulas:

```python
# asymmetric

x = tensor(dtype=float32)

quant_min = 0   # or -128 or -127
quant_max = 255 # or 127

quant_x = round((x / scale) + zero_point).clamp(quant_min, quant_max)
scale = (max(x) - min(x)) / (quant_max - quant_min)

# Solving from formulas:
quant_min = round((min(x) / scale) + zero_point)
zero_point = round(quant_min - (min(x)/scale))
# or
quant_max = round((max(x) / scale) + zero_point)
zero_point = round(quant_max - (max(x)/scale))

z = zero_point

min(x)-----------0-------------max(x)
    ??            ??             ??
    quant_min----z----quant_max


dequant_x = (quant_x - zero_point)*scale


            quant_min----z----quant_max
            ??            ??             ??
approx_min_x-------------0-------------approx_max_x

approx_min_x ~= min(x)
approx_max_x ~= max(x)

```

You will notice that to calculate the quantized value rounding is performed, which then when converted back to a floating point number, loses some accuracy.

Because a model will have numerous layers and calculation, the loss of performance is increased with each layer that is used. To counter this loss in performance, there is usually also a need to reduce the floating point model itself, or to modify the weights to accommodate for this loss. This will be discussed in a later section.

Now, for symmetric quantization, the following formulas are used:

```python
# symmetric

x = tensor(dtype=float32)

quant_min = -128 # asymmetric range or -127 for symmetric range
quant_max = 127
quant_x = round((x / scale)).clamp(quant_min, quant_max)
scale = (2*(max(abs(x)))) / (quant_max - quant_min)
zero_point = 0

-max(abs(x))-----------0-------------max(abs(x))
          ??            ??             ??
          quant_min----0----quant_max

dequant_x = quant_x*scale

                 quant_min----0----quant_max
                 ??            ??             ??
-approx_max_abs_x-------------0-------------approx_max_abs_x

approx_max_abs_x ~= max(abs(x))
```

Notice that the calculations are made simpler as mentioned before.

Now, some backends will use the inverse of the scale formula, to multiply instead of divide by the scale parameter, so one must be careful when trying to replicate a model in a different format.

### Signed and Unsigned quantization ranges

Depending on the backend, some operations can be done on different ranges of the 8-bit representation format.

- Unsigned (asymmetric range): `[0,255]`: 256 values
- Signed asymmetric range: `[-128,127]`: 256 values, zero-point can be wherever.
- Signed symmetric range: `[-127,127]`: 255 values, zero-point is always 0, and is usually used for per-axis calculations in some backends (for example, [TFLite](https://ai.google.dev/edge/litert/models/quantization_spec))


### 16-bit quantization

Some backends will allow for a 16-bit quantization format, which will lower the calculations from the memory heavy 32-bit floating point values to 16-bit floating point values. Because they are still floating point values, less precision is lost, while taking up more calculation power than 8-bit calculations, and not all backends have this implementation, making it a less common option.

### Per-Tensor and Per-Channel / Per-Axis Quantization Schemes

Some operations (but not all) allow for a quantization of a tensor where the parameters are calculated in a more granular manner by using an axis of the tensor as if it were multiple tensors.

For example, in networks that handle 2D image inputs, there are batch, RGB channels, width and height axes. After the input, within the network, the channels don't always represent the colors; however, they are still called channels.

This per-axis or per-channel quantization results in a scaling tensor and a zero point tensor, instead of a single value for each, with a size matching the number of channels in the tensor to be quantized.

The use of per-channel quantization is usually appropriate for the weight tensors, but unless the model is very simple, it is not recommended for input and output quantization (otherwise called activation quantization).

This allows for the range of each channel to not waste space in the 8-bit range when scaling the tensor, since the minimum and maximum values can be different for each of them.

## Static vs Dynamic Quantization modes

Until now, I have discussed the quantization of models by assuming that both the input and output (activation) tensors, as well as weight tensors are quantized. This is called **static** quantization.

While the weight tensors are available in a trained model, the input and output tensors are different with each inference, which makes it so that the parameters can't be calculated directly from the model. To be able to calculate the quantization of input and output parameters, some sort of approximation to the most efficient but still wide enough range of minimums and maximums in the data. This is often done by sending representative data, which is called **calibration**. This is doable for all types of operation layers, but if the data is not representative enough, the model might not perform correctly when new data is introduced. 

However, one can leave the input and output quantization parameters without calculation, and calculate them on the run when using the model. This offers more flexibility with the input, but it introduces more calculations to be done and increases the latency of the model, as well as not being available for all operations, and depending heavily on the backend.

### Choosing a quantization mode

According to the [onnx documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#method-selection), it is better to use static quantization for convolutional models, while dynamic quantization is better for transformer based and recursive neural network models. This is, however, not a rule that will fit every case, and in case of desiring faster performance over precision, static quantization is always preferred.

It is also important to note that dynamic quantization applies to Post Training Quantization but not to Quantization Aware Training. The difference between these is explained in a later section.

### Static vs. Dynamic Quantization support

The available operators for static or dynamic quantization differs by each backend. The most used are listed below:

- Pytorch
    - [Pytorch mode recommendation matrix](https://pytorch.org/docs/stable/quantization.html#quantization-mode-support)
        - Dynamic quantization is recommended for LSTM, MLP, Embedding, and Transformer models using Post Training Quantization, but not for Quantization Aware Training.
    - [Pytorch operator support matrix](https://pytorch.org/docs/stable/quantization.html#operator-support)
        - Convolutions don't support dynamic quantization in pytorch.
        - RNN, GRU, LSTM don't support static quantization in pytorch.
        - Multihead Attention is not supported in either quantization mode in pytorch.
- [ONNX quantization method selection](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#method-selection)
    - There are suggestions but not a list of supported operators
- [Tensorflow dynamic range quantization docs](https://ai.google.dev/edge/litert/models/post_training_quant?hl=en)



## Post-Training Quantization (PTQ) vs. Quantization Aware Training (QAT)

A model can be quantized during or after being trained fully. Moreover, quantization during training is a more recent development, so most quantization documentation in the past is about quantizing the model after training.

**Post-Training Quantization**, or **PTQ** is used when quantizing the model after training fully, usually referring to *static* or *dynamic* quantization, and in the case of static quantization, uses *calibration* representative data to calculate the quantization parameters. The weights of the model are in this case, unaffected by the quantization process except for the scaling and translation that occurs.

**Quantization Aware Training** or **QAT** is a bit different in that it introduces an object that stores the quantization parameters in between operators, then trains the model while actively registering the new minimums and maximums, calculating the new quantization parameters, and optimizing the weights during back propagation to reduce the loss of the model. Like I mentioned before, the rounding that happens in quantization propagates some reduction in performance, but this can be countered by the adjusting of the weights to a place where the quantization and rounding has less of an effect.

An even more recent development is **Learnable Step-size Quantization** or **LSQ**, where these quantization parameters are not calculated using a minimum or a maximum for each input, output, or weight tensor but instead are part of the learnable parameters adjusted during loss optimization and back propagation. This, however, is not yet widely implemented.

### How recent is QAT and LSQ

For quantization aware training, the earliest I could find in an academic context is [a paper by Jacob et al. in 2017](https://arxiv.org/pdf/1712.05877).
Pytorch has had implementations of quantization since 1.4.0 in 2020, which included QAT (although it has improved greatly since), while Tensorflow implemented a full API for QAT with `tensorflow-model-optimization` 0.3.0 in 2020 as well. This makes the API implementation quite recent, although guides have been more available in the last few years. 

[Learned Step Quantization was introduced by Esser et al. in 2020](https://arxiv.org/pdf/1902.08153), with pytorch [having an undocumented private module implementing it](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/_learnable_fake_quantize.py), presumably still under development; and Tensorflow not having this implementation yet as of 2024-09.


## QDQ representation format

While there are quantized operations defined in each backend that correspond to the actual quantized tensors being used for calculation, it is difficult to match all of them when converting from format to format, as well as impractical for developers to know which quantized operations correspond to which float counterparts. It is also not necessary to have these until the model is fully quantized, and there is the need to have an intermediate representation when calibrating or training models that represents the quantization process but keeps the values as floating point values for using in the next operation. This representation consists of an operation, a quantization of the output and a dequantization of the output, followed by the next operation. 

This is often called a **QDQ** representation (Q: quantization, DQ: Dequantization).

```mermaid
graph LR
    I["input (float)"] -->
    QA[Quantization] -->
    DA[De-Quantization] -->
    C["Conv2D (float)"] -->
    QB[Quantization] -->
    DB[De-Quantization] -->
    O["output (float)"]
```

Which is then interpreted by the backend and replaced with quantized versions of the nodes:

```mermaid
graph LR
    I["input (int8)"] -->
    C["Conv2D (int8)"] -->
    O["output (int8)"]
```

This allows for easier conversion between backends but it also allows for nodes where quantization is unsupported to be inserted in the model without much issue.

```mermaid
graph LR
    I["input (float)"] -->
    QA[Quantization] -->
    DA[De-Quantization] -->
    C["Conv2D (float)"] -->
    QB[Quantization] -->
    DB[De-Quantization] -->
    U1[Unsupported op 1] -->
    U2[Unsupported op 2] -->
    QC[Quantization] -->
    DC[De-Quantization] -->
    O["output (float)"]
```

```mermaid
graph LR
    I["input (int8)"] -->
    C["Conv2D (int8)"] -->
    DB[De-Quantization] -->
    U1[Unsupported op 1] -->
    U2[Unsupported op 2] -->
    QC[Quantization] -->
    O["output (int8)"]
```

## Quantization Code implementation

### Examples environment

Example code was run on a docker container with CUDA 12.1 CUDNN 8 poetry and python 3.11.10:


```Dockerfile
# @ ./env_build/dockerfiles/poetry_python_-_3-11_cuda12-1_cv-builds.dockerfile

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV CUDA_INT=121
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1
ENV CUDNN_DIR=/opt/cudnn
ENV CUDACXX=/usr/local/cuda/bin/nvcc-12.1

ENV LD_LIBRARY_PATH=/usr/local/lib64
ENV LD_LIBRARY_PATH=$CUDA_TOOLKIT_ROOT_DIR/lib64:$CUDNN_DIR/lib64:$LD_LIBRARY_PATH

ENV HOME="/root"
WORKDIR /root/workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA="1"

# cannot remove LANG even though https://bugs.python.org/issue19846 is fixed
# last attempted removal of LANG broke many users:
# https://github.com/docker-library/python/pull/570
ENV LANG=C.UTF-8

# avoid tzdata from interrupting build to ask location
RUN apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata; \
    apt-get clean

# runtime dependencies
# pyenv see https://github.com/pyenv/pyenv/wiki#troubleshooting--faq
# plus common utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        ca-certificates \
        curl \
        git \
        libbluetooth-dev \
        libbz2-dev \
        libffi-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxml2-dev \
        libxmlsec1-dev \
        llvm \
        make \
        nano \
        tk-dev \
        unzip \
        uuid-dev \
        vim \
        wget \
        xz-utils \
        zlib1g-dev \
        # usability dependencies for audio and computer vision AI
        ffmpeg \
        g++-12 \
        gcc-12 \
        libgl1 \
        libgomp1 \
        libopencv-dev \
        libprotobuf-dev protobuf-compiler \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# install pyenv and python 3.11.10
# as of 2024-09, pytorch supports 3.12 but tensorflow seems to have some issues
# https://github.com/tensorflow/tensorflow/issues/62003
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git ~/.pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

RUN pyenv install 3.11.10
RUN pyenv global 3.11.10
RUN pyenv rehash

# Install poetry with pipx
RUN pip install pipx
ENV PATH="$PATH:/root/.local/bin"
RUN pipx install poetry && \
    pipx inject poetry poetry-plugin-export && \
    pipx install toml-cli

# Install cmake
# requires libprotobuf-dev protobuf-compiler
RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.4/cmake-3.30.4-linux-x86_64.tar.gz && \
    tar -zxvf cmake-3.30.4-linux-x86_64.tar.gz && \
    rm cmake-3.30.4-linux-x86_64.tar.gz && \
    mv cmake-* cmake

ENV PATH=/root/cmake/bin:$PATH

CMD ["/bin/bash"]

```

Then build and run the container:


```sh
# @ shell(linux/mac_osx/wsl)

docker pull nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
docker build --file ./env_build/dockerfiles/poetry_python_-_3-11_cuda12-1_cv-builds.dockerfile --tag poetry_python:3.11_cuda12.1_cudnn8_cv-builds .
docker run \
    --interactive \
    --tty \
    --detach \
    --ipc=host \
    --shm-size=4gb \
    --gpus all \
    --volume ${PWD%/*}:/v \
    --name ai_dev_example \
    poetry_python:3.11_cuda12.1_cudnn8_cv-builds \
    bash
docker exec -it -w /v/ai_python_dev_reference ai_dev_example bash
```

```sh
# @ ai_dev_example::/v/ai_python_dev_reference
mkdir ai_examples_cuda_12
cd ai_examples_cuda_12
poetry init \
    --name "ai_examples" \
    --description "AI model example code for training and quantization, as well as converting, pytorch, onnx, tflite, and ai_edge_torch" \
    --python "~3.11" \
    --author "Elisa Aleman <elisa.claire.aleman.carreon@gmail.com>" \
    --license "GPL-3.0-or-later" \
    --no-interaction
mkdir ai_examples
touch ai_examples/__init__.py
touch README.md

poetry add torch==2.4.0 torchvision@* tensorflow-cpu@* onnx@* onnxruntime@* ai_edge_torch
poetry install
poetry shell

# @ ai_dev_example::poetry_shell::/v/ai_python_dev_reference/ai_examples_cuda_12
python
```

And to run it after having made the environment:

```sh
# @ shell(linux/mac_osx/wsl)
docker exec -it -w /v/ai_python_dev_reference/ai_examples_cuda_12 ai_dev_example bash

# @ ai_dev_example::/v/ai_python_dev_reference/ai_examples_cuda_12
poetry shell

# @ ai_dev_example::poetry_shell::/v/ai_python_dev_reference/ai_examples_cuda_12
python
```

But it can also be run in a Google Colab notebook with the environment, with the caveat that it will run a specific python version (at the time of writing `3.10.12`), and has no dependency solver.

```python
!pip install torch==2.4.0 torchvision==0.19.0 tensorflow-cpu==2.17.0 onnx==1.16.2 onnxruntime==1.19.2 ai_edge_torch==0.2.0
```

### Pytorch Quantization

Sources:

- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)

#### Model Quantization by Model Modes


Throughout the development of quantization API in pytorch there has been 3 major methods depending on the mode the model is in:

- **Eager Mode quantization**: The Eager Mode model (a model left as written in the class definition) is executed directly, along with quantization stubs if present.
    - This method relies on refactoring the model in such a way that the added quantization and dequantization operations are manually written inside the model.
    - This method requires a high-level knowledge of the compatibility of each operation with quantization, and it is prone to errors and missed optimizations.
- **FX Graph Mode quantization**: An FX Graph is a trace of the operations of the model ahead of time (AoT). When quantizing an FX model, quantization nodes are simply added automatically to the model.
    - This method relies on the model being traceable in a consistent way with proxy (not real) tensors. `for` loops, `if` conditions, and other operations that rely on the proxy tensor being a real tensor cannot be traced, so the model needs to be refactored.
    - Quantization is not the only thing that can be added programmatically to the model, and modules can easily be replaced if necessary.
    - Finally, the model is translated to TorchScript code, so code that is traceable with FX graphs but not compatible with TorchScript translation also needs to be refactored.
    - Tracing can be done at operation or module level, and the modules stay within the model.
- **Pytorch 2 Export (PT2E) mode quantization**: Pytorch 2 Export (PT2E) introduced a new way of tracing the model in its lowest pytorch operators using Torch Dynamo instead.
    - More thorough than FX graph mode tracing, but still won't allow certain code that is untraceable. Most models don't need any refactoring when using this mode.
    - Since it is traced at an operation instead of a module level, the model is entirely flattened, so post-tracing modification is more granular, and therefore, more advanced.
    - Since it is traced at an operation instead of a module level, it is more easily converted to other backends.
    
Each of them has its benefits (although eager mode is the least versatile and hardest to do correctly), and the way to implement them all is different. In all cases, some refactoring of the model might be needed, depending on compatibility issues.

The modes described above aren't necessarily for quantization only; instead, they are modes of representation of a model with different capabilities each, so they will be described below.

Another note is that the FX Graph mode, although very useful, has been declared as "maintenance only", which means no further development will be added. This means that some of the newer backend configurations or updates are not included at all. However, some third party models can only be trained in FX Graph mode and not in PT2E mode, so there is still use for this mode. On the other hand, a fully trained FX Graph model can then be converted to a PT2E model with some ingenuity.

##### Eager Mode models

Eager mode models are, in most cases, only referred to as eager mode when handling quantization, but in fact they are the default way that a model behaves when defined following the standard pytorch structures. A module is defined, and submodules can be added to it, followed by a `forward()` method that describes the call behavior of the model.

A simple example is shown below:

```python
import torch

class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,10,1,1)
        self.bn1 = torch.nn.BatchNorm2d(10)
        # self.pool = torch.nn.AdaptiveMaxPool2d((1,1))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool = torch.nn.MaxPool2d(28,28)
        self.fc = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc(x)
        return x

model = ExampleModel()
```

\**Note: The example models in this guide are extremely simple and will not guarantee good results, but are used just for showing a functioning program.*

To quantize a model in this mode without altering its inherent structure is impossible, since the definitions strictly only include floating point operations. It is because of this that a considerable amount of refactoring is necessary.

The forward call of the model is not recorded previous to its execution as part of the model's memory, instead, it is executed in order every time that a call is made. This might seem unnecessary to describe as it should be common knowledge, but the other modes behave differently from this.

##### FX Graph Mode

Source:
- [Pytorch API docs: torch.fx](https://pytorch.org/docs/stable/fx.html)

An FX Graph is a trace of the calling (with arguments) of submodules, functions, methods and attributes of the model ahead of time. 
It is a flattened representation of the model, with exception of leaf nodes, where the relationship between calls is stored in a graph.



```python
import torch
from torch.ao.quantization.fx.tracer import QuantizationTracer
from torch.fx import GraphModule

class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,10,1,1)
        self.bn1 = torch.nn.BatchNorm2d(10)
        # self.pool = torch.nn.AdaptiveMaxPool2d((1,1))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool = torch.nn.MaxPool2d(28,28)
        self.fc = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc(x)
        return x

model = ExampleModel()
print(model)
'''
ExampleModel(
  (conv1): Conv2d(1, 10, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc): Linear(in_features=10, out_features=10, bias=True)
)
'''

# FX trace the model
tracer = QuantizationTracer(skipped_module_names=[], skipped_module_classes=[])
graph = tracer.trace(model)
traced_fx = GraphModule(tracer.root, graph, 'ExampleModel')
print(traced_fx)
'''
ExampleModel(
  (conv1): Conv2d(1, 10, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc): Linear(in_features=10, out_features=10, bias=True)
)



def forward(self, x):
    conv1 = self.conv1(x);  x = None
    bn1 = self.bn1(conv1);  conv1 = None
    relu = torch.nn.functional.relu(bn1, inplace = False);  bn1 = None
    pool = self.pool(relu);  relu = None
    squeeze = torch.squeeze(pool, dim = 2);  pool = None
    squeeze_1 = torch.squeeze(squeeze, dim = 2);  squeeze = None
    fc = self.fc(squeeze_1);  squeeze_1 = None
    return fc

# To see more debug info, please use `graph_module.print_readable()`
'''

print(str(traced_fx.graph))
'''
graph():
    %x : [num_users=1] = placeholder[target=x]
    %conv1 : [num_users=1] = call_module[target=conv1](args = (%x,), kwargs = {})
    %bn1 : [num_users=1] = call_module[target=bn1](args = (%conv1,), kwargs = {})
    %relu : [num_users=1] = call_function[target=torch.nn.functional.relu](args = (%bn1,), kwargs = {inplace: False})
    %pool : [num_users=1] = call_module[target=pool](args = (%relu,), kwargs = {})
    %squeeze : [num_users=1] = call_function[target=torch.squeeze](args = (%pool,), kwargs = {dim: 2})
    %squeeze_1 : [num_users=1] = call_function[target=torch.squeeze](args = (%squeeze,), kwargs = {dim: 2})
    %fc : [num_users=1] = call_module[target=fc](args = (%squeeze_1,), kwargs = {})
    return fc
'''
```

The model now has this flattened information which can be accessed, parsed, and modified using the API described in the [Pytorch API docs: torch.fx](https://pytorch.org/docs/stable/fx.html)

A `GraphModule` contains the relevant submodules that still need to be called from the `Graph`, which is composed of `Node` objects, which have information about what each operation will be carried, and where the input is coming from.

A disadvantage of FX Graph mode is that the model has to be able to be flattened, with a single graph being able to handle all types of input of the model. This means that in-place modifications, boolean flag flow control and other operations aren't compatible with this mode. 

This means that sometimes, refactoring is necessary for the model. However, compared to the refactoring done for Eager mode models quantization, it is less knowledge intensive of the specific backend specifications, which can be automatically added.

##### Pytorch 2 Export (PT2E) Mode

Sources:
- [Pytorch API docs: torch.export](https://pytorch.org/docs/stable/export.html)
- [Pytorch torch.export Tutorial](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html)
- [Pytorch API docs: TorchDynamo APIs for fine-grained tracing](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html)
- [Pytorch API docs: torch.export IRs: Core Aten IR](https://pytorch.org/docs/stable/torch.compiler_ir.html#core-aten-ir)

With the introduction of torch version 2, the [`torch.export`](https://pytorch.org/docs/stable/export.html),  functionality was added.

With either an initial Eager mode or FX Graph mode model, the PT2E process uses TorchDynamo to trace the model in its most granular form, while still restricting the trace to maintain single-graph representation to be able to export the model to other backends. While TorchDynamo allows for multi-graph representation to avoid issues when tracing through flow control, for loops, if conditions, etc, the `torch.export` process also needs to avoid those to work correctly. Therefore, most (but not all) changes to the model for it to be exportable need to be made similarly to FX Graph mode. The difference however, is that FX Graph mode allows for skipping the tracing of certain nodes, while torch.export must be fully granular.

The result is a model that has only calls to the C++ API operators in `torch.ops.aten`, listed in the [Core Aten intermediate representation table](https://pytorch.org/docs/stable/torch.compiler_ir.html#core-aten-ir). These "ops" (operators) are the building blocks of torch models, and were originally only used internally, but are now useful in the exported graph to find equivalent operators in other backends, such as [OpenXLA](https://github.com/openxla/xla), for example, on [torch-xla](https://github.com/pytorch/xla).

Here's an example:

```python
import torch

class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,10,1,1)
        self.bn1 = torch.nn.BatchNorm2d(10)
        # self.pool = torch.nn.AdaptiveMaxPool2d((1,1))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool = torch.nn.MaxPool2d(28,28)
        self.fc = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc(x)
        return x

model = ExampleModel()
print(model)
'''
ExampleModel(
  (conv1): Conv2d(1, 10, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc): Linear(in_features=10, out_features=10, bias=True)
)
'''

example_inputs = (torch.randn(1,1,28,28),)

pt2e_traced_model = torch.export.export(
    model.eval(),
    example_inputs,
    ).module()

print(pt2e_traced_model)
'''
GraphModule(
  (conv1): Module()
  (bn1): Module()
  (fc): Module()
)



def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    conv1_weight = self.conv1.weight
    conv1_bias = self.conv1.bias
    bn1_weight = self.bn1.weight
    bn1_bias = self.bn1.bias
    fc_weight = self.fc.weight
    fc_bias = self.fc.bias
    bn1_running_mean = self.bn1.running_mean
    bn1_running_var = self.bn1.running_var
    conv2d = torch.ops.aten.conv2d.default(x, conv1_weight, conv1_bias);  x = conv1_weight = conv1_bias = None
    _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(conv2d, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, 0.1, 1e-05);  conv2d = bn1_weight = bn1_bias = bn1_running_mean = bn1_running_var = None
    getitem = _native_batch_norm_legit_no_training[0];  _native_batch_norm_legit_no_training = None
    relu = torch.ops.aten.relu.default(getitem);  getitem = None
    max_pool2d = torch.ops.aten.max_pool2d.default(relu, [28, 28], [28, 28]);  relu = None
    squeeze = torch.ops.aten.squeeze.dim(max_pool2d, 2);  max_pool2d = None
    squeeze_1 = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
    linear = torch.ops.aten.linear.default(squeeze_1, fc_weight, fc_bias);  squeeze_1 = fc_weight = fc_bias = None
    return pytree.tree_unflatten((linear,), self._out_spec)

# To see more debug info, please use `graph_module.print_readable()`
'''
```

#### Quantization Configuration

Pytorch handles the calculation of quantization parameters, as well as the different types of quantization, and the compatibility of quantizeable and non-quantizeable operations with a *Quantization Configuration*, as well as a *Backend Configuration* for earlier methods or a unified *Quantizer* for the PT2E methods. It can determine if the quantization is to 8-bits, what the range of values will be, if it will be symmetric or asymmetric, per-tensor or per-channel, static or dynamic, etc.

The configuration is responsible for determining where to input a *fake quantization* and *observer* modules in the earlier methods, or a quantization and dequantization operation in the PT2E methods.

While the introduced objects are different, it is basically performing a similar task, which I will explain in the following section.

##### FakeQuantize and Observer classes

The [`torch.ao.quantization.fake_quantize.FakeQuantize`](https://pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html) class is a module that encapsulates a quantize and dequantize operation for a given input, following a previous operation or input. It is basically keeping the values as floating point values while introducing the loss in performance that would occur were the model already quantized. Therefore, it is a fake quantization, and not a real one. This is also explained above in the [QDQ representation section](#qdq-representation-format). Inside it there is a [`torch.ao.quantization.observer`](https://pytorch.org/docs/stable/quantization-support.html#torch-ao-quantization-observer) module, which keeps record of the relevant values to generate the quantization parameters.

In the case of Eager mode quantization, these classes are added within the marked quantizeable areas for each module depending on the backend configuration. In the case of FX Graph Mode quantization, these classes are added for all traced nodes depending on the backend configuration. In the case of PT2E mode quantization, it depends on annotation methods registered to the Quantizer.

The type of observer also determines the kind of calculation for quantization parameters that will be made.

Currently the following are available:

Observer class                                                                                                                                                                                                                    | Description                                                                                                                                          |
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
[ObserverBase](https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase)                                                                            | Base observer Module.                                                                                                                                |
[MinMaxObserver](https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver)                                                                      | Observer module for computing the quantization parameters based on the running min and max values.                                                   |
[MovingAverageMinMaxObserver](https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html#torch.ao.quantization.observer.MovingAverageMinMaxObserver)                               | Observer module for computing the quantization parameters based on the moving average of the min and max values.                                     |
[PerChannelMinMaxObserver](https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html#torch.ao.quantization.observer.PerChannelMinMaxObserver)                                        | Observer module for computing the quantization parameters based on the running per channel min and max values.                                       |
[MovingAveragePerChannelMinMaxObserver](https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver.html#torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver) | Observer module for computing the quantization parameters based on the running per channel min and max values.                                       |
[HistogramObserver](https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.HistogramObserver.html#torch.ao.quantization.observer.HistogramObserver)                                                             | The module records the running histogram of tensor values along with min/max values.                                                                 |
[PlaceholderObserver](https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.PlaceholderObserver.html#torch.ao.quantization.observer.PlaceholderObserver)                                                       | Observer that doesn't do anything and just passes its configuration to the quantized module's .from_float(). It is used in dynamic quantization.     |
[RecordingObserver](https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.RecordingObserver.html#torch.ao.quantization.observer.RecordingObserver)                                                             | The module is mainly for debug and records the tensor values during runtime.                                                                         |
[NoopObserver](https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.NoopObserver.html#torch.ao.quantization.observer.NoopObserver)                                                                            | Observer that doesn't do anything and just passes its configuration to the quantized module's .from_float().                                         |
[FixedQParamsObserver](https://github.com/pytorch/pytorch/blob/1266be21f45e91eac32e37f9a7258a14270eab15/torch/ao/quantization/observer.py#L1384)                                                                                  | An undocumented observer used in operations where there is no need for variable quantization parameters.                                             |


There is also more than just one [fake_quantize]() class:

Fake Quantize class                                                                                                                                                                                                 | Description                                                                                                                                       |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
[FakeQuantizeBase](https://pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantizeBase.html#torch.ao.quantization.fake_quantize.FakeQuantizeBase)                                        | Base fake quantize module.                                                                                                                        |
[FakeQuantize](https://pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html#torch.ao.quantization.fake_quantize.FakeQuantize)                                                    | Simulate the quantize and dequantize operations in training time.                                                                                 |
[FixedQParamsFakeQuantize](https://pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize.html#torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize)                | Simulate quantize and dequantize in training time. It is fixed for a specific value range for operations without variable output range.           |
[FusedMovingAvgObsFakeQuantize](https://pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.html#torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize) | A fused module to observe the tensor. It is used in PT2E quantization.                                                                            |
[_LearnableFakeQuantize](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/_learnable_fake_quantize.py)                                                                                            | An undocumented private class that implements LSQ. It is presumably under development on the latest version at the time of writing (torch v2.5.0) |


##### About fused modules

Sources:

- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [Pytorch Eager Mode PTQ Static Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

During the insertion of quantize and dequantize modules in between operations in a model, some groupings of modules benefit from not undergoing quantization until all of the calculations have been finished. Moreover, some modules benefit from entirely forgoing their original structure that can be used in training and evaluation and being replaced with an operation that is optimized for evaluation only. The modules that benefit from this **fusing** of operations vary from backend to backend, but are usually the following:

- Convolution + Batch Normalization + ReLU activation
- Convolution + ReLU activation
- Linear + Batch Normalization + ReLU activation
- Linear + ReLU activation
- Operation + Clamp / Clip range

And so on, the usual fusing involves normalization operations which are more relevant during training, and ReLU activations or clamp or clip operations which are simply a restriction of the range of floating values. This restriction, being equivalent to the restriction imposed by the mapping to quantized values, will simply result in a different mapping that could be assumed from the start, making a double quantization unnecessary. The fusing of these nodes can be studied in [`torch.ao.nn.intrinsic`](https://pytorch.org/docs/stable/quantization-support.html#module-torch.ao.nn.intrinsic) and [`torch.ao.nn.intrinsic.qat`](https://pytorch.org/docs/stable/quantization-support.html#module-torch.ao.nn.intrinsic.qat) for pre-defined classes, and in different examples found in [`torch.ao.quantization.backend_config`](https://github.com/pytorch/pytorch/tree/main/torch/ao/quantization/backend_config) for other backend specific fusings.

Each mode has its own method of fusing. While FX and PT2E modes fusing patterns can be configured without refactoring a model and will be explained in the following sections, Eager mode quantization requires a manual change to the model in order to implement this correctly. This is shown in [Eager Mode fused modules](#eager-mode-fused-modules).

##### Eager Mode Quantization Configuration: QConfig and manual fused modules

Sources:

- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [Pytorch Eager Mode PTQ Static Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [Pytorch Eager Mode PTQ Dynamic Quantization example LSTM](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html?highlight=lstm)
- [Pytorch API docs: QConfig](https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig.QConfig.html)

[QConfig](https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig.QConfig.html) was the first class in pytorch to be implemented for configuring Eager Mode modules.

It is applied to a module directly by adding it to the `torch.nn.Module.qconfig` attribute. Applying it globally to the entire model is often done in Eager mode quantization when the quantization and dequantization submodules are set correctly.

Example:

```python
import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,10,1,1)
        self.bn1 = torch.nn.BatchNorm2d(10)
        # self.pool = torch.nn.AdaptiveMaxPool2d((1,1))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool = torch.nn.MaxPool2d(28,28)
        self.fc = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc(x)
        return x

model = ExampleModel()

model.qconfig = QConfig(
    activation=MovingAverageMinMaxObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        ),
    weight=MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ),
    )
print(model)
'''
ExampleModel(
  (conv1): Conv2d(1, 10, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc): Linear(in_features=10, out_features=10, bias=True)
)
'''

torch.ao.quantization.prepare(model, inplace=True)
print(model)
'''
ExampleModel(
  (conv1): Conv2d(
    1, 10, kernel_size=(1, 1), stride=(1, 1)
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (bn1): BatchNorm2d(
    10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc): Linear(
    in_features=10, out_features=10, bias=True
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
)
'''
```

The model is then configured and ready for the FakeQuantize and Observer insertion during the quantization process described later.

Note that the [tutorials](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html) will use a default QConfig:

```python
import torch

print(torch.ao.quantization.get_default_qconfig('x86'))
'''
QConfig(
    activation=functools.partial(<class 'torch.ao.quantization.observer.HistogramObserver'>,
        reduce_range=True
        ){},
    weight=functools.partial(<class 'torch.ao.quantization.observer.PerChannelMinMaxObserver'>,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric
        ){}
    )
'''
```

This works fine in pytorch but when considering other backends to convert the model to, is not necessarily what we want, not to mention we might prefer other observers even in a native environment.

###### Eager Mode fused modules

Eager mode model before adding fused modules functionality:
```python
import torch

class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool = torch.nn.AdaptiveMaxPool2d((1,1))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool = torch.nn.MaxPool2d(28,28)
        self.fc1 = torch.nn.Linear(10, 128)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 10)
    def forward(self,x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x
```

Eager mode model after adding fused modules for PTQ:

```python
import torch

# PTQ
class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool = torch.nn.AdaptiveMaxPool2d((1,1))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool = torch.nn.MaxPool2d(28,28)
        self.fc1 = torch.nn.Linear(10, 128)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 10)
    def forward(self,x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x
    def fuse_model(self):
        torch.ao.quantization.fuse_modules(
            self,
            ['conv1','bn1','act1'],
            inplace=True,
            )
        torch.ao.quantization.fuse_modules(
            self,
            ['fc1','fc1act'],
            inplace=True,
            )
        

qconfig = QConfig(
    activation=MovingAverageMinMaxObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        ),
    weight=MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ),
    )

model = ExampleModel()
model.qconfig = qconfig
print(model)
'''
ExampleModel(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=10, out_features=128, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
'''

# fusion is only for eval
model.eval()
model.fuse_model()
print(model)
'''
ExampleModel(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (conv1): ConvReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
  )
  (bn1): Identity()
  (act1): Identity()
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc1): LinearReLU(
    (0): Linear(in_features=10, out_features=128, bias=True)
    (1): ReLU()
  )
  (fc1act): Identity()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
'''

type(model.conv1)
# <class 'torch.ao.nn.intrinsic.modules.fused.ConvReLU2d'>

torch.ao.quantization.prepare(model, inplace=True)
print(model)
'''
ExampleModel(
  (quant): QuantStub(
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (dequant): DeQuantStub()
  (conv1): ConvReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (bn1): Identity()
  (act1): Identity()
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc1): LinearReLU(
    (0): Linear(in_features=10, out_features=128, bias=True)
    (1): ReLU()
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (fc1act): Identity()
  (fc2): Linear(
    in_features=128, out_features=10, bias=True
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
)
'''

```

Eager mode model after adding fused modules for QAT:
```python
import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

# QAT
class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool = torch.nn.AdaptiveMaxPool2d((1,1))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool = torch.nn.MaxPool2d(28,28)
        self.fc1 = torch.nn.Linear(10, 128)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 10)
    def forward(self,x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x
    def fuse_model(self):
        torch.ao.quantization.fuse_modules_qat(
            self,
            ['conv1','bn1','act1'],
            inplace=True,
            )
        torch.ao.quantization.fuse_modules_qat(
            self,
            ['fc1','fc1act'],
            inplace=True,
            )


qconfig = QConfig(
    activation=MovingAverageMinMaxObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        ),
    weight=MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ),
    )

model = ExampleModel()
model.qconfig = qconfig
print(model)
'''
ExampleModel(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=10, out_features=128, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
'''

# fusion for qat is done in train
model.train()
model.fuse_model()
print(model)
'''
ExampleModel(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (conv1): ConvBnReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (bn1): Identity()
  (act1): Identity()
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc1): LinearReLU(
    (0): Linear(in_features=10, out_features=128, bias=True)
    (1): ReLU()
  )
  (fc1act): Identity()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
'''

type(model.conv1)
# <class 'torch.ao.nn.intrinsic.modules.fused.ConvBnReLU2d'>


torch.ao.quantization.prepare_qat(model, inplace=True)
print(model)
'''
ExampleModel(
  (quant): QuantStub(
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (dequant): DeQuantStub()
  (conv1): ConvBnReLU2d(
    1, 32, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (bn1): Identity()
  (act1): Identity()
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc1): LinearReLU(
    in_features=10, out_features=128, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (fc1act): Identity()
  (fc2): Linear(
    in_features=128, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
)
'''

type(model.conv1)
# <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d'>

```

Note that the refactoring being different introduces difficulties that aren't present in FX or PT2E modes.


##### FX Mode Quantization Configuration

###### FX Mode Fused Modules

TODO: 
- [ ] Add note about `_fuse_fx()` being useful for qat settings to fuse non-Sequence nodes even for ptq

###### FX Mode QConfigMapping

Sources:
- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [Pytorch API docs: QConfigMapping](https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html)
- [Pytorch FX Quantization user guide](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)
- [Pytorch BackendConfig Tutorial](https://pytorch.org/tutorials/prototype/backend_config_tutorial.html)
- [Pytorch FX PTQ Static Quantization Tutorial](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)
- [Pytorch FX PTQ Dynamic Quantization Tutorial](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html)

QConfigMapping is the solution that FX mode quantization brought to the complexity of setting different configurations across the model if there were exceptions to the overall configuration for specific operations. It allows us to set a global QConfig, similar to the EagerMode example, but also provides optional API to detect different operations and add a specific configuration to those.

For the full usage guide, I recommend reading the [Pytorch API docs: QConfigMapping](https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html), but below I only show an example using global configurations.

```python
import torch
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.fx.tracer import QuantizationTracer
from torch.fx import GraphModule
from torch.ao.quantization.quantize_fx import _fuse_fx, convert_fx
from torch.ao.quantization.fx import prepare


class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool = torch.nn.AdaptiveMaxPool2d((1,1))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool = torch.nn.MaxPool2d(28,28)
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(20, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        return x

model = ExampleModel()

# define the qconfig_mapping
qconfig_mapping = QConfigMapping().set_global(
        QConfig(
            activation=MovingAverageMinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                ),
            weight=MovingAveragePerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                ),
            )
        )

# FX trace the model
tracer = QuantizationTracer(skipped_module_names=[], skipped_module_classes=[])
graph = tracer.trace(model)
traced_fx = GraphModule(tracer.root, graph, 'ExampleModel')
print(traced_fx)
'''
ExampleModel(
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=10, out_features=20, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=20, out_features=10, bias=True)
)



def forward(self, x):
    conv1 = self.conv1(x);  x = None
    bn1 = self.bn1(conv1);  conv1 = None
    act1 = self.act1(bn1);  bn1 = None
    pool = self.pool(act1);  act1 = None
    squeeze = torch.squeeze(pool, dim = 2);  pool = None
    squeeze_1 = torch.squeeze(squeeze, dim = 2);  squeeze = None
    fc1 = self.fc1(squeeze_1);  squeeze_1 = None
    fc1act = self.fc1act(fc1);  fc1 = None
    fc2 = self.fc2(fc1act);  fc1act = None
    return fc2

# To see more debug info, please use `graph_module.print_readable()`
'''

# FX fuse modules automatically
fused_fx = _fuse_fx(
    traced_fx,
    is_qat=True, #useful for non Sequence types even if not qat
    backend_config=None, # default native
    )
print(fused_fx)
'''
GraphModule(
  (conv1): ConvBnReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc1): LinearReLU(
    (0): Linear(in_features=10, out_features=20, bias=True)
    (1): ReLU()
  )
  (fc2): Linear(in_features=20, out_features=10, bias=True)
)



def forward(self, x):
    conv1 = self.conv1(x);  x = None
    pool = self.pool(conv1);  conv1 = None
    squeeze = torch.squeeze(pool, dim = 2);  pool = None
    squeeze_1 = torch.squeeze(squeeze, dim = 2);  squeeze = None
    fc1 = self.fc1(squeeze_1);  squeeze_1 = None
    fc2 = self.fc2(fc1);  fc1 = None
    return fc2

# To see more debug info, please use `graph_module.print_readable()`
'''


# FX prepare the quantization nodes
example_inputs = (torch.randn(1,1,28,28),)
prepared_fx = prepare(
    fused_fx,
    qconfig_mapping=qconfig_mapping,
    node_name_to_scope=tracer.node_name_to_scope,
    is_qat=True, # convenient even if not QAT
    example_inputs=example_inputs,
    backend_config=None, # default native
    )
# Observe at least one input so observers aren't empty or inf
prepared_fx(example_inputs[0])
print(prepared_fx)
'''
GraphModule(
  (activation_post_process_0): MovingAverageMinMaxObserver(min_val=-3.5789921283721924, max_val=2.6757922172546387)
  (conv1): ConvBnReLU2d(
    1, 32, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.9399,  0.0250,  0.6221,  0.0420,  0.3840,  0.8214,  0.9261,  0.4580,
              -0.9189,  0.5860, -0.8352,  0.9982,  0.1032, -0.8320,  0.7838,  0.0279,
               0.6654,  0.6904, -0.7964, -0.6220,  0.7120,  0.0911,  0.1661,  0.1380,
               0.8009,  0.2910,  0.0789,  0.5088, -0.7149, -0.2357,  0.2789, -0.7154]), max_val=tensor([-0.9399,  0.0250,  0.6221,  0.0420,  0.3840,  0.8214,  0.9261,  0.4580,
              -0.9189,  0.5860, -0.8352,  0.9982,  0.1032, -0.8320,  0.7838,  0.0279,
               0.6654,  0.6904, -0.7964, -0.6220,  0.7120,  0.0911,  0.1661,  0.1380,
               0.8009,  0.2910,  0.0789,  0.5088, -0.7149, -0.2357,  0.2789, -0.7154])
    )
  )
  (activation_post_process_1): MovingAverageMinMaxObserver(min_val=0.07630567997694016, max_val=3.6066136360168457)
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (activation_post_process_2): MovingAverageMinMaxObserver(min_val=0.07630567997694016, max_val=3.6066136360168457)
  (activation_post_process_3): MovingAverageMinMaxObserver(min_val=0.07630567997694016, max_val=3.6066136360168457)
  (activation_post_process_4): MovingAverageMinMaxObserver(min_val=0.07630567997694016, max_val=3.6066136360168457)
  (fc1): LinearReLU(
    in_features=10, out_features=20, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.2866, -0.2528, -0.2625, -0.2199, -0.3122, -0.1569, -0.3133, -0.1193,
              -0.2822, -0.2716, -0.2971, -0.2418, -0.2501, -0.1576, -0.3001, -0.0644,
              -0.2866, -0.2405, -0.1759, -0.2983]), max_val=tensor([ 0.2967,  0.2732,  0.3098,  0.2630,  0.2673,  0.2379,  0.2764,  0.2433,
               0.2138,  0.2066,  0.2864,  0.3044, -0.0524,  0.3081,  0.3109,  0.3048,
               0.2757,  0.2985,  0.2056,  0.1747])
    )
  )
  (activation_post_process_5): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (fc2): Linear(
    in_features=20, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_6): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
)



def forward(self, x):
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
    conv1 = self.conv1(activation_post_process_0);  activation_post_process_0 = None
    activation_post_process_1 = self.activation_post_process_1(conv1);  conv1 = None
    pool = self.pool(activation_post_process_1);  activation_post_process_1 = None
    activation_post_process_2 = self.activation_post_process_2(pool);  pool = None
    squeeze = torch.squeeze(activation_post_process_2, dim = 2);  activation_post_process_2 = None
    activation_post_process_3 = self.activation_post_process_3(squeeze);  squeeze = None
    squeeze_1 = torch.squeeze(activation_post_process_3, dim = 2);  activation_post_process_3 = None
    activation_post_process_4 = self.activation_post_process_4(squeeze_1);  squeeze_1 = None
    fc1 = self.fc1(activation_post_process_4);  activation_post_process_4 = None
    activation_post_process_5 = self.activation_post_process_5(fc1);  fc1 = None
    fc2 = self.fc2(activation_post_process_5);  activation_post_process_5 = None
    activation_post_process_6 = self.activation_post_process_6(fc2);  fc2 = None
    return activation_post_process_6

# To see more debug info, please use `graph_module.print_readable()`
'''

# different from eager mode
print(hasattr(prepared_fx, 'qconfig'))
'''
False
'''

# instead:
print(hasattr(prepared_fx, 'meta'))
'''
True
'''
import pprint
pprint.pprint(prepared_fx.meta)
'''
{'_observed_graph_module_attrs': ObservedGraphModuleAttrs(node_name_to_qconfig={'conv1': QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.MovingAverageMinMaxObserver'>, dtype=torch.quint8, qscheme=torch.per_tensor_affine){'factory_kwargs': <function _add_module_to_qconfig_obs_ctr.<locals>.get_factory_kwargs_based_on_module_device at 0x7f56c83e80e0>}, weight=functools.partial(<class 'torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_channel_symmetric){'factory_kwargs': <function _add_module_to_qconfig_obs_ctr.<locals>.get_factory_kwargs_based_on_module_device at 0x7f56c83e80e0>}),
                                                                                'fc1': QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.MovingAverageMinMaxObserver'>, dtype=torch.quint8, qscheme=torch.per_tensor_affine){'factory_kwargs': <function _add_module_to_qconfig_obs_ctr.<locals>.get_factory_kwargs_based_on_module_device at 0x7f56c83e8040>}, weight=functools.partial(<class 'torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_channel_symmetric){'factory_kwargs': <function _add_module_to_qconfig_obs_ctr.<locals>.get_factory_kwargs_based_on_module_device at 0x7f56c83e8040>}),
                                                                                'fc2': QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.MovingAverageMinMaxObserver'>, dtype=torch.quint8, qscheme=torch.per_tensor_affine){'factory_kwargs': <function _add_module_to_qconfig_obs_ctr.<locals>.get_factory_kwargs_based_on_module_device at 0x7f56c83c8040>}, weight=functools.partial(<class 'torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_channel_symmetric){'factory_kwargs': <function _add_module_to_qconfig_obs_ctr.<locals>.get_factory_kwargs_based_on_module_device at 0x7f56c83c8040>}),
                                                                                'output': None,
                                                                                'pool': QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.MovingAverageMinMaxObserver'>, dtype=torch.quint8, qscheme=torch.per_tensor_affine){'factory_kwargs': <function _add_module_to_qconfig_obs_ctr.<locals>.get_factory_kwargs_based_on_module_device at 0x7f56c83e89a0>}, weight=functools.partial(<class 'torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_channel_symmetric){'factory_kwargs': <function _add_module_to_qconfig_obs_ctr.<locals>.get_factory_kwargs_based_on_module_device at 0x7f56c83e89a0>}),
                                                                                'squeeze': QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.MovingAverageMinMaxObserver'>, dtype=torch.quint8, qscheme=torch.per_tensor_affine){}, weight=functools.partial(<class 'torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_channel_symmetric){}),
                                                                                'squeeze_1': QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.MovingAverageMinMaxObserver'>, dtype=torch.quint8, qscheme=torch.per_tensor_affine){}, weight=functools.partial(<class 'torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_channel_symmetric){}),
                                                                                'x': None},
                                                          node_name_to_scope={'act1': ('act1',
                                                                                       <class 'torch.nn.modules.activation.ReLU'>),
                                                                              'bn1': ('bn1',
                                                                                      <class 'torch.nn.modules.batchnorm.BatchNorm2d'>),
                                                                              'conv1': ('conv1',
                                                                                        <class 'torch.nn.modules.conv.Conv2d'>),
                                                                              'fc1': ('fc1',
                                                                                      <class 'torch.nn.modules.linear.Linear'>),
                                                                              'fc1act': ('fc1act',
                                                                                         <class 'torch.nn.modules.activation.ReLU'>),
                                                                              'fc2': ('fc2',
                                                                                      <class 'torch.nn.modules.linear.Linear'>),
                                                                              'output': ('',
                                                                                         None),
                                                                              'pool': ('pool',
                                                                                       <class 'torch.nn.modules.pooling.MaxPool2d'>),
                                                                              'squeeze': ('',
                                                                                          None),
                                                                              'squeeze_1': ('',
                                                                                            None),
                                                                              'x': ('',
                                                                                    None)},
                                                          prepare_custom_config=PrepareCustomConfig({}),
                                                          equalization_node_name_to_qconfig={'conv1': None,
                                                                                             'fc1': None,
                                                                                             'fc2': None,
                                                                                             'output': None,
                                                                                             'pool': None,
                                                                                             'squeeze': None,
                                                                                             'squeeze_1': None,
                                                                                             'x': None},
                                                          qconfig_mapping=QConfigMapping (
 global_qconfig
  QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.MovingAverageMinMaxObserver'>, dtype=torch.quint8, qscheme=torch.per_tensor_affine){}, weight=functools.partial(<class 'torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_channel_symmetric){})
 object_type_qconfigs
  OrderedDict()
 module_name_regex_qconfigs
  OrderedDict()
 module_name_qconfigs
  OrderedDict()
 module_name_object_type_order_qconfigs
  OrderedDict()
),
                                                          is_qat=True,
                                                          observed_node_names={'conv1',
                                                                               'fc1',
                                                                               'fc2',
                                                                               'pool',
                                                                               'squeeze',
                                                                               'squeeze_1'},
                                                          is_observed_standalone_module=False,
                                                          standalone_module_input_quantized_idxs=None,
                                                          standalone_module_output_quantized_idxs=None)}
'''

quantized_fx = convert_fx(
    prepared_fx,
    qconfig_mapping=qconfig_mapping,
    backend_config=None, # default native
    )
print(quantized_fx)
'''
GraphModule(
  (conv1): QuantizedConvReLU2d(1, 32, kernel_size=(1, 1), stride=(1, 1), scale=0.014143582433462143, zero_point=0)
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc1): QuantizedLinearReLU(in_features=10, out_features=20, scale=1.0, zero_point=0, qscheme=torch.per_channel_affine)
  (fc2): QuantizedLinear(in_features=20, out_features=10, scale=1.0, zero_point=0, qscheme=torch.per_channel_affine)
)



def forward(self, x):
    conv1_input_scale_0 = self.conv1_input_scale_0
    conv1_input_zero_point_0 = self.conv1_input_zero_point_0
    quantize_per_tensor = torch.quantize_per_tensor(x, conv1_input_scale_0, conv1_input_zero_point_0, torch.quint8);  x = conv1_input_scale_0 = conv1_input_zero_point_0 = None
    conv1 = self.conv1(quantize_per_tensor);  quantize_per_tensor = None
    pool = self.pool(conv1);  conv1 = None
    squeeze = torch.squeeze(pool, dim = 2);  pool = None
    squeeze_1 = torch.squeeze(squeeze, dim = 2);  squeeze = None
    fc1 = self.fc1(squeeze_1);  squeeze_1 = None
    fc2 = self.fc2(fc1);  fc1 = None
    dequantize_6 = fc2.dequantize();  fc2 = None
    return dequantize_6

# To see more debug info, please use `graph_module.print_readable()`
'''

```

As printed above, there are new fake_quantize/observer nodes added to the forward code and to the module as submodules, but the configuration is not saved in the `qconfig` attribute but instead in a `meta` dictionary that stores the configuration for all inserted nodes.

###### FX Mode BackendConfig and BackendPatternConfig


Sources:
- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [Pytorch API docs: QConfigMapping](https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html)
- [Pytorch API docs: torch.fx](https://pytorch.org/docs/stable/fx.html)
- [Pytorch FX Quantization user guide](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)
- [Pytorch BackendConfig Tutorial](https://pytorch.org/tutorials/prototype/backend_config_tutorial.html)
- [`torch.ao.quantization.backend_config` README](https://github.com/pytorch/pytorch/tree/main/torch/ao/quantization/backend_config)
- [Pytorch FX PTQ Static Quantization Tutorial](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)
- [Pytorch FX PTQ Dynamic Quantization Tutorial](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html)


Reading the [`torch.ao.quantization.backend_config` README](https://github.com/pytorch/pytorch/tree/main/torch/ao/quantization/backend_config) should give a better introduction to the configuration than what I can convey here, but I will still summarize.

Different backends will have different specifications of which modules can and cannot be quantized, what data types they allow, 8-bit value ranges, which modules should be fused or not, or which operations should share the same quantization parameters than their previous operation.

While new backends can be configured for manually by specifying a new `BackendConfig` class in each project, pytorch has several configurations already available:

```python
from torch.ao.quantization.backend_config import (
    get_fbgemm_backend_config,
    get_native_backend_config,
    get_qnnpack_backend_config,
    get_tensorrt_backend_config,
    get_executorch_backend_config,
    )
```

Namely for the following backends:

- `get_fbgemm_backend_config`: [FBGEMM](https://github.com/pytorch/FBGEMM)
- `get_native_backend_config`: Pytorch native inference
- `get_qnnpack_backend_config`: [QNNPACK](https://github.com/pytorch/QNNPACK), an outdated backend later replaced by XNNPACK. This configuration is forward compatible with XNNPACK too.
- `get_tensorrt_backend_config`: [TensorRT](https://github.com/NVIDIA/TensorRT), an NVIDIA specialized backend that makes full use of the GPU CUDA, making it faster than native inference.
- `get_executorch_backend_config`: [executorch](https://github.com/pytorch/executorch), an on-device AI backend for mobile inference of torch models.

There are also a few backend configurations available in third party packages that we can use for reference:

- [`mmrazor.structures.quantization.backend_config.openvino`](https://github.com/open-mmlab/mmrazor/blob/main/mmrazor/structures/quantization/backend_config/openvino.py): This backend configuration was created in torch v1.13.0, before the [backwards incompatible change to BackendPatternConfig introduced in torch 2.0.0](https://github.com/pytorch/pytorch/pull/90698), which can be seen in the [torch 2.0.0 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.0.0)

However, it should be easy to modify to the latest version if necessary.


##### PT2E Quantization Configuration: Quantizer class

- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [Pytorch 2 Export Tutorial](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html)
- [Pytorch 2 Export Quantization Tutorial](https://pytorch.org/tutorials/prototype/quantization_in_pytorch_2_0_export_tutorial.html)
- [Pytorch 2 Export QAT Tutorial](https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html)
- [Pytorch 2 torch compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

While the Eager mode works by manually setting the QConfig class to each submodule, and the FX mode works by assigning a QConfigMapping class to the preparation process, the PT2E mode works by adding [QuantizationAnnotation](https://github.com/pytorch/pytorch/blob/v2.4.0/torch/ao/quantization/quantizer/quantizer.py#L114) objects with a custom annotate method added to the customized [Quantizer](https://github.com/pytorch/pytorch/blob/v2.4.0/torch/ao/quantization/quantizer/quantizer.py) subclass that we would use.

Currently, there are only a few backends prepared for this without adding a custom Quantizer subclass:

- [EmbeddingQuantizer](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/quantizer/embedding_quantizer.py)
- [X86InductorQuantizer](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/quantizer/x86_inductor_quantizer.py)
- [XNNPACKQuantizer](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/quantizer/xnnpack_quantizer.py)

There's also third party quantizers, such as the one for [`ai_edge_torch`](https://github.com/google-ai-edge/ai-edge-torch) used for exporting torch models to TFLite.

- [`ai_edge_torch.quantize.pt2e_quantizer.PT2E_Quantizer`](https://github.com/google-ai-edge/ai-edge-torch/blob/v0.2.0/ai_edge_torch/quantize/pt2e_quantizer.py#L243)

As it is difficult to make a quantizer class from scratch at this point, I will show an example with one of the provided classes.

```python
import torch
# quantizer class
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer

# config class to set up the quantizer
# the placing of the module is strange, but this class is used for other quantizers as well.
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig

# individual settings for each type of tensor use this class for config
from torch.ao.quantization.quantizer import QuantizationSpec

# observer class
from torch.ao.quantization.observer import MovingAverageMinMaxObserver

# fake quant class
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize

# pt2e insert observers in traced module, and quantize final model
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e

input_output_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(),
    )
weight_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False,
        observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
            ),
    )
bias_spec = None
quantization_config = QuantizationConfig(
        input_output_spec,
        input_output_spec,
        weight_spec,
        bias_spec,
        is_qat=True,
    )
quantizer = XNNPACKQuantizer().set_global(quantization_config)

class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool = torch.nn.AdaptiveMaxPool2d((1,1))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool = torch.nn.MaxPool2d(28,28)
        self.fc1 = torch.nn.Linear(10, 128)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        return x

model = ExampleModel()
print(model)
'''
ExampleModel(
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=10, out_features=128, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
'''

example_inputs = (torch.randn(1,1,28,28),)

# for pt2e ptq: model.eval()
# for pt2e qat: model.train()

pt2e_traced_model = torch.export.export(
    model.eval(),
    example_inputs,
    ).module()


print(pt2e_traced_model)
'''
GraphModule(
  (conv1): Module()
  (bn1): Module()
  (fc): Module()
)



def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    conv1_weight = self.conv1.weight
    conv1_bias = self.conv1.bias
    bn1_weight = self.bn1.weight
    bn1_bias = self.bn1.bias
    fc_weight = self.fc.weight
    fc_bias = self.fc.bias
    bn1_running_mean = self.bn1.running_mean
    bn1_running_var = self.bn1.running_var
    conv2d = torch.ops.aten.conv2d.default(x, conv1_weight, conv1_bias);  x = conv1_weight = conv1_bias = None
    _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(conv2d, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, 0.1, 1e-05);  conv2d = bn1_weight = bn1_bias = bn1_running_mean = bn1_running_var = None
    getitem = _native_batch_norm_legit_no_training[0];  _native_batch_norm_legit_no_training = None
    relu = torch.ops.aten.relu.default(getitem);  getitem = None
    max_pool2d = torch.ops.aten.max_pool2d.default(relu, [28, 28], [28, 28]);  relu = None
    squeeze = torch.ops.aten.squeeze.dim(max_pool2d, 2);  max_pool2d = None
    squeeze_1 = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
    linear = torch.ops.aten.linear.default(squeeze_1, fc_weight, fc_bias);  squeeze_1 = fc_weight = fc_bias = None
    return pytree.tree_unflatten((linear,), self._out_spec)

# To see more debug info, please use `graph_module.print_readable()`
'''

pt2e_prepared_model = prepare_pt2e(pt2e_traced_model, quantizer)
# Observe at least one input
pt2e_prepared_model(example_inputs[0])
print(pt2e_prepared_model)
'''
GraphModule(
  (conv1): Module()
  (activation_post_process_1): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0072]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.5439894199371338, max_val=0.9105219841003418)
  )
  (fc): Module()
  (activation_post_process_6): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0025]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.2811250388622284, max_val=0.31249532103538513)
  )
  (activation_post_process_0): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0227]), zero_point=tensor([5], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-3.0279316902160645, max_val=2.7590487003326416)
  )
  (activation_post_process_2): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0137]), zero_point=tensor([-128], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.4919281005859375)
  )
  (activation_post_process_3): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0137]), zero_point=tensor([-128], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.4919281005859375)
  )
  (activation_post_process_4): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0137]), zero_point=tensor([-128], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.4919281005859375)
  )
  (activation_post_process_5): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0137]), zero_point=tensor([-128], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.4919281005859375)
  )
  (activation_post_process_7): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0053]), zero_point=tensor([-67], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.3214762508869171, max_val=1.03191339969635)
  )
)



def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    conv1_weight = self.conv1.weight
    activation_post_process_1 = self.activation_post_process_1(conv1_weight);  conv1_weight = None
    conv1_bias = self.conv1.bias
    fc_weight = self.fc.weight
    activation_post_process_6 = self.activation_post_process_6(fc_weight);  fc_weight = None
    fc_bias = self.fc.bias
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
    conv2d = torch.ops.aten.conv2d.default(activation_post_process_0, activation_post_process_1, conv1_bias);  activation_post_process_0 = activation_post_process_1 = conv1_bias = None
    relu = torch.ops.aten.relu.default(conv2d);  conv2d = None
    activation_post_process_2 = self.activation_post_process_2(relu);  relu = None
    max_pool2d = torch.ops.aten.max_pool2d.default(activation_post_process_2, [28, 28], [28, 28]);  activation_post_process_2 = None
    activation_post_process_3 = self.activation_post_process_3(max_pool2d);  max_pool2d = None
    squeeze = torch.ops.aten.squeeze.dim(activation_post_process_3, 2);  activation_post_process_3 = None
    activation_post_process_4 = self.activation_post_process_4(squeeze);  squeeze = None
    squeeze_1 = torch.ops.aten.squeeze.dim(activation_post_process_4, 2);  activation_post_process_4 = None
    activation_post_process_5 = self.activation_post_process_5(squeeze_1);  squeeze_1 = None
    linear = torch.ops.aten.linear.default(activation_post_process_5, activation_post_process_6, fc_bias);  activation_post_process_5 = activation_post_process_6 = fc_bias = None
    activation_post_process_7 = self.activation_post_process_7(linear);  linear = None
    return pytree.tree_unflatten((activation_post_process_7,), self._out_spec)

# To see more debug info, please use `graph_module.print_readable()`
'''


from pprint import pprint
pprint(pt2e_prepared_model.meta)
'''
{'_observed_graph_module_attrs': ObservedGraphModuleAttrs(node_name_to_qconfig={},
                                                          node_name_to_scope={'_native_batch_norm_legit_no_training': ('bn1',
                                                                                                                       'torch.nn.modules.batchnorm.BatchNorm2d'),
                                                                              'bn1_bias': ('',
                                                                                           <class 'NoneType'>),
                                                                              'bn1_running_mean': ('',
                                                                                                   <class 'NoneType'>),
                                                                              'bn1_running_var': ('',
                                                                                                  <class 'NoneType'>),
                                                                              'bn1_weight': ('',
                                                                                             <class 'NoneType'>),
                                                                              'conv1_bias': ('',
                                                                                             <class 'NoneType'>),
                                                                              'conv1_weight': ('',
                                                                                               <class 'NoneType'>),
                                                                              'conv2d': ('conv1',
                                                                                         'torch.nn.modules.conv.Conv2d'),
                                                                              'fc_bias': ('',
                                                                                          <class 'NoneType'>),
                                                                              'fc_weight': ('',
                                                                                            <class 'NoneType'>),
                                                                              'getitem': ('bn1',
                                                                                          'torch.nn.modules.batchnorm.BatchNorm2d'),
                                                                              'linear': ('fc',
                                                                                         'torch.nn.modules.linear.Linear'),
                                                                              'max_pool2d': ('pool',
                                                                                             'torch.nn.modules.pooling.MaxPool2d'),
                                                                              'output_1': ('',
                                                                                           <class 'NoneType'>),
                                                                              'relu': ('',
                                                                                       '__main__.ExampleModel'),
                                                                              'squeeze': ('',
                                                                                          '__main__.ExampleModel'),
                                                                              'squeeze_1': ('',
                                                                                            '__main__.ExampleModel'),
                                                                              'x': ('',
                                                                                    <class 'NoneType'>)},
                                                          prepare_custom_config=PrepareCustomConfig({}),
                                                          equalization_node_name_to_qconfig={},
                                                          qconfig_mapping=QConfigMapping (
 global_qconfig
  None
 object_type_qconfigs
  OrderedDict()
 module_name_regex_qconfigs
  OrderedDict()
 module_name_qconfigs
  OrderedDict()
 module_name_object_type_order_qconfigs
  OrderedDict()
),
                                                          is_qat=False,
                                                          observed_node_names=set(),
                                                          is_observed_standalone_module=False,
                                                          standalone_module_input_quantized_idxs=None,
                                                          standalone_module_output_quantized_idxs=None),
 'dynamo_flat_name_to_original_fqn': {'L__self___bn1_bias': 'bn1.bias',
                                      'L__self___bn1_num_batches_tracked': 'bn1.num_batches_tracked',
                                      'L__self___bn1_running_mean': 'bn1.running_mean',
                                      'L__self___bn1_running_var': 'bn1.running_var',
                                      'L__self___bn1_weight': 'bn1.weight',
                                      'L__self___conv1_bias': 'conv1.bias',
                                      'L__self___conv1_weight': 'conv1.weight',
                                      'L__self___fc_bias': 'fc.bias',
                                      'L__self___fc_weight': 'fc.weight'},
 'forward_arg_names': ['x'],
 'inline_constraints': {},
 'input_shape_constraints': [],
 'module_call_specs': {}}
'''


pt2e_quantized_model = convert_pt2e(pt2e_prepared_model)
print(pt2e_quantized_model)
'''
GraphModule(
  (conv1): Module()
  (fc): Module()
)



def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _frozen_param0 = self._frozen_param0
    dequantize_per_tensor_default = torch.ops.quantized_decomposed.dequantize_per_tensor.default(_frozen_param0, 0.007141348905861378, 0, -128, 127, torch.int8);  _frozen_param0 = None
    conv1_bias = self.conv1.bias
    _frozen_param1 = self._frozen_param1
    dequantize_per_tensor_default_1 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(_frozen_param1, 0.0024509436916559935, 0, -128, 127, torch.int8);  _frozen_param1 = None
    fc_bias = self.fc.bias
    quantize_per_tensor_default_2 = torch.ops.quantized_decomposed.quantize_per_tensor.default(x, 0.022694041952490807, 5, -128, 127, torch.int8);  x = None
    dequantize_per_tensor_default_2 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_2, 0.022694041952490807, 5, -128, 127, torch.int8);  quantize_per_tensor_default_2 = None
    conv2d = torch.ops.aten.conv2d.default(dequantize_per_tensor_default_2, dequantize_per_tensor_default, conv1_bias);  dequantize_per_tensor_default_2 = dequantize_per_tensor_default = conv1_bias = None
    relu = torch.ops.aten.relu.default(conv2d);  conv2d = None
    quantize_per_tensor_default_3 = torch.ops.quantized_decomposed.quantize_per_tensor.default(relu, 0.013693835586309433, -128, -128, 127, torch.int8);  relu = None
    dequantize_per_tensor_default_3 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_3, 0.013693835586309433, -128, -128, 127, torch.int8);  quantize_per_tensor_default_3 = None
    max_pool2d = torch.ops.aten.max_pool2d.default(dequantize_per_tensor_default_3, [28, 28], [28, 28]);  dequantize_per_tensor_default_3 = None
    quantize_per_tensor_default_4 = torch.ops.quantized_decomposed.quantize_per_tensor.default(max_pool2d, 0.013693835586309433, -128, -128, 127, torch.int8);  max_pool2d = None
    dequantize_per_tensor_default_4 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_4, 0.013693835586309433, -128, -128, 127, torch.int8);  quantize_per_tensor_default_4 = None
    squeeze = torch.ops.aten.squeeze.dim(dequantize_per_tensor_default_4, 2);  dequantize_per_tensor_default_4 = None
    quantize_per_tensor_default_5 = torch.ops.quantized_decomposed.quantize_per_tensor.default(squeeze, 0.013693835586309433, -128, -128, 127, torch.int8);  squeeze = None
    dequantize_per_tensor_default_5 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_5, 0.013693835586309433, -128, -128, 127, torch.int8);  quantize_per_tensor_default_5 = None
    squeeze_1 = torch.ops.aten.squeeze.dim(dequantize_per_tensor_default_5, 2);  dequantize_per_tensor_default_5 = None
    quantize_per_tensor_default_6 = torch.ops.quantized_decomposed.quantize_per_tensor.default(squeeze_1, 0.013693835586309433, -128, -128, 127, torch.int8);  squeeze_1 = None
    dequantize_per_tensor_default_6 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_6, 0.013693835586309433, -128, -128, 127, torch.int8);  quantize_per_tensor_default_6 = None
    linear = torch.ops.aten.linear.default(dequantize_per_tensor_default_6, dequantize_per_tensor_default_1, fc_bias);  dequantize_per_tensor_default_6 = dequantize_per_tensor_default_1 = fc_bias = None
    quantize_per_tensor_default_7 = torch.ops.quantized_decomposed.quantize_per_tensor.default(linear, 0.005307410378009081, -67, -128, 127, torch.int8);  linear = None
    dequantize_per_tensor_default_7 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_7, 0.005307410378009081, -67, -128, 127, torch.int8);  quantize_per_tensor_default_7 = None
    return pytree.tree_unflatten((dequantize_per_tensor_default_7,), self._out_spec)

# To see more debug info, please use `graph_module.print_readable()`
'''
```

As we can see, when preparing the observers, the inserted nodes are not lowered to aten operations just yet, since they still could be used for QAT training (which need modules).

Then, the data indicating which nodes were annotated for quantization and how are saved in the meta attribute as a dictionary.

Finally, when quantized, the observers are now lowered to actual quantize and dequantize nodes.

Notice too that when preparing the model in eval mode, the batch normalization nodes are fused without us needing to specify so. However, when preparing for QAT, they will remain.

##### Custom FX BackendConfigs based on PT2E code

Since the FX Graph mode has been declared as "maintenance only", it will not get further updated since the pytorch team began developing the PT2E mode.

However, some functions are useful in FX Graph mode that are not fully yet implemented in PT2E, so I decided to create some custom BackendConfigs on my own.

###### Custom FX BackendConfigs based on PT2E code: AIEdgeTorch

[AI Edge Torch](https://github.com/google-ai-edge/ai-edge-torch/) is a newer tool made to convert pytorch models to a [TFLite](https://ai.google.dev/edge/litert) backend using PT2E.

It is extremely new in development (the current latest version is 0.2.0 at the time of writing), and not all models can be converted to PT2E during training (although they can be refactored to be exported to PT2E post-training), so I decided to write a new backend configuration for FX model training, followed by a later conversion to PT2E

Based on [`ai_edge_torch.quantize.pt2e_quantizer.PT2EQuantizer`](https://github.com/google-ai-edge/ai-edge-torch/blob/main/ai_edge_torch/quantize/pt2e_quantizer.py)


```python
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

```

###### Custom FX BackendConfigs based on 3rd party code: OpenVino

Based on [this outdated pytorch version implementation of openvino backend config](https://github.com/open-mmlab/mmrazor/blob/main/mmrazor/structures/quantization/backend_config/openvino.py) from mmrazor.

```python
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

```


#### Quantization Methods and model refactoring

In progress...

##### Eager Mode

In progress...

###### Eager mode refactoring for execution

In progress...

##### FX Symbolic Tracing

In progress...

###### FX Symbolic Tracing Refactoring for traceability

In progress...

##### PT2E

In progress...

###### torch.export and TorchDynamo/torch.compile traceability refactoring

In progress...

#### FX Graph mode: Backend specific post processing

##### ONNXRuntime split quantization parameters

There is [onnx](https://github.com/onnx/onnx) exporting functionality in pytorch ([`torch.onnx.export`](https://pytorch.org/docs/stable/onnx.html#torchscript-based-onnx-exporter) for Eager and FX Graph modes, and [`torch.onnx.dynamo_export`](https://pytorch.org/docs/stable/onnx_dynamo.html) for dynamo and PT2E models).

This is usually done as a middle step for later conversion to another backend that is compatible with onnx but not directly with pytorch.

However, there is also the option to run these models in [ONNXRuntime](https://github.com/microsoft/onnxruntime/), which is also capable of inference with quantized models, with a variety of [execution providers](https://onnxruntime.ai/docs/execution-providers/) for the specific hardware / device that the model will be executed on.

While most of the specifications of the ONNXRuntime backend can be similar to the pytorch native backend, there are some variations.

An example not provided in the source documentation is the [ONNXRuntime quantized split node](https://github.com/microsoft/onnxruntime/blob/ee6a91533cc8034a78b183cc1d6595170a366886/onnxruntime/python/tools/quantization/operators/split.py#L63), which has specifications that the previous node and all children of the split node should have the same quantization parameters. However, the [pytorch native backend configuration](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/backend_config/native.py), which imports from the [`_common_operator_config_utils`](https://github.com/pytorch/pytorch/blob/8225e7706ebe900593ccb6790f51e4251e7de16d/torch/ao/quantization/backend_config/_common_operator_config_utils.py#L606), can only configure shared parameters for single tensor outputs, and not for tuples. 

It is also similarly necessary for tflite model exports.

I personally add a post processing method before exporting to ONNX if I need split nodes to be quantized correctly:

```python
from torch.ao.quantization.backend_config import ObservationType, BackendConfig
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
```

```python
import torch
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.fx.tracer import QuantizationTracer
from torch.fx import GraphModule
from torch.ao.quantization.fx import prepare
from torch.ao.quantization.backend_config import get_native_backend_config

class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1,30,1,1)
        self.spconv1 = torch.nn.Conv2d(15,5,1,1)
        self.spconv2 = torch.nn.Conv2d(15,5,1,1)
        # self.pool = torch.nn.AdaptiveMaxPool2d((1,1))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool = torch.nn.MaxPool2d(28,28)
        self.fc = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.conv(x)
        y,z = torch.split(x,2)
        y = self.spconv1(y)
        z = self.spconv2(z)
        x = torch.cat([y,z], dim=1)
        x = self.pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc(x)
        return x


model = ExampleModel()

# define the qconfig_mapping
qconfig_mapping = QConfigMapping().set_global(
        QConfig(
            activation=MovingAverageMinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                ),
            weight=MovingAveragePerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                ),
            )
        )

# FX trace the model
tracer = QuantizationTracer(skipped_module_names=[], skipped_module_classes=[])
graph = tracer.trace(model)
traced_fx = GraphModule(tracer.root, graph, 'ExampleModel')
print(traced_fx)
'''
ExampleModel(
  (conv): Conv2d(1, 30, kernel_size=(1, 1), stride=(1, 1))
  (spconv1): Conv2d(15, 5, kernel_size=(1, 1), stride=(1, 1))
  (spconv2): Conv2d(15, 5, kernel_size=(1, 1), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc): Linear(in_features=10, out_features=10, bias=True)
)



def forward(self, x):
    conv = self.conv(x);  x = None
    split = torch.functional.split(conv, 2, dim = 0);  conv = None
    getitem = split[0]
    getitem_1 = split[1];  split = None
    spconv1 = self.spconv1(getitem);  getitem = None
    spconv2 = self.spconv2(getitem_1);  getitem_1 = None
    cat = torch.cat([spconv1, spconv2], dim = 1);  spconv1 = spconv2 = None
    pool = self.pool(cat);  cat = None
    squeeze = torch.squeeze(pool, dim = 2);  pool = None
    squeeze_1 = torch.squeeze(squeeze, dim = 2);  squeeze = None
    fc = self.fc(squeeze_1);  squeeze_1 = None
    return fc

# To see more debug info, please use `graph_module.print_readable()`
'''

# FX prepare the quantization nodes
example_inputs = (torch.randn(1,1,28,28),)
backend_config = get_native_backend_config()
prepared_fx = prepare(
    traced_fx,
    qconfig_mapping=qconfig_mapping,
    node_name_to_scope=tracer.node_name_to_scope,
    is_qat=True, # convenient even if not QAT
    example_inputs=example_inputs,
    backend_config=backend_config,
    )
print(prepared_fx)
'''
GraphModule(
  (activation_post_process_0): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (conv): Conv2d(
    1, 30, kernel_size=(1, 1), stride=(1, 1)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_1): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (activation_post_process_2): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (activation_post_process_4): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (spconv1): Conv2d(
    15, 5, kernel_size=(1, 1), stride=(1, 1)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_3): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (spconv2): Conv2d(
    15, 5, kernel_size=(1, 1), stride=(1, 1)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_5): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (activation_post_process_6): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (activation_post_process_7): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (activation_post_process_8): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (activation_post_process_9): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (fc): Linear(
    in_features=10, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_10): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
)



def forward(self, x):
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
    conv = self.conv(activation_post_process_0);  activation_post_process_0 = None
    activation_post_process_1 = self.activation_post_process_1(conv);  conv = None
    split = torch.functional.split(activation_post_process_1, 2, dim = 0);  activation_post_process_1 = None
    getitem = split[0]
    activation_post_process_2 = self.activation_post_process_2(getitem);  getitem = None
    getitem_1 = split[1];  split = None
    activation_post_process_4 = self.activation_post_process_4(getitem_1);  getitem_1 = None
    spconv1 = self.spconv1(activation_post_process_2);  activation_post_process_2 = None
    activation_post_process_3 = self.activation_post_process_3(spconv1);  spconv1 = None
    spconv2 = self.spconv2(activation_post_process_4);  activation_post_process_4 = None
    activation_post_process_5 = self.activation_post_process_5(spconv2);  spconv2 = None
    cat = torch.cat([activation_post_process_3, activation_post_process_5], dim = 1);  activation_post_process_3 = activation_post_process_5 = None
    activation_post_process_6 = self.activation_post_process_6(cat);  cat = None
    pool = self.pool(activation_post_process_6);  activation_post_process_6 = None
    activation_post_process_7 = self.activation_post_process_7(pool);  pool = None
    squeeze = torch.squeeze(activation_post_process_7, dim = 2);  activation_post_process_7 = None
    activation_post_process_8 = self.activation_post_process_8(squeeze);  squeeze = None
    squeeze_1 = torch.squeeze(activation_post_process_8, dim = 2);  activation_post_process_8 = None
    activation_post_process_9 = self.activation_post_process_9(squeeze_1);  squeeze_1 = None
    fc = self.fc(activation_post_process_9);  activation_post_process_9 = None
    activation_post_process_10 = self.activation_post_process_10(fc);  fc = None
    return activation_post_process_10

# To see more debug info, please use `graph_module.print_readable()`
'''

propagate_split_share_qparams_pre_process(prepared_fx, backend_config)
print(prepared_fx)
'''
GraphModule(
  (activation_post_process_0): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (conv): Conv2d(
    1, 30, kernel_size=(1, 1), stride=(1, 1)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_1): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (spconv1): Conv2d(
    15, 5, kernel_size=(1, 1), stride=(1, 1)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_3): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (spconv2): Conv2d(
    15, 5, kernel_size=(1, 1), stride=(1, 1)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_5): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (activation_post_process_6): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (activation_post_process_7): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (activation_post_process_8): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (activation_post_process_9): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (fc): Linear(
    in_features=10, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_10): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
)



def forward(self, x):
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
    conv = self.conv(activation_post_process_0);  activation_post_process_0 = None
    activation_post_process_1 = self.activation_post_process_1(conv);  conv = None
    split = torch.functional.split(activation_post_process_1, 2, dim = 0);  activation_post_process_1 = None
    getitem = split[0]
    activation_post_process_2 = self.activation_post_process_1(getitem);  getitem = None
    getitem_1 = split[1];  split = None
    activation_post_process_4 = self.activation_post_process_1(getitem_1);  getitem_1 = None
    spconv1 = self.spconv1(activation_post_process_2);  activation_post_process_2 = None
    activation_post_process_3 = self.activation_post_process_3(spconv1);  spconv1 = None
    spconv2 = self.spconv2(activation_post_process_4);  activation_post_process_4 = None
    activation_post_process_5 = self.activation_post_process_5(spconv2);  spconv2 = None
    cat = torch.cat([activation_post_process_3, activation_post_process_5], dim = 1);  activation_post_process_3 = activation_post_process_5 = None
    activation_post_process_6 = self.activation_post_process_6(cat);  cat = None
    pool = self.pool(activation_post_process_6);  activation_post_process_6 = None
    activation_post_process_7 = self.activation_post_process_7(pool);  pool = None
    squeeze = torch.squeeze(activation_post_process_7, dim = 2);  activation_post_process_7 = None
    activation_post_process_8 = self.activation_post_process_7(squeeze);  squeeze = None
    squeeze_1 = torch.squeeze(activation_post_process_8, dim = 2);  activation_post_process_8 = None
    activation_post_process_9 = self.activation_post_process_7(squeeze_1);  squeeze_1 = None
    fc = self.fc(activation_post_process_9);  activation_post_process_9 = None
    activation_post_process_10 = self.activation_post_process_10(fc);  fc = None
    return activation_post_process_10

# To see more debug info, please use `graph_module.print_readable()`
'''
```

##### Batch normalization fusing in QAT backend export support

Another example is that QAT fused nodes would be exported in their float mode, without being folded into quantized nodes.

This is a lot more complicated to solve, and needs to have some post processing alterations before the exporting of a model.

```python
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
```

Calling the `to_float` method on the QAT fused modules will fuse the Batch Normalization modules into the actual Convolution weight and bias tensors.

Here's an example of the change this method performs (skipping model definitions). Notice how Batch Normalization layers are inherently fused via the weights:
```python
print(prepared_model)
'''
ExampleModel(
  (quant): QuantStub(
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.4242129623889923, max_val=2.821486711502075)
  )
  (dequant): DeQuantStub()
  (conv1): ConvBnReLU2d(
    1, 32, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([ 0.9805,  0.9878, -0.9778, -0.9910, -0.9607,  1.0020,  0.9854, -0.9932,
               0.9920, -1.0192,  0.9573,  0.9959,  1.0661, -0.9587, -0.4509,  0.8841,
               0.9185,  0.9696, -0.9722, -1.0099,  1.0207, -1.0131,  0.9228,  0.9731,
              -0.8032, -0.9803, -0.9691,  1.0209, -0.9520,  1.0132,  1.0179, -1.0628],
             device='cuda:0'), max_val=tensor([ 0.9805,  0.9878, -0.9778, -0.9910, -0.9607,  1.0020,  0.9854, -0.9932,
               0.9920, -1.0192,  0.9573,  0.9959,  1.0661, -0.9587, -0.4509,  0.8841,
               0.9185,  0.9696, -0.9722, -1.0099,  1.0207, -1.0131,  0.9228,  0.9731,
              -0.8032, -0.9803, -0.9691,  1.0209, -0.9520,  1.0132,  1.0179, -1.0628],
             device='cuda:0')
    )
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.486288547515869)
  )
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvBnReLU2d(
    32, 64, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-2.1960, -0.2812, -1.7402, -0.8100, -3.3399, -1.0657, -0.1177, -0.6993,
              -0.2671, -4.9234, -0.5415, -3.2926, -0.1593, -0.3427, -0.4231, -3.2403,
              -0.7386, -2.6828, -5.1171, -3.6965, -0.2139, -9.6442, -0.3108, -2.1643,
              -0.7892, -2.2819, -2.0034, -0.4834, -0.3995, -4.6650, -1.5611, -1.3696,
              -3.9522, -0.3022, -1.5632, -0.4557, -5.8931, -0.4400, -1.2626, -0.5098,
              -3.3187, -0.3899, -0.4554, -2.8338, -0.4487, -0.2008, -1.1349, -5.3991,
              -4.4046, -0.4110, -1.2552, -0.5631, -0.3380, -2.7315, -2.2920, -2.1396,
              -0.4084, -6.2974, -1.1824, -0.1679, -2.1181, -0.8331, -1.1392, -5.4736],
             device='cuda:0'), max_val=tensor([2.1027, 0.2706, 2.1971, 0.7181, 2.7690, 0.9632, 0.1437, 0.6662, 0.2961,
              3.6605, 0.5448, 2.8305, 0.1597, 0.3670, 0.4135, 4.4315, 0.7591, 3.2193,
              4.7957, 4.5160, 0.2183, 8.5344, 0.3707, 2.4349, 1.2237, 2.4649, 2.5588,
              0.4111, 0.5481, 5.3878, 1.8818, 0.9792, 3.3811, 0.3097, 0.9306, 0.5589,
              5.2913, 0.3896, 0.8146, 0.8410, 2.5406, 0.2423, 0.4662, 3.2510, 0.4216,
              0.2331, 0.7295, 5.5472, 3.2039, 0.3767, 1.2058, 0.4847, 0.2246, 2.3556,
              2.5913, 1.7738, 0.4303, 6.8638, 0.9290, 0.2386, 2.2515, 1.1236, 0.7875,
              4.1441], device='cuda:0')
    )
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=7.873152732849121)
  )
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    in_features=3136, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.1110, -0.1682, -0.1074, -0.0716, -0.1544, -0.1686, -0.1512, -0.2297,
              -0.1243, -0.1656], device='cuda:0'), max_val=tensor([0.1568, 0.1229, 0.1314, 0.0929, 0.1936, 0.1594, 0.1684, 0.1468, 0.1751,
              0.1768], device='cuda:0')
    )
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=52.048946380615234)
  )
  (fc1act): Identity()
  (fc2): Linear(
    in_features=10, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.4732, -0.7787, -0.3168, -0.1518, -0.6355, -0.3758, -0.5703, -0.3268,
              -0.3168, -0.5046], device='cuda:0'), max_val=tensor([0.3203, 0.4310, 0.3799, 0.3980, 0.3096, 0.2491, 0.4260, 0.2836, 0.3210,
              0.3214], device='cuda:0')
    )
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-21.06275177001953, max_val=15.43133544921875)
  )
)
'''

qconfig = QConfig(
    activation=MovingAverageMinMaxObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        ),
    weight=MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ),
    )
device='cuda:0'

fuse_qat_bn_post_process(
    prepared_model,
    qconfig,
    device,
    update_weight_with_fakequant=False,
    keep_w_fake_quant=True)
print(prepared_model)
'''
ExampleModel(
  (quant): QuantStub(
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.4242129623889923, max_val=2.821486711502075)
  )
  (dequant): DeQuantStub()
  (conv1): ConvReLU2d(
    1, 32, kernel_size=(1, 1), stride=(1, 1)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([ 0.9773,  0.9846, -0.9746, -0.9878, -0.9576,  0.9987,  0.9822, -0.9899,
               0.9888, -1.0159,  0.9542,  0.9927,  1.0627, -0.9556, -0.4495,  0.8815,
               0.9156,  0.9664, -0.9691, -1.0067,  1.0174, -1.0098,  0.9199,  0.9699,
              -0.8006, -0.9771, -0.9660,  1.0176, -0.9489,  1.0099,  1.0146, -1.0593],
             device='cuda:0'), max_val=tensor([ 0.9773,  0.9846, -0.9746, -0.9878, -0.9576,  0.9987,  0.9822, -0.9899,
               0.9888, -1.0159,  0.9542,  0.9927,  1.0627, -0.9556, -0.4495,  0.8815,
               0.9156,  0.9664, -0.9691, -1.0067,  1.0174, -1.0098,  0.9199,  0.9699,
              -0.8006, -0.9771, -0.9660,  1.0176, -0.9489,  1.0099,  1.0146, -1.0593],
             device='cuda:0')
    )
  )
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvReLU2d(
    32, 64, kernel_size=(1, 1), stride=(1, 1)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-2.2004, -0.2815, -1.7259, -0.8108, -3.3225, -1.0663, -0.1178, -0.6998,
              -0.2673, -4.9116, -0.5416, -3.3015, -0.1594, -0.3430, -0.4228, -3.2384,
              -0.7388, -2.6793, -5.0133, -3.6836, -0.2141, -9.6123, -0.3111, -2.1644,
              -0.7847, -2.2629, -1.9944, -0.4836, -0.3984, -4.6541, -1.5561, -1.3597,
              -3.9317, -0.3023, -1.5552, -0.4559, -5.8850, -0.4401, -1.2523, -0.5099,
              -3.3166, -0.3899, -0.4554, -2.8402, -0.4490, -0.2010, -1.1246, -5.3832,
              -4.3736, -0.4115, -1.2560, -0.5640, -0.3383, -2.7325, -2.2894, -2.1422,
              -0.4086, -6.1985, -1.1782, -0.1680, -2.1212, -0.8303, -1.1334, -5.4791],
             device='cuda:0'), max_val=tensor([2.1069, 0.2708, 2.1790, 0.7187, 2.7546, 0.9638, 0.1438, 0.6667, 0.2964,
              3.6516, 0.5449, 2.8382, 0.1598, 0.3673, 0.4133, 4.4289, 0.7594, 3.2151,
              4.6985, 4.5002, 0.2185, 8.5062, 0.3711, 2.4350, 1.2167, 2.4443, 2.5474,
              0.4113, 0.5465, 5.3751, 1.8757, 0.9722, 3.3635, 0.3099, 0.9258, 0.5592,
              5.2839, 0.3897, 0.8079, 0.8413, 2.5389, 0.2423, 0.4662, 3.2584, 0.4219,
              0.2332, 0.7228, 5.5310, 3.1814, 0.3771, 1.2065, 0.4855, 0.2248, 2.3564,
              2.5883, 1.7760, 0.4306, 6.7560, 0.9257, 0.2388, 2.2547, 1.1199, 0.7835,
              4.1483], device='cuda:0')
    )
  )
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    in_features=3136, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.1110, -0.1682, -0.1074, -0.0716, -0.1544, -0.1686, -0.1512, -0.2297,
              -0.1243, -0.1656], device='cuda:0'), max_val=tensor([0.1568, 0.1229, 0.1314, 0.0929, 0.1936, 0.1594, 0.1684, 0.1468, 0.1751,
              0.1768], device='cuda:0')
    )
  )
  (fc1act): Identity()
  (fc2): Linear(
    in_features=10, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.4732, -0.7787, -0.3168, -0.1518, -0.6355, -0.3758, -0.5703, -0.3268,
              -0.3168, -0.5046], device='cuda:0'), max_val=tensor([0.3203, 0.4310, 0.3799, 0.3980, 0.3096, 0.2491, 0.4260, 0.2836, 0.3210,
              0.3214], device='cuda:0')
    )
  )
)
'''
```

##### Merging ReLU and clamp into quantization parameters for QAT backend export support

Some backend configurations will not recognize a ReLU following operations that are not supported as fused modules. Other backend configurations will actually add ReLU as one of the operations that share the previous node's parameters, which is wasteful of the quantization range.

In many of the backends themselves there is support for PTQ models to merge ReLU or clamp into the quantization parameters of the output observer.

However, during QAT, this might not be implemented correctly by pytorch, which is why I made a post processing function to handle this in the case of FX mode.

This is possible because quantization itself only allows a range from the quantization minimum to the maximum, so clamping is necessary. All we need to do is adjust the scaling to the clamped output. ReLU is also a sort of clamp with the range `0,inf` for floating point values. 

Now, if the ReLU or clamp happens before any quantization observer is added, then we can just delete that operation after training or calibration of the quantization parameters, since the scale and zero point should match that range.

However if there is a shared parameters quantization on ReLU and clamp nodes, as is the case in backends that use the [common shared quantization parameters ops list](https://github.com/pytorch/pytorch/blob/e80f47fb4d1556257f75be3859231eaffa1c8acc/torch/ao/quantization/backend_config/_common_operator_config_utils.py#L606C1-L683C81), such as the [native](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/backend_config/native.py) backend if we try to use it for ONNXRuntime, the following will happen, for example, in asymmetric quantization:

```python
# asymmetric

x = tensor(dtype=float32)

quant_min = 0   # or -128 or -127
quant_max = 255 # or 127

quant_x = round((x / scale) + zero_point).clamp(quant_min, quant_max)
scale = (max(x) - min(x)) / (quant_max - quant_min)

# Solving from formulas:
quant_min = round((min(x) / scale) + zero_point)
zero_point = round(quant_min - (min(x)/scale))
# or
quant_max = round((max(x) / scale) + zero_point)
zero_point = round(quant_max - (max(x)/scale))

z = zero_point

min(x)-----------0-------------max(x)
    ??            ??             ??
    quant_min----z----quant_max


dequant_x = (quant_x - zero_point)*scale


            quant_min----z----quant_max
            ??            ??             ??
approx_min_x-------------0-------------approx_max_x

approx_min_x ~= min(x)
approx_max_x ~= max(x)

# clamp_min would be 0 for relu
relu_clamp_out = dequant_x.clamp(clamp_min, clamp_max)
quant_relu_clamp_out = round((relu_clamp_out / scale) + zero_point).clamp(quant_min, quant_max)

min(dequant_x)-----clamp_min--0---clamp_max----max(dequant_x)
            ??           ??     ??     ??          ??
            quant_min---------z--------quant_max

dequant_relu_clamp_out = (quant_x - zero_point)*scale

            quant_min--------------z--------------quant_max
                   ??       ??       ??       ??       ??
approx_min_dequant_x---clamp_min---0---clamp_max---approx_max_dequant_x

```

Or the symmetric example:

```python
# symmetric

x = tensor(dtype=float32)

quant_min = -128 # asymmetric range or -127 for symmetric range
quant_max = 127
quant_x = round((x / scale)).clamp(quant_min, quant_max)
scale = (2*(max(abs(x)))) / (quant_max - quant_min)
zero_point = 0

-max(abs(x))-----------0-------------max(abs(x))
          ??            ??             ??
          quant_min----0----quant_max

dequant_x = quant_x*scale

                 quant_min----0----quant_max
                 ??            ??             ??
-approx_max_abs_x-------------0-------------approx_max_abs_x

approx_max_abs_x ~= max(abs(x))

# clamp_min would be 0 for relu
relu_clamp_out = dequant_x.clamp(clamp_min, clamp_max)
quant_relu_clamp_out = round((relu_clamp_out / scale)).clamp(quant_min, quant_max)

-max(abs(dequant_x))----clamp_min--0---clamp_max----max(abs(dequant_x))
                   ??         ??     ??     ??          ??
                   quant_min-------0--------quant_max

dequant_relu_clamp_out = (quant_x - zero_point)*scale

            quant_min--------------0--------------quant_max
                   ??       ??       ??       ??       ??
approx_min_dequant_x---clamp_min---0---clamp_max---approx_max_dequant_x
```

This is wasteful of the quantization range and can cause less precise results. We can instead do this:


```python
# asymmetric clamp merge

x = tensor(dtype=float32)

quant_min = 0   # or -128 or -127
quant_max = 255 # or 127

new_max = min(clamp_max, max(x))
new_min = max(clamp_min, min(x))

quant_clamp_x = round((x / new_scale) + new_zero_point).clamp(quant_min, quant_max)
new_scale = (new_max - new_min) / (quant_max - quant_min)

# Solving from formulas:
quant_min = round((new_min / scale) + new_zero_point)
new_zero_point = round(quant_min - (new_min/new_scale))
# or
quant_max = round((new_max / scale) + new_zero_point)
new_zero_point = round(quant_max - (new_max/new_scale))

z = new_zero_point

new_min-----------0-------------new_max
    ??            ??             ??
    quant_min----z----quant_max


dequant_clamp_x = (quant_clamp_x - new_zero_point)*new_scale


            quant_min----z----quant_max
            ??            ??             ??
approx_min_clamp_x-------0------approx_max_clamp_x

approx_min_clamp_x ~= new_min
approx_max_clamp_x ~= new_max
```

Or the symmetric clamp merge:

```python
# symmetric clamp merge 

x = tensor(dtype=float32)

quant_min = -128 # asymmetric range or -127 for symmetric range
quant_max = 127

old_max = max(abs(x))
old_min = -old_max
asymmetric_new_max = min(clamp_max, old_max)
asymmetric_new_min = max(clamp_min, old_min)
symmetric_new_max = max(abs(asymmetric_new_max),abs(asymmetric_new_min))
symmetric_new_min = -symmetric_new_max

quant_clamp_x = round((x / new_scale)).clamp(quant_min, quant_max)
new_scale = (2*symmetric_new_max) / (quant_max - quant_min)
zero_point = 0

-symmetric_new_max-----0-----symmetric_new_max
          ??            ??             ??
          quant_min----0----quant_max

dequant_clamp_x = quant_clamp_x*new_scale

                 quant_min----0----quant_max
                 ??            ??             ??
-approx_max_abs_clamp_x-------0-------approx_max_abs_clamp_x

approx_max_abs_clamp_x ~= max(abs(x))

```

However, ideally, we should make a backend configuration that doesn't share quantization parameters in the first place. The reason for that is that the weights would 

For example, we might export the native backend model to onnx. If we have the chance to prepare the model before training, we should alter the backend config:

```python
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

```

```python

from typing import Any
import torch
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.fx.tracer import QuantizationTracer
from torch.fx import GraphModule
from torch.ao.quantization.fx import prepare
from torch.ao.quantization.backend_config import get_native_backend_config


class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.conv2 = torch.nn.Conv2d(32,10,1,1)
        # self.pool = torch.nn.AdaptiveMaxPool2d((1,1))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool = torch.nn.MaxPool2d(28,28)
        self.fc = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        x = self.fc(x)
        return x

model = ExampleModel()

# define the qconfig_mapping
qconfig_mapping = QConfigMapping().set_global(
        QConfig(
            activation=MovingAverageMinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                ),
            weight=MovingAveragePerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                ),
            )
        )

# FX trace the model
tracer = QuantizationTracer(skipped_module_names=[], skipped_module_classes=[])
graph = tracer.trace(model)
traced_fx = GraphModule(tracer.root, graph, 'ExampleModel')
print(traced_fx)
'''
ExampleModel(
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (conv2): Conv2d(32, 10, kernel_size=(1, 1), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (fc): Linear(in_features=10, out_features=10, bias=True)
)



def forward(self, x):
    conv1 = self.conv1(x);  x = None
    relu = torch.nn.functional.relu(conv1, inplace = False);  conv1 = None
    conv2 = self.conv2(relu);  relu = None
    relu_1 = torch.nn.functional.relu(conv2, inplace = False);  conv2 = None
    pool = self.pool(relu_1);  relu_1 = None
    squeeze = torch.squeeze(pool, dim = 2);  pool = None
    squeeze_1 = torch.squeeze(squeeze, dim = 2);  squeeze = None
    fc = self.fc(squeeze_1);  squeeze_1 = None
    return fc

# To see more debug info, please use `graph_module.print_readable()`
'''

# Don't fuse just to prove what happens to nodes that can't fuse with ReLU
# FX prepare the quantization nodes
example_inputs = (torch.randn(1,1,28,28),)
backend_config = get_native_backend_config()
backend_config = relu_clamp_backend_config_unshare_observers(backend_config)

prepared_fx = prepare(
    traced_fx,
    qconfig_mapping=qconfig_mapping,
    node_name_to_scope=tracer.node_name_to_scope,
    is_qat=True, # convenient even if not QAT
    example_inputs=example_inputs,
    backend_config=backend_config,
    )
# pass some input to see the observed scales
prepared_fx(example_inputs[0])
print(prepared_fx)
'''
GraphModule(
  (activation_post_process_0): MovingAverageMinMaxObserver(min_val=-3.358337879180908, max_val=3.094209671020508)
  (conv1): Conv2d(
    1, 32, kernel_size=(1, 1), stride=(1, 1)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([ 0.3141,  0.8659, -0.3041, -0.4085, -0.3968,  0.2745,  0.3924,  0.6181,
              -0.6419,  0.3388, -0.6847,  0.6948, -0.6194,  0.5268,  0.5759, -0.2281,
               0.0610,  0.7425, -0.1603,  0.7990, -0.9284,  0.3517,  0.5457, -0.3669,
               0.5580, -0.4137, -0.4218, -0.9705, -0.5250, -0.5045,  0.4345, -0.2527]), max_val=tensor([ 0.3141,  0.8659, -0.3041, -0.4085, -0.3968,  0.2745,  0.3924,  0.6181,
              -0.6419,  0.3388, -0.6847,  0.6948, -0.6194,  0.5268,  0.5759, -0.2281,
               0.0610,  0.7425, -0.1603,  0.7990, -0.9284,  0.3517,  0.5457, -0.3669,
               0.5580, -0.4137, -0.4218, -0.9705, -0.5250, -0.5045,  0.4345, -0.2527])
    )
  )
  (activation_post_process_1): MovingAverageMinMaxObserver(min_val=-3.481182813644409, max_val=3.503840446472168)
  (activation_post_process_2): MovingAverageMinMaxObserver(min_val=-3.481182813644409, max_val=3.503840446472168)
  (conv2): Conv2d(
    32, 10, kernel_size=(1, 1), stride=(1, 1)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.1766, -0.1619, -0.1684, -0.1532, -0.1666, -0.1548, -0.1617, -0.1751,
              -0.1721, -0.1630]), max_val=tensor([0.1635, 0.1671, 0.1758, 0.1766, 0.1646, 0.1757, 0.1751, 0.1609, 0.1639,
              0.1756])
    )
  )
  (activation_post_process_3): MovingAverageMinMaxObserver(min_val=-1.0726615190505981, max_val=1.4624204635620117)
  (activation_post_process_4): MovingAverageMinMaxObserver(min_val=-1.0726615190505981, max_val=1.4624204635620117)
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (activation_post_process_5): MovingAverageMinMaxObserver(min_val=-1.0726615190505981, max_val=1.4624204635620117)
  (activation_post_process_6): MovingAverageMinMaxObserver(min_val=-1.0726615190505981, max_val=1.4624204635620117)
  (activation_post_process_7): MovingAverageMinMaxObserver(min_val=-1.0726615190505981, max_val=1.4624204635620117)
  (fc): Linear(
    in_features=10, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.3113, -0.1477, -0.3055, -0.2916, -0.2433, -0.3093, -0.3096, -0.2451,
              -0.3053, -0.2737]), max_val=tensor([0.3001, 0.2993, 0.2062, 0.2732, 0.1871, 0.2586, 0.2617, 0.2384, 0.1637,
              0.3034])
    )
  )
  (activation_post_process_8): MovingAverageMinMaxObserver(min_val=-0.4853910207748413, max_val=0.9318119287490845)
)



def forward(self, x):
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
    conv1 = self.conv1(activation_post_process_0);  activation_post_process_0 = None
    activation_post_process_1 = self.activation_post_process_1(conv1);  conv1 = None
    relu = torch.nn.functional.relu(activation_post_process_1, inplace = False);  activation_post_process_1 = None
    activation_post_process_2 = self.activation_post_process_2(relu);  relu = None
    conv2 = self.conv2(activation_post_process_2);  activation_post_process_2 = None
    activation_post_process_3 = self.activation_post_process_3(conv2);  conv2 = None
    relu_1 = torch.nn.functional.relu(activation_post_process_3, inplace = False);  activation_post_process_3 = None
    activation_post_process_4 = self.activation_post_process_4(relu_1);  relu_1 = None
    pool = self.pool(activation_post_process_4);  activation_post_process_4 = None
    activation_post_process_5 = self.activation_post_process_5(pool);  pool = None
    squeeze = torch.squeeze(activation_post_process_5, dim = 2);  activation_post_process_5 = None
    activation_post_process_6 = self.activation_post_process_6(squeeze);  squeeze = None
    squeeze_1 = torch.squeeze(activation_post_process_6, dim = 2);  activation_post_process_6 = None
    activation_post_process_7 = self.activation_post_process_7(squeeze_1);  squeeze_1 = None
    fc = self.fc(activation_post_process_7);  activation_post_process_7 = None
    activation_post_process_8 = self.activation_post_process_8(fc);  fc = None
    return activation_post_process_8

# To see more debug info, please use `graph_module.print_readable()`
'''
```

As we can see, the `min_val` for the `activation_post_process_1` and `activation_post_process_2`, as well as `activation_post_process_3` and `activation_post_process_4`, and so on on nodes followed by unfused relu with their own observer are different after passing some example input to the observers, which means we can just eliminate the previous activation and nodes if we no longer need to train or calibrate a model. Assuming we had done that:


```python
import torch
from torch.ao.quantization.fx.graph_module import ObservedGraphModule


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


```

```python

merge_relu_clamp_to_qparams_post_process(prepared_fx)
print(prepared_fx)
'''
GraphModule(
  (activation_post_process_0): MovingAverageMinMaxObserver(min_val=-3.358337879180908, max_val=3.094209671020508)
  (conv1): Conv2d(
    1, 32, kernel_size=(1, 1), stride=(1, 1)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([ 0.3141,  0.8659, -0.3041, -0.4085, -0.3968,  0.2745,  0.3924,  0.6181,
              -0.6419,  0.3388, -0.6847,  0.6948, -0.6194,  0.5268,  0.5759, -0.2281,
               0.0610,  0.7425, -0.1603,  0.7990, -0.9284,  0.3517,  0.5457, -0.3669,
               0.5580, -0.4137, -0.4218, -0.9705, -0.5250, -0.5045,  0.4345, -0.2527]), max_val=tensor([ 0.3141,  0.8659, -0.3041, -0.4085, -0.3968,  0.2745,  0.3924,  0.6181,
              -0.6419,  0.3388, -0.6847,  0.6948, -0.6194,  0.5268,  0.5759, -0.2281,
               0.0610,  0.7425, -0.1603,  0.7990, -0.9284,  0.3517,  0.5457, -0.3669,
               0.5580, -0.4137, -0.4218, -0.9705, -0.5250, -0.5045,  0.4345, -0.2527])
    )
  )
  (activation_post_process_1): MovingAverageMinMaxObserver(min_val=-3.481182813644409, max_val=3.503840446472168)
  (activation_post_process_2): MovingAverageMinMaxObserver(min_val=-3.481182813644409, max_val=3.503840446472168)
  (conv2): Conv2d(
    32, 10, kernel_size=(1, 1), stride=(1, 1)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.1766, -0.1619, -0.1684, -0.1532, -0.1666, -0.1548, -0.1617, -0.1751,
              -0.1721, -0.1630]), max_val=tensor([0.1635, 0.1671, 0.1758, 0.1766, 0.1646, 0.1757, 0.1751, 0.1609, 0.1639,
              0.1756])
    )
  )
  (activation_post_process_3): MovingAverageMinMaxObserver(min_val=-1.0726615190505981, max_val=1.4624204635620117)
  (activation_post_process_4): MovingAverageMinMaxObserver(min_val=-1.0726615190505981, max_val=1.4624204635620117)
  (pool): MaxPool2d(kernel_size=28, stride=28, padding=0, dilation=1, ceil_mode=False)
  (activation_post_process_5): MovingAverageMinMaxObserver(min_val=-1.0726615190505981, max_val=1.4624204635620117)
  (activation_post_process_6): MovingAverageMinMaxObserver(min_val=-1.0726615190505981, max_val=1.4624204635620117)
  (activation_post_process_7): MovingAverageMinMaxObserver(min_val=-1.0726615190505981, max_val=1.4624204635620117)
  (fc): Linear(
    in_features=10, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.3113, -0.1477, -0.3055, -0.2916, -0.2433, -0.3093, -0.3096, -0.2451,
              -0.3053, -0.2737]), max_val=tensor([0.3001, 0.2993, 0.2062, 0.2732, 0.1871, 0.2586, 0.2617, 0.2384, 0.1637,
              0.3034])
    )
  )
  (activation_post_process_8): MovingAverageMinMaxObserver(min_val=-0.4853910207748413, max_val=0.9318119287490845)
)



def forward(self, x):
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
    conv1 = self.conv1(activation_post_process_0);  activation_post_process_0 = None
    activation_post_process_1 = self.activation_post_process_1(conv1);  conv1 = None
    relu = torch.nn.functional.relu(activation_post_process_1, inplace = False);  activation_post_process_1 = None
    activation_post_process_2 = self.activation_post_process_2(relu);  relu = None
    conv2 = self.conv2(activation_post_process_2);  activation_post_process_2 = None
    activation_post_process_3 = self.activation_post_process_3(conv2);  conv2 = None
    relu_1 = torch.nn.functional.relu(activation_post_process_3, inplace = False);  activation_post_process_3 = None
    activation_post_process_4 = self.activation_post_process_4(relu_1);  relu_1 = None
    pool = self.pool(activation_post_process_4);  activation_post_process_4 = None
    activation_post_process_5 = self.activation_post_process_5(pool);  pool = None
    squeeze = torch.squeeze(activation_post_process_5, dim = 2);  activation_post_process_5 = None
    activation_post_process_6 = self.activation_post_process_6(squeeze);  squeeze = None
    squeeze_1 = torch.squeeze(activation_post_process_6, dim = 2);  activation_post_process_6 = None
    activation_post_process_7 = self.activation_post_process_7(squeeze_1);  squeeze_1 = None
    fc = self.fc(activation_post_process_7);  activation_post_process_7 = None
    activation_post_process_8 = self.activation_post_process_8(fc);  fc = None
    return activation_post_process_8

# To see more debug info, please use `graph_module.print_readable()`
'''

```

While convolutions and linear layers usually can be fused without resorting to this method, other unfusable layers can benefit from this process.


#### Quantization complete examples

In this section, I will show actual example models training and being quantized in the following ways:

Model mode    | PTQ static                                   | PTQ dynamic                                   | QAT static                                   |
--------------|----------------------------------------------|-----------------------------------------------|----------------------------------------------|
Eager mode    | [example](#eager-mode-ptq-static-example)    | [example](#eager-mode-ptq-dynamic-example)    | [example](#eager-mode-qat-static-example)    |
FX Graph mode | [example](#fx-graph-mode-ptq-static-example) | [example](#fx-graph-mode-ptq-dynamic-example) | [example](#fx-graph-mode-qat-static-example) |
PT2E mode     | [example](#pt2e-mode-ptq-static-example)     | [example](#pt2e-mode-ptq-dynamic-example)     | [example](#pt2e-mode-qat-static-example)     |


##### Eager mode PTQ static example

Sources:

- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [Pytorch Eager Mode PTQ Static Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)


```python
# model imports
import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

# data imports
import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transformsv2
from torch.utils.data import DataLoader

## Define training and testing cycles
def train(
        model,
        device,
        train_loader,
        optimizer,
        criterion,
        epoch,
        log_interval=1,
        dry_run=False,
    ):
    model.train()
    n_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            item = batch_idx * len(data)
            item_percent = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{item}/{n_data} ({item_percent:.0f}%)]\tLoss: {loss.item():.6f}')
            if dry_run:
                break


def test(
        model,
        device,
        test_loader,
        criterion,
    ):
    model.eval()
    test_loss = 0
    correct = 0
    n_data = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # sum up batch loss
            pred = torch.argmax(output,dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= n_data
    test_percent = 100. * correct / n_data
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n_data} ({test_percent:.0f}%)\n')


# test a few batches for calibration
def calibrate_ptq(
        prepared_model,
        device,
        test_loader,
        criterion,
        batch_size,
        n_batches=32,
    ):
    prepared_model.eval()
    cal_loss = 0
    correct = 0
    n_data = (len(test_loader.dataset) / batch_size) * n_batches
    with torch.no_grad():
        for (i,(data, target)) in enumerate(test_loader):
            if i<=n_batches:
                data, target = data.to(device), target.to(device)
                output = prepared_model(data)
                cal_loss += criterion(output, target).item()
                # sum up batch loss
                pred = torch.argmax(output,dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                break
    cal_loss /= n_data
    cal_percent = 100. * correct / n_data
    print(f'\nCalibrate set: Average loss: {cal_loss:.4f}, Accuracy: {correct}/{n_data} ({cal_percent:.0f}%)\n')


class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool1 = torch.nn.AdaptiveMaxPool2d((14,14))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(32,64,1,1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.act2 = torch.nn.ReLU()
        # self.pool2 = torch.nn.AdaptiveMaxPool2d((7,7))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool2 = torch.nn.MaxPool2d(2,2)
        self.fl1 = torch.nn.Flatten(start_dim=1)
        self.fc1 = torch.nn.Linear(3136, 10)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x
    def fuse_model(self):
        torch.ao.quantization.fuse_modules(
            self,
            ['conv1','bn1','act1'],
            inplace=True,
            )
        torch.ao.quantization.fuse_modules(
            self,
            ['conv2','bn2','act2'],
            inplace=True,
            )
        torch.ao.quantization.fuse_modules(
            self,
            ['fc1','fc1act'],
            inplace=True,
            )


qconfig = QConfig(
    activation=MovingAverageMinMaxObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        ),
    weight=MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ),
    )

model = ExampleModel()
model.qconfig = qconfig
print(model)
'''
ExampleModel(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)
'''

## Load datasets
batch_size = 64
transform=transformsv2.Compose([
        transformsv2.ToTensor(),
        transformsv2.Normalize((0.1307,), (0.3081,)),
        ])
dataset1 = datasets.MNIST(
    '../data',
    train=True,
    download=True,
    transform=transform,
    )
dataset2 = datasets.MNIST(
    '../data',
    train=False,
    transform=transform,
    )
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)


device = 'cuda:0'
model.to(device)
lr = 0.9
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
max_epochs = 20
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=int(((len(train_loader)-1)//batch_size + 1)),
    epochs=max_epochs,
    cycle_momentum=False)
log_interval = 60

# train float model
for epoch in range(1, max_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch, log_interval)
    test(model, device, test_loader, criterion)
    scheduler.step()

# Test set: Average loss: 0.0053, Accuracy: 8973/10000 (90%)
# not the best model at all, but an example of the process


# fusion is only for eval
model.eval()
model.fuse_model()
print(model)
'''
ExampleModel(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (conv1): ConvReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
  )
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvReLU2d(
    (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
  )
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    (0): Linear(in_features=3136, out_features=10, bias=True)
    (1): ReLU()
  )
  (fc1act): Identity()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)
'''

# prepare model with observers
prepared_model = torch.ao.quantization.prepare(model, inplace=False)
print(prepared_model)
'''
ExampleModel(
  (quant): QuantStub(
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (dequant): DeQuantStub()
  (conv1): ConvReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvReLU2d(
    (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    (0): Linear(in_features=3136, out_features=10, bias=True)
    (1): ReLU()
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (fc1act): Identity()
  (fc2): Linear(
    in_features=10, out_features=10, bias=True
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
)
'''

# calibrate for 32 batches
calibrate_ptq(prepared_model, device, test_loader, criterion, batch_size, n_batches=32)
print(prepared_model)
'''
ExampleModel(
  (quant): QuantStub(
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.4242129623889923, max_val=2.821486711502075)
  )
  (dequant): DeQuantStub()
  (conv1): ConvReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.325491428375244)
  )
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvReLU2d(
    (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=7.998041152954102)
  )
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    (0): Linear(in_features=3136, out_features=10, bias=True)
    (1): ReLU()
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=32.929901123046875)
  )
  (fc1act): Identity()
  (fc2): Linear(
    in_features=10, out_features=10, bias=True
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-24.292884826660156, max_val=10.0259428024292)
  )
)
'''

# freeze observers for a pre-conversion test
torch.ao.quantization.fake_quantize.disable_observer(prepared_model)
test(prepared_model.cpu(), 'cpu', test_loader, criterion)
# Test set: Average loss: 0.0053, Accuracy: 8973/10000 (90%)


quantized_model = torch.ao.quantization.convert(prepared_model.cpu(), inplace=False)
print(quantized_model)
'''
ExampleModel(
  (quant): Quantize(scale=tensor([0.0127]), zero_point=tensor([33]), dtype=torch.quint8)
  (dequant): DeQuantize()
  (conv1): QuantizedConvReLU2d(1, 32, kernel_size=(1, 1), stride=(1, 1), scale=0.01304114330559969, zero_point=0)
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): QuantizedConvReLU2d(32, 64, kernel_size=(1, 1), stride=(1, 1), scale=0.03141188994050026, zero_point=0)
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): QuantizedLinearReLU(in_features=3136, out_features=10, scale=0.12418901175260544, zero_point=0, qscheme=torch.per_channel_affine)
  (fc1act): Identity()
  (fc2): QuantizedLinear(in_features=10, out_features=10, scale=0.1393708884716034, zero_point=175, qscheme=torch.per_channel_affine)
)
'''
test(quantized_model, 'cpu', test_loader, criterion)

# Test set: Average loss: 0.0065, Accuracy: 8774/10000 (88%)
# when tested with converted quantization might lose some accuracy
```


##### Eager mode PTQ dynamic example


Sources:

- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [Pytorch Eager Mode PTQ Dynamic Quantization example LSTM](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html?highlight=lstm)

The difference is that the activation observer is just a `PlaceholderObserver` so only weights are quantized.
Also, dynamic quantization basically only applies to:

- `nn.Linear`
- `nn.LSTM`
- `nn.GRU`
- `nn.LSTMCell`
- `nn.RNNCell`
- `nn.GRUCell`

This means that the example needs a layer from this list to make sense, so only the Linear layers will get quantized



```python
# model imports
import torch
from torch.utils.data import DataLoader
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import PlaceholderObserver, MovingAveragePerChannelMinMaxObserver
# data imports
import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transformsv2

## Define training and testing cycles
def train(
        model,
        device,
        train_loader,
        optimizer,
        criterion,
        epoch,
        log_interval=1,
        dry_run=False,
    ):
    model.train()
    n_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            item = batch_idx * len(data)
            item_percent = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{item}/{n_data} ({item_percent:.0f}%)]\tLoss: {loss.item():.6f}')
            if dry_run:
                break


def test(
        model,
        device,
        test_loader,
        criterion,
    ):
    model.eval()
    test_loss = 0
    correct = 0
    n_data = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # sum up batch loss
            pred = torch.argmax(output,dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= n_data
    test_percent = 100. * correct / n_data
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n_data} ({test_percent:.0f}%)\n')


class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool1 = torch.nn.AdaptiveMaxPool2d((14,14))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(32,64,1,1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.act2 = torch.nn.ReLU()
        # self.pool2 = torch.nn.AdaptiveMaxPool2d((7,7))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool2 = torch.nn.MaxPool2d(2,2)
        self.fl1 = torch.nn.Flatten(start_dim=1)
        self.fc1 = torch.nn.Linear(3136, 10)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x
    def fuse_model(self):
        torch.ao.quantization.fuse_modules(
            self,
            ['conv1','bn1','act1'],
            inplace=True,
            )
        torch.ao.quantization.fuse_modules(
            self,
            ['conv2','bn2','act2'],
            inplace=True,
            )
        torch.ao.quantization.fuse_modules(
            self,
            ['fc1','fc1act'],
            inplace=True,
            )


qconfig = QConfig(
    activation=PlaceholderObserver.with_args(
        dtype=torch.quint8,
        quant_min=0,
        quant_max=255,
        is_dynamic=True,
        ),
    weight=MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ),
    )
# for dynamic models, we only need qconfig for these layers:
qconfig_spec = qconfig_spec = {
    torch.nn.Linear: qconfig,
    torch.nn.LSTM: qconfig,
    torch.nn.GRU: qconfig,
    torch.nn.LSTMCell: qconfig,
    torch.nn.RNNCell: qconfig,
    torch.nn.GRUCell: qconfig,
    torch.ao.nn.intrinsic.modules.fused.LinearReLU: qconfig,
    torch.ao.nn.intrinsic.modules.fused.LinearBn1d: qconfig,
    }

model = ExampleModel()
print(model)
'''
ExampleModel(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)
'''

## Load datasets
batch_size = 64
transform=transformsv2.Compose([
        transformsv2.ToTensor(),
        transformsv2.Normalize((0.1307,), (0.3081,)),
        ])
dataset1 = datasets.MNIST(
    '../data',
    train=True,
    download=True,
    transform=transform,
    )
dataset2 = datasets.MNIST(
    '../data',
    train=False,
    transform=transform,
    )
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)


device = 'cuda:0'
model.to(device)
lr = 0.9
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
max_epochs = 20
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=int(((len(train_loader)-1)//batch_size + 1)),
    epochs=max_epochs,
    cycle_momentum=False)
log_interval = 60

# train float model
for epoch in range(1, max_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch, log_interval)
    test(model, device, test_loader, criterion)
    scheduler.step()

# Test set: Average loss: 0.0056, Accuracy: 8893/10000 (89%)
# not the best model at all, but an example of the process


# fusion is only for eval
model.eval()
model.fuse_model()
print(model)
'''
ExampleModel(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (conv1): ConvReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
  )
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvReLU2d(
    (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
  )
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    (0): Linear(in_features=3136, out_features=10, bias=True)
    (1): ReLU()
  )
  (fc1act): Identity()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)
'''

# prepare model with observers

torch.ao.quantization.propagate_qconfig_(model,qconfig_spec)
prepared_model = torch.ao.quantization.prepare(model, inplace=False)
print(prepared_model)
'''
ExampleModel(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (conv1): ConvReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
  )
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvReLU2d(
    (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
  )
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    (0): Linear(in_features=3136, out_features=10, bias=True)
    (1): ReLU()
    (activation_post_process): PlaceholderObserver(dtype=torch.quint8, is_dynamic=True)
  )
  (fc1act): Identity()
  (fc2): Linear(
    in_features=10, out_features=10, bias=True
    (activation_post_process): PlaceholderObserver(dtype=torch.quint8, is_dynamic=True)
  )
)
'''

# freeze observers for a pre-conversion test
torch.ao.quantization.fake_quantize.disable_observer(prepared_model)
test(prepared_model.cpu(), 'cpu', test_loader, criterion)
# Test set: Average loss: 0.0056, Accuracy: 8893/10000 (89%)

dynamic_mapping = torch.ao.quantization.quantization_mappings.get_default_dynamic_quant_module_mappings()

import pprint
pprint.pprint(dynamic_mapping)
'''
{<class 'torch.nn.modules.rnn.LSTM'>: <class 'torch.ao.nn.quantized.dynamic.modules.rnn.LSTM'>,
 <class 'torch.nn.modules.rnn.GRU'>: <class 'torch.ao.nn.quantized.dynamic.modules.rnn.GRU'>,
 <class 'torch.nn.modules.linear.Linear'>: <class 'torch.ao.nn.quantized.dynamic.modules.linear.Linear'>,
 <class 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear'>: <class 'torch.ao.nn.quantized.dynamic.modules.linear.Linear'>,
 <class 'torch.nn.modules.sparse.EmbeddingBag'>: <class 'torch.ao.nn.quantized.modules.embedding_ops.EmbeddingBag'>,
 <class 'torch.nn.modules.sparse.Embedding'>: <class 'torch.ao.nn.quantized.modules.embedding_ops.Embedding'>,
 <class 'torch.nn.modules.rnn.RNNCell'>: <class 'torch.ao.nn.quantized.dynamic.modules.rnn.RNNCell'>,
 <class 'torch.nn.modules.rnn.LSTMCell'>: <class 'torch.ao.nn.quantized.dynamic.modules.rnn.LSTMCell'>,
 <class 'torch.nn.modules.rnn.GRUCell'>: <class 'torch.ao.nn.quantized.dynamic.modules.rnn.GRUCell'>,
 <class 'torch.ao.nn.intrinsic.modules.fused.LinearReLU'>: <class 'torch.ao.nn.intrinsic.quantized.dynamic.modules.linear_relu.LinearReLU'>,
 <class 'torch.ao.nn.qat.dynamic.modules.linear.Linear'>: <class 'torch.ao.nn.quantized.dynamic.modules.linear.Linear'>}
'''


quantized_model = torch.ao.quantization.convert(
    prepared_model,
    mapping=dynamic_mapping,
    inplace=False,
    )
print(quantized_model)
'''
ExampleModel(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (conv1): ConvReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
  )
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvReLU2d(
    (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
  )
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): DynamicQuantizedLinearReLU(in_features=3136, out_features=10, dtype=torch.qint8, qscheme=torch.per_channel_affine)
  (fc1act): Identity()
  (fc2): DynamicQuantizedLinear(in_features=10, out_features=10, dtype=torch.qint8, qscheme=torch.per_channel_affine)
)
'''
test(quantized_model, 'cpu', test_loader, criterion)

# Test set: Average loss: 0.0056, Accuracy: 8891/10000 (89%)

```

Notice the only changed modules are `DynamicQuantizedLinear` and `DynamiQuantizedLinearReLU`.

 
##### Eager mode QAT static example


Sources:

- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [Pytorch Eager Mode PTQ Static Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

The difference is that we train instead of calibrate just the observers. During the training, the observers will be calibrated as well as the weights being updated.

With both done at the same time, the weights will theoretically approach an optimal value that has less accuracy drop when quantized. However, since the number of operations is low enough, it might not be as noticeable in this example, since the accuracy drops every time that values are scaled to 8 bits and back.

```python
# model imports
import torch
from torch.utils.data import DataLoader
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
# data imports
import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transformsv2


## Define training and testing cycles
def train(
        model,
        device,
        train_loader,
        optimizer,
        criterion,
        epoch,
        log_interval=1,
        dry_run=False,
    ):
    model.train()
    n_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            item = batch_idx * len(data)
            item_percent = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{item}/{n_data} ({item_percent:.0f}%)]\tLoss: {loss.item():.6f}')
            if dry_run:
                break


def test(
        model,
        device,
        test_loader,
        criterion,
    ):
    model.eval()
    test_loss = 0
    correct = 0
    n_data = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # sum up batch loss
            pred = torch.argmax(output,dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= n_data
    test_percent = 100. * correct / n_data
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n_data} ({test_percent:.0f}%)\n')

class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool1 = torch.nn.AdaptiveMaxPool2d((14,14))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(32,64,1,1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.act2 = torch.nn.ReLU()
        # self.pool2 = torch.nn.AdaptiveMaxPool2d((7,7))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool2 = torch.nn.MaxPool2d(2,2)
        self.fl1 = torch.nn.Flatten(start_dim=1)
        self.fc1 = torch.nn.Linear(3136, 10)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x
    def fuse_model(self):
        torch.ao.quantization.fuse_modules_qat(
            self,
            ['conv1','bn1','act1'],
            inplace=True,
            )
        torch.ao.quantization.fuse_modules_qat(
            self,
            ['conv2','bn2','act2'],
            inplace=True,
            )
        torch.ao.quantization.fuse_modules_qat(
            self,
            ['fc1','fc1act'],
            inplace=True,
            )

qconfig = QConfig(
    activation=MovingAverageMinMaxObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        ),
    weight=MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ),
    )


model = ExampleModel()
model.qconfig = qconfig
print(model)
'''
ExampleModel(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)
'''

## Load datasets
batch_size = 64
transform=transformsv2.Compose([
        transformsv2.ToTensor(),
        transformsv2.Normalize((0.1307,), (0.3081,)),
        ])
dataset1 = datasets.MNIST(
    '../data',
    train=True,
    download=True,
    transform=transform,
    )
dataset2 = datasets.MNIST(
    '../data',
    train=False,
    transform=transform,
    )
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)


device = 'cuda:0'
model.to(device)
lr = 0.9
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
max_epochs = 20
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=int(((len(train_loader)-1)//batch_size + 1)),
    epochs=max_epochs,
    cycle_momentum=False)
log_interval = 60

# train float model
for epoch in range(1, max_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch, log_interval)
    test(model, device, test_loader, criterion)
    scheduler.step()

# Test set: Average loss: 0.0057, Accuracy: 8869/10000 (89%)
# not the best model at all, but an example of the process

# fusion is train mode for QAT
model.train()
model.fuse_model()
print(model)
'''
ExampleModel(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (conv1): ConvBnReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvBnReLU2d(
    (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    (0): Linear(in_features=3136, out_features=10, bias=True)
    (1): ReLU()
  )
  (fc1act): Identity()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)
'''


# prepare model with observers
prepared_model = torch.ao.quantization.prepare_qat(model, inplace=False)
print(prepared_model)
'''
ExampleModel(
  (quant): QuantStub(
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (dequant): DeQuantStub()
  (conv1): ConvBnReLU2d(
    1, 32, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([], device='cuda:0'), max_val=tensor([], device='cuda:0'))
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvBnReLU2d(
    32, 64, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([], device='cuda:0'), max_val=tensor([], device='cuda:0'))
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    in_features=3136, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([], device='cuda:0'), max_val=tensor([], device='cuda:0'))
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (fc1act): Identity()
  (fc2): Linear(
    in_features=10, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([], device='cuda:0'), max_val=tensor([], device='cuda:0'))
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
)
'''

# set parameters for qat
device = 'cuda:0'
prepared_model.to(device)
# small LR to avoid changing the weights much after float model training
lr = 0.00001
optimizer = torch.optim.Adadelta(prepared_model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
# not as many epochs as the float model, since we are only adjusting weights for quantization
max_epochs = 5
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=int(((len(train_loader)-1)//batch_size + 1)),
    epochs=max_epochs,
    cycle_momentum=False)
log_interval = 60

# train qat model
for epoch in range(1, max_epochs + 1):
    train(prepared_model, device, train_loader, optimizer, criterion, epoch, log_interval)
    test(prepared_model, device, test_loader, criterion)

# Test set: Average loss: 0.0050, Accuracy: 9030/10000 (90%)

print(prepared_model)
'''
ExampleModel(
  (quant): QuantStub(
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.4242129623889923, max_val=2.821486711502075)
  )
  (dequant): DeQuantStub()
  (conv1): ConvBnReLU2d(
    1, 32, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([ 0.9805,  0.9878, -0.9778, -0.9910, -0.9607,  1.0020,  0.9854, -0.9932,
               0.9920, -1.0192,  0.9573,  0.9959,  1.0661, -0.9587, -0.4509,  0.8841,
               0.9185,  0.9696, -0.9722, -1.0099,  1.0207, -1.0131,  0.9228,  0.9731,
              -0.8032, -0.9803, -0.9691,  1.0209, -0.9520,  1.0132,  1.0179, -1.0628],
             device='cuda:0'), max_val=tensor([ 0.9805,  0.9878, -0.9778, -0.9910, -0.9607,  1.0020,  0.9854, -0.9932,
               0.9920, -1.0192,  0.9573,  0.9959,  1.0661, -0.9587, -0.4509,  0.8841,
               0.9185,  0.9696, -0.9722, -1.0099,  1.0207, -1.0131,  0.9228,  0.9731,
              -0.8032, -0.9803, -0.9691,  1.0209, -0.9520,  1.0132,  1.0179, -1.0628],
             device='cuda:0')
    )
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.486288547515869)
  )
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvBnReLU2d(
    32, 64, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-2.1960, -0.2812, -1.7402, -0.8100, -3.3399, -1.0657, -0.1177, -0.6993,
              -0.2671, -4.9234, -0.5415, -3.2926, -0.1593, -0.3427, -0.4231, -3.2403,
              -0.7386, -2.6828, -5.1171, -3.6965, -0.2139, -9.6442, -0.3108, -2.1643,
              -0.7892, -2.2819, -2.0034, -0.4834, -0.3995, -4.6650, -1.5611, -1.3696,
              -3.9522, -0.3022, -1.5632, -0.4557, -5.8931, -0.4400, -1.2626, -0.5098,
              -3.3187, -0.3899, -0.4554, -2.8338, -0.4487, -0.2008, -1.1349, -5.3991,
              -4.4046, -0.4110, -1.2552, -0.5631, -0.3380, -2.7315, -2.2920, -2.1396,
              -0.4084, -6.2974, -1.1824, -0.1679, -2.1181, -0.8331, -1.1392, -5.4736],
             device='cuda:0'), max_val=tensor([2.1027, 0.2706, 2.1971, 0.7181, 2.7690, 0.9632, 0.1437, 0.6662, 0.2961,
              3.6605, 0.5448, 2.8305, 0.1597, 0.3670, 0.4135, 4.4315, 0.7591, 3.2193,
              4.7957, 4.5160, 0.2183, 8.5344, 0.3707, 2.4349, 1.2237, 2.4649, 2.5588,
              0.4111, 0.5481, 5.3878, 1.8818, 0.9792, 3.3811, 0.3097, 0.9306, 0.5589,
              5.2913, 0.3896, 0.8146, 0.8410, 2.5406, 0.2423, 0.4662, 3.2510, 0.4216,
              0.2331, 0.7295, 5.5472, 3.2039, 0.3767, 1.2058, 0.4847, 0.2246, 2.3556,
              2.5913, 1.7738, 0.4303, 6.8638, 0.9290, 0.2386, 2.2515, 1.1236, 0.7875,
              4.1441], device='cuda:0')
    )
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=7.873152732849121)
  )
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    in_features=3136, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.1110, -0.1682, -0.1074, -0.0716, -0.1544, -0.1686, -0.1512, -0.2297,
              -0.1243, -0.1656], device='cuda:0'), max_val=tensor([0.1568, 0.1229, 0.1314, 0.0929, 0.1936, 0.1594, 0.1684, 0.1468, 0.1751,
              0.1768], device='cuda:0')
    )
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=52.048946380615234)
  )
  (fc1act): Identity()
  (fc2): Linear(
    in_features=10, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.4732, -0.7787, -0.3168, -0.1518, -0.6355, -0.3758, -0.5703, -0.3268,
              -0.3168, -0.5046], device='cuda:0'), max_val=tensor([0.3203, 0.4310, 0.3799, 0.3980, 0.3096, 0.2491, 0.4260, 0.2836, 0.3210,
              0.3214], device='cuda:0')
    )
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-21.06275177001953, max_val=15.43133544921875)
  )
)
'''

# freeze observers for a pre-conversion test
torch.ao.quantization.fake_quantize.disable_observer(prepared_model)
test(prepared_model, device, test_loader, criterion)

# Test set: Average loss: 0.0050, Accuracy: 9031/10000 (90%)

# quantize can only do eval and cpu
quantized_model = torch.ao.quantization.convert(prepared_model.eval().cpu(), inplace=False)
print(quantized_model)
'''
ExampleModel(
  (quant): Quantize(scale=tensor([0.0127]), zero_point=tensor([33]), dtype=torch.quint8)
  (dequant): DeQuantize()
  (conv1): QuantizedConvReLU2d(1, 32, kernel_size=(1, 1), stride=(1, 1), scale=0.01362644787877798, zero_point=0)
  (bn1): Identity()
  (act1): Identity()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): QuantizedConvReLU2d(32, 64, kernel_size=(1, 1), stride=(1, 1), scale=0.030343253165483475, zero_point=0)
  (bn2): Identity()
  (act2): Identity()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): QuantizedLinearReLU(in_features=3136, out_features=10, scale=0.20409592986106873, zero_point=0, qscheme=torch.per_channel_affine)
  (fc1act): Identity()
  (fc2): QuantizedLinear(in_features=10, out_features=10, scale=0.14229224622249603, zero_point=147, qscheme=torch.per_channel_affine)
)
'''
test(quantized_model, 'cpu', test_loader, criterion)

# Test set: Average loss: 0.0051, Accuracy: 9021/10000 (90%)

```

Technically there should be some loss in accuracy, but since the model is simple and there's not many quantization nodes, there's not any noticeable loss in accuracy in this example.


##### FX Graph mode PTQ static example

Sources:
- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [Pytorch API docs: QConfigMapping](https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html)
- [Pytorch FX Quantization user guide](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)
- [Pytorch FX PTQ Static Quantization Tutorial](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)


```python
import torch
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.fx.tracer import QuantizationTracer
from torch.fx import GraphModule
from torch.ao.quantization.quantize_fx import _fuse_fx, convert_fx
from torch.ao.quantization.fx import prepare

# data imports
import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transformsv2
from torch.utils.data import DataLoader

import copy

## Define training and testing cycles
def train(
        model,
        device,
        train_loader,
        optimizer,
        criterion,
        epoch,
        log_interval=1,
        dry_run=False,
    ):
    model.train()
    n_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            item = batch_idx * len(data)
            item_percent = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{item}/{n_data} ({item_percent:.0f}%)]\tLoss: {loss.item():.6f}')
            if dry_run:
                break


def test(
        model,
        device,
        test_loader,
        criterion,
    ):
    model.eval()
    test_loss = 0
    correct = 0
    n_data = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # sum up batch loss
            pred = torch.argmax(output,dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= n_data
    test_percent = 100. * correct / n_data
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n_data} ({test_percent:.0f}%)\n')


# test a few batches for calibration
def calibrate_ptq(
        prepared_model,
        device,
        test_loader,
        criterion,
        batch_size,
        n_batches=32,
    ):
    prepared_model.eval()
    cal_loss = 0
    correct = 0
    n_data = (len(test_loader.dataset) / batch_size) * n_batches
    with torch.no_grad():
        for (i,(data, target)) in enumerate(test_loader):
            if i<=n_batches:
                data, target = data.to(device), target.to(device)
                output = prepared_model(data)
                cal_loss += criterion(output, target).item()
                # sum up batch loss
                pred = torch.argmax(output,dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                break
    cal_loss /= n_data
    cal_percent = 100. * correct / n_data
    print(f'\nCalibrate set: Average loss: {cal_loss:.4f}, Accuracy: {correct}/{n_data} ({cal_percent:.0f}%)\n')


class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool1 = torch.nn.AdaptiveMaxPool2d((14,14))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(32,64,1,1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.act2 = torch.nn.ReLU()
        # self.pool2 = torch.nn.AdaptiveMaxPool2d((7,7))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool2 = torch.nn.MaxPool2d(2,2)
        self.fl1 = torch.nn.Flatten(start_dim=1)
        self.fc1 = torch.nn.Linear(3136, 10)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        return x


# define the qconfig_mapping
qconfig_mapping = QConfigMapping().set_global(
        QConfig(
            activation=MovingAverageMinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                ),
            weight=MovingAveragePerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                ),
            )
        )


model = ExampleModel()
print(model)
'''
ExampleModel(
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)
'''

# FX trace the model
tracer = QuantizationTracer(skipped_module_names=[], skipped_module_classes=[])
graph = tracer.trace(model)
traced_fx = GraphModule(tracer.root, graph, 'ExampleModel')
print(traced_fx)
'''
ExampleModel(
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)



def forward(self, x):
    conv1 = self.conv1(x);  x = None
    bn1 = self.bn1(conv1);  conv1 = None
    act1 = self.act1(bn1);  bn1 = None
    pool1 = self.pool1(act1);  act1 = None
    conv2 = self.conv2(pool1);  pool1 = None
    bn2 = self.bn2(conv2);  conv2 = None
    act2 = self.act2(bn2);  bn2 = None
    pool2 = self.pool2(act2);  act2 = None
    fl1 = self.fl1(pool2);  pool2 = None
    fc1 = self.fc1(fl1);  fl1 = None
    fc1act = self.fc1act(fc1);  fc1 = None
    fc2 = self.fc2(fc1act);  fc1act = None
    return fc2
    
# To see more debug info, please use `graph_module.print_readable()`
'''


## Load datasets
batch_size = 64
transform=transformsv2.Compose([
        transformsv2.ToTensor(),
        transformsv2.Normalize((0.1307,), (0.3081,)),
        ])
dataset1 = datasets.MNIST(
    '../data',
    train=True,
    download=True,
    transform=transform,
    )
dataset2 = datasets.MNIST(
    '../data',
    train=False,
    transform=transform,
    )
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)


device = 'cuda:0'
traced_fx.to(device)
lr = 0.9
optimizer = torch.optim.Adadelta(traced_fx.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
max_epochs = 20
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=int(((len(train_loader)-1)//batch_size + 1)),
    epochs=max_epochs,
    cycle_momentum=False)
log_interval = 60


# train float model
for epoch in range(1, max_epochs + 1):
    train(traced_fx, device, train_loader, optimizer, criterion, epoch, log_interval)
    test(traced_fx, device, test_loader, criterion)
    scheduler.step()

# Test set: Average loss: 0.0049, Accuracy: 9052/10000 (91%)
# not the best model at all, but an example of the process


# FX fuse modules automatically
# using qat temporarily so we don't need to worry about eval or train
fused_fx = _fuse_fx(
    copy.deepcopy(traced_fx), # default behavior is in-place
    is_qat=True, #useful for non Sequence types even if not qat
    backend_config=None, # default native
    )
print(fused_fx)
'''
GraphModule(
  (conv1): ConvBnReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvBnReLU2d(
    (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    (0): Linear(in_features=3136, out_features=10, bias=True)
    (1): ReLU()
  )
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)



def forward(self, x):
    conv1 = self.conv1(x);  x = None
    pool1 = self.pool1(conv1);  conv1 = None
    conv2 = self.conv2(pool1);  pool1 = None
    pool2 = self.pool2(conv2);  conv2 = None
    fl1 = self.fl1(pool2);  pool2 = None
    fc1 = self.fc1(fl1);  fl1 = None
    fc2 = self.fc2(fc1);  fc1 = None
    return fc2
    
# To see more debug info, please use `graph_module.print_readable()`
'''



# FX prepare the quantization nodes
example_inputs = (torch.randn(1,1,28,28),)
prepared_fx = prepare(
    fused_fx,
    qconfig_mapping=qconfig_mapping,
    node_name_to_scope=tracer.node_name_to_scope,
    is_qat=True, # convenient even if not QAT
    example_inputs=example_inputs,
    backend_config=None, # default native
    )

# calibrate for 32 batches
calibrate_ptq(prepared_fx, device, test_loader, criterion, batch_size, n_batches=32)
print(prepared_fx)
'''
GraphModule(
  (activation_post_process_0): MovingAverageMinMaxObserver(min_val=-0.4242129623889923, max_val=2.821486711502075)
  (conv1): ConvBnReLU2d(
    1, 32, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-1.0541,  1.1192,  0.9966,  0.8533,  0.9130, -0.8899,  0.9133,  0.9188,
              -0.9015,  0.9355,  1.1225,  1.0703, -0.9944, -0.9238, -1.0428, -0.9259,
              -0.9659,  0.9695, -0.8809, -0.3419,  0.9046,  0.9958,  0.9538, -0.9103,
               0.9404, -1.0426, -0.9998, -0.9212,  1.0393,  0.9543, -1.0904, -1.0882],
             device='cuda:0'), max_val=tensor([-1.0541,  1.1192,  0.9966,  0.8533,  0.9130, -0.8899,  0.9133,  0.9188,
              -0.9015,  0.9355,  1.1225,  1.0703, -0.9944, -0.9238, -1.0428, -0.9259,
              -0.9659,  0.9695, -0.8809, -0.3419,  0.9046,  0.9958,  0.9538, -0.9103,
               0.9404, -1.0426, -0.9998, -0.9212,  1.0393,  0.9543, -1.0904, -1.0882],
             device='cuda:0')
    )
  )
  (activation_post_process_1): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.6681931018829346)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (activation_post_process_2): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.6681931018829346)
  (conv2): ConvBnReLU2d(
    32, 64, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([ -0.3927,  -0.3222,  -1.5281,  -0.8538,  -2.8257,  -2.7587,  -1.0526,
               -1.5627,  -2.7982,  -3.1387,  -0.2418,  -0.8986,  -0.9966,  -0.7794,
               -0.5468,  -3.2806,  -0.7812,  -0.3512,  -0.3102,  -1.0725,  -4.2466,
               -3.1377,  -0.8714,  -1.1984,  -1.3492,  -2.9758,  -0.2321,  -6.4170,
               -1.8255,  -3.0744,  -0.3434,  -0.4103,  -2.3586,  -2.1883,  -0.1271,
               -3.8437,  -5.6612,  -1.8619,  -2.3714,  -0.2705,  -0.6426, -12.0482,
               -2.0589,  -1.1064,  -6.1820,  -1.1337,  -2.6746,  -0.5178,  -1.1923,
               -4.2477,  -0.3463,  -0.4841,  -4.9971,  -0.1930,  -1.5936,  -0.5505,
               -0.2784,  -0.2574,  -0.2902,  -2.4461,  -3.2089,  -0.9089,  -3.9912,
               -1.9629], device='cuda:0'), max_val=tensor([0.4634, 0.3499, 1.9163, 0.7068, 2.9404, 2.2394, 1.1182, 1.3347, 3.0304,
              2.6291, 0.2671, 0.8090, 0.9679, 1.2698, 0.5145, 3.0952, 1.1440, 0.3551,
              0.3992, 0.5905, 3.6407, 3.6593, 0.9004, 1.3843, 0.9959, 2.3280, 0.2082,
              4.5630, 1.4452, 2.6930, 0.2621, 0.3042, 2.5872, 2.6573, 0.2083, 2.2701,
              3.2977, 1.3170, 2.8580, 0.2521, 1.1895, 6.4022, 2.2630, 0.7048, 3.9667,
              0.7951, 2.2147, 0.8853, 1.7365, 3.7394, 0.3032, 0.5592, 4.3958, 0.1736,
              2.1026, 0.4553, 0.2629, 0.2573, 0.2822, 1.8175, 2.9617, 0.8305, 1.6504,
              1.0457], device='cuda:0')
    )
  )
  (activation_post_process_3): MovingAverageMinMaxObserver(min_val=0.0, max_val=8.76562213897705)
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (activation_post_process_4): MovingAverageMinMaxObserver(min_val=0.0, max_val=8.76562213897705)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (activation_post_process_5): MovingAverageMinMaxObserver(min_val=0.0, max_val=8.766530990600586)
  (fc1): LinearReLU(
    in_features=3136, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.1720, -0.1783, -0.2047, -0.2311, -0.1966, -0.1872, -0.1699, -0.1503,
              -0.1947, -0.1436], device='cuda:0'), max_val=tensor([0.1856, 0.1963, 0.1965, 0.1233, 0.1830, 0.1965, 0.1782, 0.1988, 0.2141,
              0.2874], device='cuda:0')
    )
  )
  (activation_post_process_6): MovingAverageMinMaxObserver(min_val=0.0, max_val=46.4373779296875)
  (fc2): Linear(
    in_features=10, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(
      min_val=tensor([-0.4710, -0.5442, -0.6563, -0.3092, -0.5604, -0.6425, -0.5238, -0.4620,
              -0.3159, -0.3505], device='cuda:0'), max_val=tensor([0.3491, 0.4453, 0.3340, 0.2617, 0.3874, 0.3443, 0.3004, 0.3230, 0.3145,
              0.4552], device='cuda:0')
    )
  )
  (activation_post_process_7): MovingAverageMinMaxObserver(min_val=-32.19971466064453, max_val=11.140828132629395)
)



def forward(self, x):
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
    conv1 = self.conv1(activation_post_process_0);  activation_post_process_0 = None
    activation_post_process_1 = self.activation_post_process_1(conv1);  conv1 = None
    pool1 = self.pool1(activation_post_process_1);  activation_post_process_1 = None
    activation_post_process_2 = self.activation_post_process_2(pool1);  pool1 = None
    conv2 = self.conv2(activation_post_process_2);  activation_post_process_2 = None
    activation_post_process_3 = self.activation_post_process_3(conv2);  conv2 = None
    pool2 = self.pool2(activation_post_process_3);  activation_post_process_3 = None
    activation_post_process_4 = self.activation_post_process_4(pool2);  pool2 = None
    fl1 = self.fl1(activation_post_process_4);  activation_post_process_4 = None
    activation_post_process_5 = self.activation_post_process_5(fl1);  fl1 = None
    fc1 = self.fc1(activation_post_process_5);  activation_post_process_5 = None
    activation_post_process_6 = self.activation_post_process_6(fc1);  fc1 = None
    fc2 = self.fc2(activation_post_process_6);  activation_post_process_6 = None
    activation_post_process_7 = self.activation_post_process_7(fc2);  fc2 = None
    return activation_post_process_7
    
# To see more debug info, please use `graph_module.print_readable()`
'''

# freeze observers for a pre-conversion test
torch.ao.quantization.fake_quantize.disable_observer(prepared_fx)
test(prepared_fx.cpu(), 'cpu', test_loader, criterion)
# Test set: Average loss: 0.0049, Accuracy: 9052/10000 (91%)


quantized_fx = convert_fx(prepared_fx.eval().cpu())
print(quantized_fx)
'''
GraphModule(
  (conv1): QuantizedConvReLU2d(1, 32, kernel_size=(1, 1), stride=(1, 1), scale=0.014385070651769638, zero_point=0)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): QuantizedConvReLU2d(32, 64, kernel_size=(1, 1), stride=(1, 1), scale=0.034487225115299225, zero_point=0)
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): QuantizedLinearReLU(in_features=3136, out_features=10, scale=0.1664087176322937, zero_point=0, qscheme=torch.per_channel_affine)
  (fc2): QuantizedLinear(in_features=10, out_features=10, scale=0.16956472396850586, zero_point=183, qscheme=torch.per_channel_affine)
)



def forward(self, x):
    conv1_input_scale_0 = self.conv1_input_scale_0
    conv1_input_zero_point_0 = self.conv1_input_zero_point_0
    quantize_per_tensor = torch.quantize_per_tensor(x, conv1_input_scale_0, conv1_input_zero_point_0, torch.quint8);  x = conv1_input_scale_0 = conv1_input_zero_point_0 = None
    conv1 = self.conv1(quantize_per_tensor);  quantize_per_tensor = None
    pool1 = self.pool1(conv1);  conv1 = None
    conv2 = self.conv2(pool1);  pool1 = None
    pool2 = self.pool2(conv2);  conv2 = None
    dequantize_4 = pool2.dequantize();  pool2 = None
    fl1 = self.fl1(dequantize_4);  dequantize_4 = None
    fl1_scale_0 = self.fl1_scale_0
    fl1_zero_point_0 = self.fl1_zero_point_0
    quantize_per_tensor_5 = torch.quantize_per_tensor(fl1, fl1_scale_0, fl1_zero_point_0, torch.quint8);  fl1 = fl1_scale_0 = fl1_zero_point_0 = None
    fc1 = self.fc1(quantize_per_tensor_5);  quantize_per_tensor_5 = None
    fc2 = self.fc2(fc1);  fc1 = None
    dequantize_7 = fc2.dequantize();  fc2 = None
    return dequantize_7
    
# To see more debug info, please use `graph_module.print_readable()`
'''

test(quantized_fx, 'cpu', test_loader, criterion)
# Test set: Average loss: 0.0051, Accuracy: 8998/10000 (90%)
# when tested with converted quantization might lose some accuracy
```


##### FX Graph mode PTQ dynamic example

Sources:
- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [Pytorch API docs: QConfigMapping](https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html)
- [Pytorch FX Quantization user guide](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)
- [Pytorch FX PTQ Dynamic Quantization Tutorial](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html)

```python
# model imports
import torch
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import PlaceholderObserver, MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.fx.tracer import QuantizationTracer
from torch.fx import GraphModule
from torch.ao.quantization.quantize_fx import _fuse_fx, convert_fx
from torch.ao.quantization.fx import prepare

# data imports
import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transformsv2
from torch.utils.data import DataLoader

import copy

## Define training and testing cycles
def train(
        model,
        device,
        train_loader,
        optimizer,
        criterion,
        epoch,
        log_interval=1,
        dry_run=False,
    ):
    model.train()
    n_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            item = batch_idx * len(data)
            item_percent = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{item}/{n_data} ({item_percent:.0f}%)]\tLoss: {loss.item():.6f}')
            if dry_run:
                break


def test(
        model,
        device,
        test_loader,
        criterion,
    ):
    model.eval()
    test_loss = 0
    correct = 0
    n_data = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # sum up batch loss
            pred = torch.argmax(output,dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= n_data
    test_percent = 100. * correct / n_data
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n_data} ({test_percent:.0f}%)\n')


class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool1 = torch.nn.AdaptiveMaxPool2d((14,14))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(32,64,1,1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.act2 = torch.nn.ReLU()
        # self.pool2 = torch.nn.AdaptiveMaxPool2d((7,7))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool2 = torch.nn.MaxPool2d(2,2)
        self.fl1 = torch.nn.Flatten(start_dim=1)
        self.fc1 = torch.nn.Linear(3136, 10)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        return x



# define the qconfig_mapping
dynamic_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(
        dtype=torch.quint8,
        quant_min=0,
        quant_max=255,
        is_dynamic=True,
        ),
    weight=MovingAveragePerChannelMinMaxObserver.with_args(
        averaging_constant=1,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ),
    )

# while technically we don't need to add qconfig to the convolutions, because of fusion, 
# we should apply it globally. 
# Note that unlike in Eager mode, this is not a problem.

qconfig_mapping = QConfigMapping().set_global(dynamic_qconfig)


model = ExampleModel()
print(model)
'''
ExampleModel(
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)
'''

# FX trace the model
tracer = QuantizationTracer(skipped_module_names=[], skipped_module_classes=[])
graph = tracer.trace(model)
traced_fx = GraphModule(tracer.root, graph, 'ExampleModel')
print(traced_fx)
'''
ExampleModel(
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)



def forward(self, x):
    conv1 = self.conv1(x);  x = None
    bn1 = self.bn1(conv1);  conv1 = None
    act1 = self.act1(bn1);  bn1 = None
    pool1 = self.pool1(act1);  act1 = None
    conv2 = self.conv2(pool1);  pool1 = None
    bn2 = self.bn2(conv2);  conv2 = None
    act2 = self.act2(bn2);  bn2 = None
    pool2 = self.pool2(act2);  act2 = None
    fl1 = self.fl1(pool2);  pool2 = None
    fc1 = self.fc1(fl1);  fl1 = None
    fc1act = self.fc1act(fc1);  fc1 = None
    fc2 = self.fc2(fc1act);  fc1act = None
    return fc2
    
# To see more debug info, please use `graph_module.print_readable()`
'''


## Load datasets
batch_size = 64
transform=transformsv2.Compose([
        transformsv2.ToTensor(),
        transformsv2.Normalize((0.1307,), (0.3081,)),
        ])
dataset1 = datasets.MNIST(
    '../data',
    train=True,
    download=True,
    transform=transform,
    )
dataset2 = datasets.MNIST(
    '../data',
    train=False,
    transform=transform,
    )
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)


device = 'cuda:0'
traced_fx.to(device)
lr = 0.9
optimizer = torch.optim.Adadelta(traced_fx.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
max_epochs = 20
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=int(((len(train_loader)-1)//batch_size + 1)),
    epochs=max_epochs,
    cycle_momentum=False)
log_interval = 60


# train float model
for epoch in range(1, max_epochs + 1):
    train(traced_fx, device, train_loader, optimizer, criterion, epoch, log_interval)
    test(traced_fx, device, test_loader, criterion)
    scheduler.step()

# Test set: Average loss: 0.0051, Accuracy: 8999/10000 (90%)
# not the best model at all, but an example of the process


# FX fuse modules automatically
# using qat temporarily so we don't need to worry about eval or train
fused_fx = _fuse_fx(
    copy.deepcopy(traced_fx), # default behavior is in-place
    is_qat=True, #useful for non Sequence types even if not qat
    backend_config=None, # default native
    )
print(fused_fx)
'''
GraphModule(
  (conv1): ConvBnReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvBnReLU2d(
    (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    (0): Linear(in_features=3136, out_features=10, bias=True)
    (1): ReLU()
  )
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)



def forward(self, x):
    conv1 = self.conv1(x);  x = None
    pool1 = self.pool1(conv1);  conv1 = None
    conv2 = self.conv2(pool1);  pool1 = None
    pool2 = self.pool2(conv2);  conv2 = None
    fl1 = self.fl1(pool2);  pool2 = None
    fc1 = self.fc1(fl1);  fl1 = None
    fc2 = self.fc2(fc1);  fc1 = None
    return fc2
    
# To see more debug info, please use `graph_module.print_readable()`
'''



# FX prepare the quantization nodes
example_inputs = (torch.randn(1,1,28,28),)
prepared_fx = prepare(
    fused_fx,
    qconfig_mapping=qconfig_mapping,
    node_name_to_scope=tracer.node_name_to_scope,
    is_qat=True, # convenient even if not QAT
    example_inputs=example_inputs,
    backend_config=None, # default native
    )
print(prepared_fx)
'''
GraphModule(
  (conv1): ConvBnReLU2d(
    1, 32, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvBnReLU2d(
    32, 64, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (activation_post_process_0): PlaceholderObserver(dtype=torch.quint8, is_dynamic=True)
  (fc1): LinearReLU(
    in_features=3136, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_1): PlaceholderObserver(dtype=torch.quint8, is_dynamic=True)
  (fc2): Linear(
    in_features=10, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
)



def forward(self, x):
    conv1 = self.conv1(x);  x = None
    pool1 = self.pool1(conv1);  conv1 = None
    conv2 = self.conv2(pool1);  pool1 = None
    pool2 = self.pool2(conv2);  conv2 = None
    fl1 = self.fl1(pool2);  pool2 = None
    activation_post_process_0 = self.activation_post_process_0(fl1);  fl1 = None
    fc1 = self.fc1(activation_post_process_0);  activation_post_process_0 = None
    activation_post_process_1 = self.activation_post_process_1(fc1);  fc1 = None
    fc2 = self.fc2(activation_post_process_1);  activation_post_process_1 = None
    return fc2
    
# To see more debug info, please use `graph_module.print_readable()`
'''

# freeze observers for a pre-conversion test
torch.ao.quantization.fake_quantize.disable_observer(prepared_fx)
test(prepared_fx.eval().cpu(), 'cpu', test_loader, criterion)
# Test set: Average loss: 0.0051, Accuracy: 8999/10000 (90%)



quantized_fx = convert_fx(prepared_fx.eval().cpu())
print(quantized_fx)
'''
GraphModule(
  (conv1): ConvReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
  )
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvReLU2d(
    (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
  )
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): DynamicQuantizedLinearReLU(in_features=3136, out_features=10, dtype=torch.qint8, qscheme=torch.per_channel_affine)
  (fc2): DynamicQuantizedLinear(in_features=10, out_features=10, dtype=torch.qint8, qscheme=torch.per_channel_affine)
)



def forward(self, x):
    conv1 = self.conv1(x);  x = None
    pool1 = self.pool1(conv1);  conv1 = None
    conv2 = self.conv2(pool1);  pool1 = None
    pool2 = self.pool2(conv2);  conv2 = None
    fl1 = self.fl1(pool2);  pool2 = None
    fc1 = self.fc1(fl1);  fl1 = None
    fc2 = self.fc2(fc1);  fc1 = None
    return fc2
    
# To see more debug info, please use `graph_module.print_readable()`
'''

test(quantized_fx, 'cpu', test_loader, criterion)
# Test set: Average loss: 0.0051, Accuracy: 9006/10000 (90%)
# when tested with converted quantization might lose some accuracy
```

Notice that despite the lack of specific qconfig, the conversion handles the dynamic quantized conversions correctly.


##### FX Graph mode QAT static example


Sources:
- [Pytorch Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [Pytorch API docs: QConfigMapping](https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html)
- [Pytorch FX Quantization user guide](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)
- [Pytorch FX PTQ Static Quantization Tutorial](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)


```python
import torch
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.fx.tracer import QuantizationTracer
from torch.fx import GraphModule
from torch.ao.quantization.quantize_fx import _fuse_fx, convert_fx
from torch.ao.quantization.fx import prepare

# data imports
import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transformsv2
from torch.utils.data import DataLoader

import copy

## Define training and testing cycles
def train(
        model,
        device,
        train_loader,
        optimizer,
        criterion,
        epoch,
        log_interval=1,
        dry_run=False,
    ):
    model.train()
    n_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            item = batch_idx * len(data)
            item_percent = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{item}/{n_data} ({item_percent:.0f}%)]\tLoss: {loss.item():.6f}')
            if dry_run:
                break


def test(
        model,
        device,
        test_loader,
        criterion,
    ):
    model.eval()
    test_loss = 0
    correct = 0
    n_data = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # sum up batch loss
            pred = torch.argmax(output,dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= n_data
    test_percent = 100. * correct / n_data
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n_data} ({test_percent:.0f}%)\n')


# test a few batches for calibration
def calibrate_ptq(
        prepared_model,
        device,
        test_loader,
        criterion,
        batch_size,
        n_batches=32,
    ):
    prepared_model.eval()
    cal_loss = 0
    correct = 0
    n_data = (len(test_loader.dataset) / batch_size) * n_batches
    with torch.no_grad():
        for (i,(data, target)) in enumerate(test_loader):
            if i<=n_batches:
                data, target = data.to(device), target.to(device)
                output = prepared_model(data)
                cal_loss += criterion(output, target).item()
                # sum up batch loss
                pred = torch.argmax(output,dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                break
    cal_loss /= n_data
    cal_percent = 100. * correct / n_data
    print(f'\nCalibrate set: Average loss: {cal_loss:.4f}, Accuracy: {correct}/{n_data} ({cal_percent:.0f}%)\n')


class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool1 = torch.nn.AdaptiveMaxPool2d((14,14))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(32,64,1,1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.act2 = torch.nn.ReLU()
        # self.pool2 = torch.nn.AdaptiveMaxPool2d((7,7))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool2 = torch.nn.MaxPool2d(2,2)
        self.fl1 = torch.nn.Flatten(start_dim=1)
        self.fc1 = torch.nn.Linear(3136, 10)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        return x


# define the qconfig_mapping
qconfig_mapping = QConfigMapping().set_global(
        QConfig(
            activation=MovingAverageMinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                ),
            weight=MovingAveragePerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                ),
            )
        )


model = ExampleModel()
print(model)
'''
ExampleModel(
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)
'''

# FX trace the model
tracer = QuantizationTracer(skipped_module_names=[], skipped_module_classes=[])
graph = tracer.trace(model)
traced_fx = GraphModule(tracer.root, graph, 'ExampleModel')
print(traced_fx)
'''
ExampleModel(
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)



def forward(self, x):
    conv1 = self.conv1(x);  x = None
    bn1 = self.bn1(conv1);  conv1 = None
    act1 = self.act1(bn1);  bn1 = None
    pool1 = self.pool1(act1);  act1 = None
    conv2 = self.conv2(pool1);  pool1 = None
    bn2 = self.bn2(conv2);  conv2 = None
    act2 = self.act2(bn2);  bn2 = None
    pool2 = self.pool2(act2);  act2 = None
    fl1 = self.fl1(pool2);  pool2 = None
    fc1 = self.fc1(fl1);  fl1 = None
    fc1act = self.fc1act(fc1);  fc1 = None
    fc2 = self.fc2(fc1act);  fc1act = None
    return fc2
    
# To see more debug info, please use `graph_module.print_readable()`
'''


## Load datasets
batch_size = 64
transform=transformsv2.Compose([
        transformsv2.ToTensor(),
        transformsv2.Normalize((0.1307,), (0.3081,)),
        ])
dataset1 = datasets.MNIST(
    '../data',
    train=True,
    download=True,
    transform=transform,
    )
dataset2 = datasets.MNIST(
    '../data',
    train=False,
    transform=transform,
    )
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)


# device = 'cuda:0'
device='cpu'
traced_fx.to(device)
lr = 0.9
optimizer = torch.optim.Adadelta(traced_fx.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
max_epochs = 20
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=int(((len(train_loader)-1)//batch_size + 1)),
    epochs=max_epochs,
    cycle_momentum=False)
log_interval = 60


# train float model
for epoch in range(1, max_epochs + 1):
    train(traced_fx, device, train_loader, optimizer, criterion, epoch, log_interval)
    test(traced_fx, device, test_loader, criterion)
    scheduler.step()

# Test set: Average loss: 0.0066, Accuracy: 8708/10000 (87%)
# not the best model at all, but an example of the process


# FX fuse modules automatically
# using qat temporarily so we don't need to worry about eval or train
fused_fx = _fuse_fx(
    copy.deepcopy(traced_fx), # default behavior is in-place
    is_qat=True, #useful for non Sequence types even if not qat
    backend_config=None, # default native
    )
print(fused_fx)
'''
GraphModule(
  (conv1): ConvBnReLU2d(
    (0): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): ConvBnReLU2d(
    (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): LinearReLU(
    (0): Linear(in_features=3136, out_features=10, bias=True)
    (1): ReLU()
  )
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)



def forward(self, x):
    conv1 = self.conv1(x);  x = None
    pool1 = self.pool1(conv1);  conv1 = None
    conv2 = self.conv2(pool1);  pool1 = None
    pool2 = self.pool2(conv2);  conv2 = None
    fl1 = self.fl1(pool2);  pool2 = None
    fc1 = self.fc1(fl1);  fl1 = None
    fc2 = self.fc2(fc1);  fc1 = None
    return fc2
    
# To see more debug info, please use `graph_module.print_readable()`
'''



# FX prepare the quantization nodes
example_inputs = (torch.randn(1,1,28,28),)
prepared_fx = prepare(
    fused_fx, # default behavior is in-place
    qconfig_mapping=qconfig_mapping,
    node_name_to_scope=tracer.node_name_to_scope,
    is_qat=True, # convenient even if not QAT
    example_inputs=example_inputs,
    backend_config=None, # default native
    )

print(prepared_fx)
'''
GraphModule(
  (activation_post_process_0): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (conv1): ConvBnReLU2d(
    1, 32, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_1): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (activation_post_process_2): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (conv2): ConvBnReLU2d(
    32, 64, kernel_size=(1, 1), stride=(1, 1)
    (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_3): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (activation_post_process_4): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (activation_post_process_5): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (fc1): LinearReLU(
    in_features=3136, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_6): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  (fc2): Linear(
    in_features=10, out_features=10, bias=True
    (weight_fake_quant): MovingAveragePerChannelMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (activation_post_process_7): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
)



def forward(self, x):
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
    conv1 = self.conv1(activation_post_process_0);  activation_post_process_0 = None
    activation_post_process_1 = self.activation_post_process_1(conv1);  conv1 = None
    pool1 = self.pool1(activation_post_process_1);  activation_post_process_1 = None
    activation_post_process_2 = self.activation_post_process_2(pool1);  pool1 = None
    conv2 = self.conv2(activation_post_process_2);  activation_post_process_2 = None
    activation_post_process_3 = self.activation_post_process_3(conv2);  conv2 = None
    pool2 = self.pool2(activation_post_process_3);  activation_post_process_3 = None
    activation_post_process_4 = self.activation_post_process_4(pool2);  pool2 = None
    fl1 = self.fl1(activation_post_process_4);  activation_post_process_4 = None
    activation_post_process_5 = self.activation_post_process_5(fl1);  fl1 = None
    fc1 = self.fc1(activation_post_process_5);  activation_post_process_5 = None
    activation_post_process_6 = self.activation_post_process_6(fc1);  fc1 = None
    fc2 = self.fc2(activation_post_process_6);  activation_post_process_6 = None
    activation_post_process_7 = self.activation_post_process_7(fc2);  fc2 = None
    return activation_post_process_7
    
# To see more debug info, please use `graph_module.print_readable()`
'''


# device = 'cuda:0'
device='cpu'
prepared_fx.to(device)
lr = 0.9
optimizer = torch.optim.Adadelta(prepared_fx.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
max_epochs = 5
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=int(((len(train_loader)-1)//batch_size + 1)),
    epochs=max_epochs,
    cycle_momentum=False)
log_interval = 60

# train qat model
for epoch in range(1, max_epochs + 1):
    train(prepared_fx, device, train_loader, optimizer, criterion, epoch, log_interval)
    test(prepared_fx, device, test_loader, criterion)
    scheduler.step()


# Test set: Average loss: 0.0049, Accuracy: 9028/10000 (90%)

# freeze observers for a pre-conversion test
torch.ao.quantization.fake_quantize.disable_observer(prepared_fx)
test(prepared_fx.cpu(), 'cpu', test_loader, criterion)
# Test set: Average loss: 0.0049, Accuracy: 9028/10000 (90%)


quantized_fx = convert_fx(prepared_fx.eval().cpu())
print(quantized_fx)
'''
GraphModule(
  (conv1): QuantizedConvReLU2d(1, 32, kernel_size=(1, 1), stride=(1, 1), scale=0.01393101830035448, zero_point=0)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): QuantizedConvReLU2d(32, 64, kernel_size=(1, 1), stride=(1, 1), scale=0.03298066183924675, zero_point=0)
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): QuantizedLinearReLU(in_features=3136, out_features=10, scale=0.12397158890962601, zero_point=0, qscheme=torch.per_channel_affine)
  (fc2): QuantizedLinear(in_features=10, out_features=10, scale=0.1387060433626175, zero_point=158, qscheme=torch.per_channel_affine)
)



def forward(self, x):
    conv1_input_scale_0 = self.conv1_input_scale_0
    conv1_input_zero_point_0 = self.conv1_input_zero_point_0
    quantize_per_tensor = torch.quantize_per_tensor(x, conv1_input_scale_0, conv1_input_zero_point_0, torch.quint8);  x = conv1_input_scale_0 = conv1_input_zero_point_0 = None
    conv1 = self.conv1(quantize_per_tensor);  quantize_per_tensor = None
    pool1 = self.pool1(conv1);  conv1 = None
    conv2 = self.conv2(pool1);  pool1 = None
    pool2 = self.pool2(conv2);  conv2 = None
    dequantize_4 = pool2.dequantize();  pool2 = None
    fl1 = self.fl1(dequantize_4);  dequantize_4 = None
    fl1_scale_0 = self.fl1_scale_0
    fl1_zero_point_0 = self.fl1_zero_point_0
    quantize_per_tensor_5 = torch.quantize_per_tensor(fl1, fl1_scale_0, fl1_zero_point_0, torch.quint8);  fl1 = fl1_scale_0 = fl1_zero_point_0 = None
    fc1 = self.fc1(quantize_per_tensor_5);  quantize_per_tensor_5 = None
    fc2 = self.fc2(fc1);  fc1 = None
    dequantize_7 = fc2.dequantize();  fc2 = None
    return dequantize_7
    
# To see more debug info, please use `graph_module.print_readable()`
'''

test(quantized_fx, 'cpu', test_loader, criterion)
# Test set: Average loss: 0.0048, Accuracy: 9053/10000 (91%)
# when tested with converted quantization might lose some accuracy
```


##### PT2E PTQ static example

Sources:

- https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html
- https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html


*\*Note: For PT2E models, there is no `model.train()` or `model.eval()`, and instead we must call a different method. Instead we must call `torch.ao.quantization.move_exported_model_to_train(model)` or `torch.ao.quantization.move_exported_model_to_eval(model)`.*

Also, for training a PT2E model, the export method is different.

```python
example_inputs = (torch.rand(2, 3, 224, 224),)
# for pytorch 2.5+
exported_model = torch.export.export_for_training(float_model, example_inputs).module()
# for pytorch 2.4 and before
# from torch._export import capture_pre_autograd_graph
# exported_model = capture_pre_autograd_graph(model_to_quantize, example_inputs)
```

```python
import torch
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig
from torch.ao.quantization.quantizer import QuantizationSpec
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch._export import capture_pre_autograd_graph

# data imports
import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transformsv2
from torch.utils.data import DataLoader

import copy



## Define training and testing cycles
def train(
        model,
        device,
        train_loader,
        optimizer,
        criterion,
        epoch,
        log_interval=1,
        dry_run=False,
    ):
    if hasattr(model, 'graph'):
        torch.ao.quantization.move_exported_model_to_train(model)
    else:
        model.train()
    n_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            item = batch_idx * len(data)
            item_percent = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{item}/{n_data} ({item_percent:.0f}%)]\tLoss: {loss.item():.6f}')
            if dry_run:
                break


def test(
        model,
        device,
        test_loader,
        criterion,
    ):
    if hasattr(model, 'graph'):
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()
    test_loss = 0
    correct = 0
    n_data = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # sum up batch loss
            pred = torch.argmax(output,dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= n_data
    test_percent = 100. * correct / n_data
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n_data} ({test_percent:.0f}%)\n')


# test a few batches for calibration
def calibrate_ptq(
        prepared_model,
        device,
        test_loader,
        criterion,
        batch_size,
        n_batches=32,
    ):
    if hasattr(model, 'graph'):
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()
    cal_loss = 0
    correct = 0
    n_data = (len(test_loader.dataset) / batch_size) * n_batches
    with torch.no_grad():
        for (i,(data, target)) in enumerate(test_loader):
            if i<=n_batches:
                data, target = data.to(device), target.to(device)
                output = prepared_model(data)
                cal_loss += criterion(output, target).item()
                # sum up batch loss
                pred = torch.argmax(output,dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                break
    cal_loss /= n_data
    cal_percent = 100. * correct / n_data
    print(f'\nCalibrate set: Average loss: {cal_loss:.4f}, Accuracy: {correct}/{n_data} ({cal_percent:.0f}%)\n')



class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool1 = torch.nn.AdaptiveMaxPool2d((14,14))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(32,64,1,1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.act2 = torch.nn.ReLU()
        # self.pool2 = torch.nn.AdaptiveMaxPool2d((7,7))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool2 = torch.nn.MaxPool2d(2,2)
        self.fl1 = torch.nn.Flatten(start_dim=1)
        self.fc1 = torch.nn.Linear(3136, 10)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        return x



input_output_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(),
    )
weight_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False,
        observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
            ),
    )
bias_spec = None
quantization_config = QuantizationConfig(
        input_output_spec,
        input_output_spec,
        weight_spec,
        bias_spec,
        is_qat=True, # irrelevant in this case
    )
quantizer = XNNPACKQuantizer().set_global(quantization_config)


model = ExampleModel()
print(model)
'''
ExampleModel(
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)
'''

# in the case of PT2E PTQ, it's easier to run into complications during exporting, so we train before tracing

# however, we can still train a PT2E model if we use
# # for pytorch 2.5+
# exported_model = torch.export.export_for_training(model, example_inputs).module()
# # for pytorch 2.4 and before
# # from torch._export import capture_pre_autograd_graph
# # exported_model = capture_pre_autograd_graph(model, example_inputs)

## Load datasets
batch_size = 64
transform=transformsv2.Compose([
        transformsv2.ToTensor(),
        transformsv2.Normalize((0.1307,), (0.3081,)),
        ])
dataset1 = datasets.MNIST(
    '../data',
    train=True,
    download=True,
    transform=transform,
    )
dataset2 = datasets.MNIST(
    '../data',
    train=False,
    transform=transform,
    )
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

# device = 'cuda:0'
device='cpu'
model.to(device)
lr = 0.9
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
max_epochs = 20
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=int(((len(train_loader)-1)//batch_size + 1)),
    epochs=max_epochs,
    cycle_momentum=False)
log_interval = 60


# train float model
for epoch in range(1, max_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch, log_interval)
    test(model, device, test_loader, criterion)
    scheduler.step()

# Test set: Average loss: 0.0051, Accuracy: 9025/10000 (90%)


# in the case of PT2E PTQ, the batch size gets hard coded unless
# we add an exception as dynamic shapes
##  example_inputs = (torch.randn(1,1,28,28),)
##  # for pt2e ptq: model.eval()
##  # for pt2e qat: model.train()
##  pt2e_traced_model = capture_pre_autograd_graph(
##      model.eval(),
##      example_inputs,
##      dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
##      )
##  print(pt2e_traced_model)
# however, convolution and batch normalization don't allow for this.

example_inputs = (torch.randn(1,1,28,28),)
pt2e_traced_model = capture_pre_autograd_graph(
    model.eval(),
    example_inputs,
    )
print(pt2e_traced_model)
'''
GraphModule()



def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    arg0_1 = arg0
    _param_constant0 = self.conv1_weight
    _param_constant1 = self.conv1_bias
    conv2d = torch.ops.aten.conv2d.default(arg0_1, _param_constant0, _param_constant1);  arg0_1 = _param_constant0 = _param_constant1 = None
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
    _param_constant2 = self.bn1_weight
    _param_constant3 = self.bn1_bias
    _tensor_constant0 = self.bn1_running_mean
    _tensor_constant1 = self.bn1_running_var
    _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(conv2d, _param_constant2, _param_constant3, _tensor_constant0, _tensor_constant1, 0.1, 1e-05);  conv2d = _param_constant2 = _param_constant3 = _tensor_constant0 = _tensor_constant1 = None
    getitem = _native_batch_norm_legit_no_training[0]
    getitem_1 = _native_batch_norm_legit_no_training[1]
    getitem_2 = _native_batch_norm_legit_no_training[2];  _native_batch_norm_legit_no_training = None
    relu = torch.ops.aten.relu.default(getitem);  getitem = None
    max_pool2d = torch.ops.aten.max_pool2d.default(relu, [2, 2], [2, 2]);  relu = None
    _param_constant4 = self.conv2_weight
    _param_constant5 = self.conv2_bias
    conv2d_1 = torch.ops.aten.conv2d.default(max_pool2d, _param_constant4, _param_constant5);  max_pool2d = _param_constant4 = _param_constant5 = None
    empty_1 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
    _param_constant6 = self.bn2_weight
    _param_constant7 = self.bn2_bias
    _tensor_constant2 = self.bn2_running_mean
    _tensor_constant3 = self.bn2_running_var
    _native_batch_norm_legit_no_training_1 = torch.ops.aten._native_batch_norm_legit_no_training.default(conv2d_1, _param_constant6, _param_constant7, _tensor_constant2, _tensor_constant3, 0.1, 1e-05);  conv2d_1 = _param_constant6 = _param_constant7 = _tensor_constant2 = _tensor_constant3 = None
    getitem_3 = _native_batch_norm_legit_no_training_1[0]
    getitem_4 = _native_batch_norm_legit_no_training_1[1]
    getitem_5 = _native_batch_norm_legit_no_training_1[2];  _native_batch_norm_legit_no_training_1 = None
    relu_1 = torch.ops.aten.relu.default(getitem_3);  getitem_3 = None
    max_pool2d_1 = torch.ops.aten.max_pool2d.default(relu_1, [2, 2], [2, 2]);  relu_1 = None
    flatten = torch.ops.aten.flatten.using_ints(max_pool2d_1, 1);  max_pool2d_1 = None
    _param_constant8 = self.fc1_weight
    _param_constant9 = self.fc1_bias
    linear = torch.ops.aten.linear.default(flatten, _param_constant8, _param_constant9);  flatten = _param_constant8 = _param_constant9 = None
    relu_2 = torch.ops.aten.relu.default(linear);  linear = None
    _param_constant10 = self.fc2_weight
    _param_constant11 = self.fc2_bias
    linear_1 = torch.ops.aten.linear.default(relu_2, _param_constant10, _param_constant11);  relu_2 = _param_constant10 = _param_constant11 = None
    return pytree.tree_unflatten([linear_1], self._out_spec)
    
# To see more debug info, please use `graph_module.print_readable()`
'''


# Prepare (and fuse) PT2E model
pt2e_prepared_model = prepare_pt2e(pt2e_traced_model, quantizer)

# calibrate for 32 batches
calibrate_ptq(pt2e_prepared_model, device, test_loader, criterion, batch_size, n_batches=32)
print(pt2e_prepared_model)
'''
GraphModule(
  (activation_post_process_0): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0127]), zero_point=tensor([-95], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.4242129623889923, max_val=2.819929838180542)
  )
  (activation_post_process_1): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0084]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-1.0703085660934448, max_val=1.0625624656677246)
  )
  (activation_post_process_2): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0136]), zero_point=tensor([-128], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.4710187911987305)
  )
  (activation_post_process_3): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0136]), zero_point=tensor([-128], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=3.4710187911987305)
  )
  (activation_post_process_4): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0586]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-7.499833106994629, max_val=6.473066329956055)
  )
  (activation_post_process_5): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0266]), zero_point=tensor([-128], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=6.7827911376953125)
  )
  (activation_post_process_6): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0266]), zero_point=tensor([-128], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=6.7827911376953125)
  )
  (activation_post_process_7): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0266]), zero_point=tensor([-128], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.0, max_val=6.7827911376953125)
  )
  (activation_post_process_8): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0019]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.18611857295036316, max_val=0.24004152417182922)
  )
  (activation_post_process_9): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0992]), zero_point=tensor([-128], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=0.18618673086166382, max_val=25.291959762573242)
  )
  (activation_post_process_10): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0047]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-0.5963500738143921, max_val=0.3838652968406677)
  )
  (activation_post_process_11): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([0.0705]), zero_point=tensor([63], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=-13.436151504516602, max_val=4.531009674072266)
  )
)



def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    arg0_1 = arg0
    activation_post_process_0 = self.activation_post_process_0(arg0_1);  arg0_1 = None
    _param_constant0 = self.conv1_weight
    activation_post_process_1 = self.activation_post_process_1(_param_constant0);  _param_constant0 = None
    _param_constant1 = self.conv1_bias
    conv2d = torch.ops.aten.conv2d.default(activation_post_process_0, activation_post_process_1, _param_constant1);  activation_post_process_0 = activation_post_process_1 = _param_constant1 = None
    relu = torch.ops.aten.relu.default(conv2d);  conv2d = None
    activation_post_process_2 = self.activation_post_process_2(relu);  relu = None
    max_pool2d = torch.ops.aten.max_pool2d.default(activation_post_process_2, [2, 2], [2, 2]);  activation_post_process_2 = None
    activation_post_process_3 = self.activation_post_process_3(max_pool2d);  max_pool2d = None
    _param_constant4 = self.conv2_weight
    activation_post_process_4 = self.activation_post_process_4(_param_constant4);  _param_constant4 = None
    _param_constant5 = self.conv2_bias
    conv2d_1 = torch.ops.aten.conv2d.default(activation_post_process_3, activation_post_process_4, _param_constant5);  activation_post_process_3 = activation_post_process_4 = _param_constant5 = None
    relu_1 = torch.ops.aten.relu.default(conv2d_1);  conv2d_1 = None
    activation_post_process_5 = self.activation_post_process_5(relu_1);  relu_1 = None
    max_pool2d_1 = torch.ops.aten.max_pool2d.default(activation_post_process_5, [2, 2], [2, 2]);  activation_post_process_5 = None
    activation_post_process_6 = self.activation_post_process_6(max_pool2d_1);  max_pool2d_1 = None
    flatten = torch.ops.aten.flatten.using_ints(activation_post_process_6, 1);  activation_post_process_6 = None
    activation_post_process_7 = self.activation_post_process_7(flatten);  flatten = None
    _param_constant8 = self.fc1_weight
    activation_post_process_8 = self.activation_post_process_8(_param_constant8);  _param_constant8 = None
    _param_constant9 = self.fc1_bias
    linear = torch.ops.aten.linear.default(activation_post_process_7, activation_post_process_8, _param_constant9);  activation_post_process_7 = activation_post_process_8 = _param_constant9 = None
    relu_2 = torch.ops.aten.relu.default(linear);  linear = None
    activation_post_process_9 = self.activation_post_process_9(relu_2);  relu_2 = None
    _param_constant10 = self.fc2_weight
    activation_post_process_10 = self.activation_post_process_10(_param_constant10);  _param_constant10 = None
    _param_constant11 = self.fc2_bias
    linear_1 = torch.ops.aten.linear.default(activation_post_process_9, activation_post_process_10, _param_constant11);  activation_post_process_9 = activation_post_process_10 = _param_constant11 = None
    activation_post_process_11 = self.activation_post_process_11(linear_1);  linear_1 = None
    return pytree.tree_unflatten([activation_post_process_11], self._out_spec)
    
# To see more debug info, please use `graph_module.print_readable()`
'''


# freeze observers for a pre-conversion test
torch.ao.quantization.fake_quantize.disable_observer(pt2e_prepared_model)
test(pt2e_prepared_model.cpu(), 'cpu', test_loader, criterion)
# Test set: Average loss: 0.3565, Accuracy: 8948/10000 (89%)


pt2e_quantized_model = convert_pt2e(pt2e_prepared_model)
print(pt2e_quantized_model)
'''
0 ?
pt2e_quantized_model = convert_pt2e(pt2e_prepared_model)
print(pt2e_quantized_model)
GraphModule()



def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    arg0_1 = arg0
    quantize_per_tensor_default = torch.ops.quantized_decomposed.quantize_per_tensor.default(arg0_1, 0.012722395360469818, -95, -128, 127, torch.int8);  arg0_1 = None
    dequantize_per_tensor_default = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default, 0.012722395360469818, -95, -128, 127, torch.int8);  quantize_per_tensor_default = None
    _frozen_param0 = self._frozen_param0
    dequantize_per_tensor_default_1 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(_frozen_param0, 0.008394576609134674, 0, -128, 127, torch.int8);  _frozen_param0 = None
    _param_constant1 = self.conv1_bias
    conv2d = torch.ops.aten.conv2d.default(dequantize_per_tensor_default, dequantize_per_tensor_default_1, _param_constant1);  dequantize_per_tensor_default = dequantize_per_tensor_default_1 = _param_constant1 = None
    relu = torch.ops.aten.relu.default(conv2d);  conv2d = None
    quantize_per_tensor_default_2 = torch.ops.quantized_decomposed.quantize_per_tensor.default(relu, 0.013613359071314335, -128, -128, 127, torch.int8);  relu = None
    dequantize_per_tensor_default_2 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_2, 0.013613359071314335, -128, -128, 127, torch.int8);  quantize_per_tensor_default_2 = None
    max_pool2d = torch.ops.aten.max_pool2d.default(dequantize_per_tensor_default_2, [2, 2], [2, 2]);  dequantize_per_tensor_default_2 = None
    quantize_per_tensor_default_3 = torch.ops.quantized_decomposed.quantize_per_tensor.default(max_pool2d, 0.013613359071314335, -128, -128, 127, torch.int8);  max_pool2d = None
    dequantize_per_tensor_default_3 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_3, 0.013613359071314335, -128, -128, 127, torch.int8);  quantize_per_tensor_default_3 = None
    _frozen_param1 = self._frozen_param1
    dequantize_per_tensor_default_4 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(_frozen_param1, 0.05882222205400467, 0, -128, 127, torch.int8);  _frozen_param1 = None
    _param_constant5 = self.conv2_bias
    conv2d_1 = torch.ops.aten.conv2d.default(dequantize_per_tensor_default_3, dequantize_per_tensor_default_4, _param_constant5);  dequantize_per_tensor_default_3 = dequantize_per_tensor_default_4 = _param_constant5 = None
    relu_1 = torch.ops.aten.relu.default(conv2d_1);  conv2d_1 = None
    quantize_per_tensor_default_5 = torch.ops.quantized_decomposed.quantize_per_tensor.default(relu_1, 0.02676599845290184, -128, -128, 127, torch.int8);  relu_1 = None
    dequantize_per_tensor_default_5 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_5, 0.02676599845290184, -128, -128, 127, torch.int8);  quantize_per_tensor_default_5 = None
    max_pool2d_1 = torch.ops.aten.max_pool2d.default(dequantize_per_tensor_default_5, [2, 2], [2, 2]);  dequantize_per_tensor_default_5 = None
    quantize_per_tensor_default_6 = torch.ops.quantized_decomposed.quantize_per_tensor.default(max_pool2d_1, 0.02676599845290184, -128, -128, 127, torch.int8);  max_pool2d_1 = None
    dequantize_per_tensor_default_6 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_6, 0.02676599845290184, -128, -128, 127, torch.int8);  quantize_per_tensor_default_6 = None
    flatten = torch.ops.aten.flatten.using_ints(dequantize_per_tensor_default_6, 1);  dequantize_per_tensor_default_6 = None
    quantize_per_tensor_default_7 = torch.ops.quantized_decomposed.quantize_per_tensor.default(flatten, 0.02676599845290184, -128, -128, 127, torch.int8);  flatten = None
    dequantize_per_tensor_default_7 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_7, 0.02676599845290184, -128, -128, 127, torch.int8);  quantize_per_tensor_default_7 = None
    _frozen_param2 = self._frozen_param2
    dequantize_per_tensor_default_8 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(_frozen_param2, 0.001882678596302867, 0, -128, 127, torch.int8);  _frozen_param2 = None
    _param_constant9 = self.fc1_bias
    linear = torch.ops.aten.linear.default(dequantize_per_tensor_default_7, dequantize_per_tensor_default_8, _param_constant9);  dequantize_per_tensor_default_7 = dequantize_per_tensor_default_8 = _param_constant9 = None
    relu_2 = torch.ops.aten.relu.default(linear);  linear = None
    quantize_per_tensor_default_9 = torch.ops.quantized_decomposed.quantize_per_tensor.default(relu_2, 0.1014489233493805, -128, -128, 127, torch.int8);  relu_2 = None
    dequantize_per_tensor_default_9 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_9, 0.1014489233493805, -128, -128, 127, torch.int8);  quantize_per_tensor_default_9 = None
    _frozen_param3 = self._frozen_param3
    dequantize_per_tensor_default_10 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(_frozen_param3, 0.004677255637943745, 0, -128, 127, torch.int8);  _frozen_param3 = None
    _param_constant11 = self.fc2_bias
    linear_1 = torch.ops.aten.linear.default(dequantize_per_tensor_default_9, dequantize_per_tensor_default_10, _param_constant11);  dequantize_per_tensor_default_9 = dequantize_per_tensor_default_10 = _param_constant11 = None
    quantize_per_tensor_default_11 = torch.ops.quantized_decomposed.quantize_per_tensor.default(linear_1, 0.07243377715349197, 70, -128, 127, torch.int8);  linear_1 = None
    dequantize_per_tensor_default_11 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_11, 0.07243377715349197, 70, -128, 127, torch.int8);  quantize_per_tensor_default_11 = None
    return pytree.tree_unflatten([dequantize_per_tensor_default_11], self._out_spec)
    
# To see more debug info, please use `graph_module.print_readable()`
'''

test(pt2e_quantized_model, 'cpu', test_loader, criterion)
# Test set: Average loss: 0.3718, Accuracy: 8941/10000 (89%)
```

##### PT2E PTQ dynamic example

In progress...


```python
import torch
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig
from torch.ao.quantization.quantizer import QuantizationSpec
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, PlaceholderObserver
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize, FakeQuantize
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e


# data imports
import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transformsv2
from torch.utils.data import DataLoader

import copy


## Define training and testing cycles
def train(
        model,
        device,
        train_loader,
        optimizer,
        criterion,
        epoch,
        log_interval=1,
        dry_run=False,
    ):
    if hasattr(model, 'graph'):
        torch.ao.quantization.move_exported_model_to_train(model)
    else:
        model.train()
    n_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            item = batch_idx * len(data)
            item_percent = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{item}/{n_data} ({item_percent:.0f}%)]\tLoss: {loss.item():.6f}')
            if dry_run:
                break


def test(
        model,
        device,
        test_loader,
        criterion,
    ):
    if hasattr(model, 'graph'):
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()
    test_loss = 0
    correct = 0
    n_data = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # sum up batch loss
            pred = torch.argmax(output,dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= n_data
    test_percent = 100. * correct / n_data
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n_data} ({test_percent:.0f}%)\n')



class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool1 = torch.nn.AdaptiveMaxPool2d((14,14))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(32,64,1,1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.act2 = torch.nn.ReLU()
        # self.pool2 = torch.nn.AdaptiveMaxPool2d((7,7))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool2 = torch.nn.MaxPool2d(2,2)
        self.fl1 = torch.nn.Flatten(start_dim=1)
        self.fc1 = torch.nn.Linear(3136, 10)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        return x



input_output_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=True,
        observer_or_fake_quant_ctr=PlaceholderObserver.with_args(),
    )
weight_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False, # only needed in activation for dynamic
        observer_or_fake_quant_ctr=FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver.with_args(
                    averaging_constant=1, # necessary for dynamic
                    ),
            ),
    )
bias_spec = None
quantization_config = QuantizationConfig(
        input_output_spec,
        None, # different in dynamic
        weight_spec,
        bias_spec,
        is_qat=False, # relevant for dynamic
    )
quantizer = XNNPACKQuantizer().set_global(quantization_config)


model = ExampleModel()
print(model)
'''
ExampleModel(
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)
'''


# in the case of PT2E PTQ, it's easier to run into complications during exporting, so we train before tracing

# however, we can still train a PT2E model if we use
# # for pytorch 2.5+
# exported_model = torch.export.export_for_training(model, example_inputs).module()
# # for pytorch 2.4 and before
# # from torch._export import capture_pre_autograd_graph
# # exported_model = capture_pre_autograd_graph(model, example_inputs)

## Load datasets
batch_size = 64
transform=transformsv2.Compose([
        transformsv2.ToTensor(),
        transformsv2.Normalize((0.1307,), (0.3081,)),
        ])
dataset1 = datasets.MNIST(
    '../data',
    train=True,
    download=True,
    transform=transform,
    )
dataset2 = datasets.MNIST(
    '../data',
    train=False,
    transform=transform,
    )
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)


# device = 'cuda:0'
device='cpu'
model.to(device)
lr = 0.9
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
max_epochs = 20
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=int(((len(train_loader)-1)//batch_size + 1)),
    epochs=max_epochs,
    cycle_momentum=False)
log_interval = 60


# train float model
for epoch in range(1, max_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch, log_interval)
    test(model, device, test_loader, criterion)
    scheduler.step()


# Test set: Average loss: 0.0051, Accuracy: 9025/10000 (90%)


# in the case of PT2E PTQ, the batch size gets hard coded unless
# we add an exception as dynamic shapes
##  example_inputs = (torch.randn(1,1,28,28),)
##  # for pt2e ptq: model.eval()
##  # for pt2e qat: model.train()
##  pt2e_traced_model = capture_pre_autograd_graph(
##      model.eval(),
##      example_inputs,
##      dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
##      )
##  print(pt2e_traced_model)
# however, convolution and batch normalization don't allow for this.

example_inputs = (torch.randn(1,1,28,28),)
pt2e_traced_model = capture_pre_autograd_graph(
    model.eval(),
    example_inputs,
    )
print(pt2e_traced_model)
'''
GraphModule()



def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    arg0_1 = arg0
    _param_constant0 = self.conv1_weight
    _param_constant1 = self.conv1_bias
    conv2d = torch.ops.aten.conv2d.default(arg0_1, _param_constant0, _param_constant1);  arg0_1 = _param_constant0 = _param_constant1 = None
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
    _param_constant2 = self.bn1_weight
    _param_constant3 = self.bn1_bias
    _tensor_constant0 = self.bn1_running_mean
    _tensor_constant1 = self.bn1_running_var
    _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(conv2d, _param_constant2, _param_constant3, _tensor_constant0, _tensor_constant1, 0.1, 1e-05);  conv2d = _param_constant2 = _param_constant3 = _tensor_constant0 = _tensor_constant1 = None
    getitem = _native_batch_norm_legit_no_training[0]
    getitem_1 = _native_batch_norm_legit_no_training[1]
    getitem_2 = _native_batch_norm_legit_no_training[2];  _native_batch_norm_legit_no_training = None
    relu = torch.ops.aten.relu.default(getitem);  getitem = None
    max_pool2d = torch.ops.aten.max_pool2d.default(relu, [2, 2], [2, 2]);  relu = None
    _param_constant4 = self.conv2_weight
    _param_constant5 = self.conv2_bias
    conv2d_1 = torch.ops.aten.conv2d.default(max_pool2d, _param_constant4, _param_constant5);  max_pool2d = _param_constant4 = _param_constant5 = None
    empty_1 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
    _param_constant6 = self.bn2_weight
    _param_constant7 = self.bn2_bias
    _tensor_constant2 = self.bn2_running_mean
    _tensor_constant3 = self.bn2_running_var
    _native_batch_norm_legit_no_training_1 = torch.ops.aten._native_batch_norm_legit_no_training.default(conv2d_1, _param_constant6, _param_constant7, _tensor_constant2, _tensor_constant3, 0.1, 1e-05);  conv2d_1 = _param_constant6 = _param_constant7 = _tensor_constant2 = _tensor_constant3 = None
    getitem_3 = _native_batch_norm_legit_no_training_1[0]
    getitem_4 = _native_batch_norm_legit_no_training_1[1]
    getitem_5 = _native_batch_norm_legit_no_training_1[2];  _native_batch_norm_legit_no_training_1 = None
    relu_1 = torch.ops.aten.relu.default(getitem_3);  getitem_3 = None
    max_pool2d_1 = torch.ops.aten.max_pool2d.default(relu_1, [2, 2], [2, 2]);  relu_1 = None
    flatten = torch.ops.aten.flatten.using_ints(max_pool2d_1, 1);  max_pool2d_1 = None
    _param_constant8 = self.fc1_weight
    _param_constant9 = self.fc1_bias
    linear = torch.ops.aten.linear.default(flatten, _param_constant8, _param_constant9);  flatten = _param_constant8 = _param_constant9 = None
    relu_2 = torch.ops.aten.relu.default(linear);  linear = None
    _param_constant10 = self.fc2_weight
    _param_constant11 = self.fc2_bias
    linear_1 = torch.ops.aten.linear.default(relu_2, _param_constant10, _param_constant11);  relu_2 = _param_constant10 = _param_constant11 = None
    return pytree.tree_unflatten([linear_1], self._out_spec)
    
# To see more debug info, please use `graph_module.print_readable()`
'''



# Prepare (and fuse) PT2E model
pt2e_prepared_model = prepare_pt2e(pt2e_traced_model, quantizer)
print(pt2e_prepared_model)
'''
GraphModule(
  (activation_post_process_0): PlaceholderObserver(dtype=torch.int8, is_dynamic=True)
  (activation_post_process_1): FakeQuantize(
    fake_quant_enabled=tensor([1], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.int8, qscheme=torch.per_tensor_symmetric, ch_axis=-1, scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32)
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (activation_post_process_2): PlaceholderObserver(dtype=torch.int8, is_dynamic=True)
  (activation_post_process_3): FakeQuantize(
    fake_quant_enabled=tensor([1], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.int8, qscheme=torch.per_tensor_symmetric, ch_axis=-1, scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32)
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
)



def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    arg0_1 = arg0
    _param_constant0 = self.conv1_weight
    _param_constant1 = self.conv1_bias
    conv2d = torch.ops.aten.conv2d.default(arg0_1, _param_constant0, _param_constant1);  arg0_1 = _param_constant0 = _param_constant1 = None
    relu = torch.ops.aten.relu.default(conv2d);  conv2d = None
    max_pool2d = torch.ops.aten.max_pool2d.default(relu, [2, 2], [2, 2]);  relu = None
    _param_constant4 = self.conv2_weight
    _param_constant5 = self.conv2_bias
    conv2d_1 = torch.ops.aten.conv2d.default(max_pool2d, _param_constant4, _param_constant5);  max_pool2d = _param_constant4 = _param_constant5 = None
    relu_1 = torch.ops.aten.relu.default(conv2d_1);  conv2d_1 = None
    max_pool2d_1 = torch.ops.aten.max_pool2d.default(relu_1, [2, 2], [2, 2]);  relu_1 = None
    flatten = torch.ops.aten.flatten.using_ints(max_pool2d_1, 1);  max_pool2d_1 = None
    activation_post_process_0 = self.activation_post_process_0(flatten);  flatten = None
    _param_constant8 = self.fc1_weight
    activation_post_process_1 = self.activation_post_process_1(_param_constant8);  _param_constant8 = None
    _param_constant9 = self.fc1_bias
    linear = torch.ops.aten.linear.default(activation_post_process_0, activation_post_process_1, _param_constant9);  activation_post_process_0 = activation_post_process_1 = _param_constant9 = None
    relu_2 = torch.ops.aten.relu.default(linear);  linear = None
    activation_post_process_2 = self.activation_post_process_2(relu_2);  relu_2 = None
    _param_constant10 = self.fc2_weight
    activation_post_process_3 = self.activation_post_process_3(_param_constant10);  _param_constant10 = None
    _param_constant11 = self.fc2_bias
    linear_1 = torch.ops.aten.linear.default(activation_post_process_2, activation_post_process_3, _param_constant11);  activation_post_process_2 = activation_post_process_3 = _param_constant11 = None
    return pytree.tree_unflatten([linear_1], self._out_spec)
    
# To see more debug info, please use `graph_module.print_readable()`
'''


# freeze observers for a pre-conversion test
torch.ao.quantization.fake_quantize.disable_observer(pt2e_prepared_model)
test(pt2e_prepared_model.cpu(), 'cpu', test_loader, criterion)
# Test set: Average loss: 0.0051, Accuracy: 9010/10000 (90%)


pt2e_quantized_model = convert_pt2e(pt2e_prepared_model)
print(pt2e_quantized_model)
'''
GraphModule()



def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    arg0_1 = arg0
    _param_constant0 = self.conv1_weight
    _param_constant1 = self.conv1_bias
    conv2d = torch.ops.aten.conv2d.default(arg0_1, _param_constant0, _param_constant1);  arg0_1 = _param_constant0 = _param_constant1 = None
    relu = torch.ops.aten.relu.default(conv2d);  conv2d = None
    max_pool2d = torch.ops.aten.max_pool2d.default(relu, [2, 2], [2, 2]);  relu = None
    _param_constant4 = self.conv2_weight
    _param_constant5 = self.conv2_bias
    conv2d_1 = torch.ops.aten.conv2d.default(max_pool2d, _param_constant4, _param_constant5);  max_pool2d = _param_constant4 = _param_constant5 = None
    relu_1 = torch.ops.aten.relu.default(conv2d_1);  conv2d_1 = None
    max_pool2d_1 = torch.ops.aten.max_pool2d.default(relu_1, [2, 2], [2, 2]);  relu_1 = None
    flatten = torch.ops.aten.flatten.using_ints(max_pool2d_1, 1);  max_pool2d_1 = None
    choose_qparams_tensor = torch.ops.quantized_decomposed.choose_qparams.tensor(flatten, -128, 127, 1.1920928955078125e-07, torch.int8)
    getitem_6 = choose_qparams_tensor[0]
    getitem_7 = choose_qparams_tensor[1];  choose_qparams_tensor = None
    quantize_per_tensor_tensor = torch.ops.quantized_decomposed.quantize_per_tensor.tensor(flatten, getitem_6, getitem_7, -128, 127, torch.int8);  flatten = None
    dequantize_per_tensor_tensor = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(quantize_per_tensor_tensor, getitem_6, getitem_7, -128, 127, torch.int8);  quantize_per_tensor_tensor = getitem_6 = getitem_7 = None
    _frozen_param0 = self._frozen_param0
    dequantize_per_tensor_default = torch.ops.quantized_decomposed.dequantize_per_tensor.default(_frozen_param0, 0.001882678596302867, 0, -128, 127, torch.int8);  _frozen_param0 = None
    _param_constant9 = self.fc1_bias
    linear = torch.ops.aten.linear.default(dequantize_per_tensor_tensor, dequantize_per_tensor_default, _param_constant9);  dequantize_per_tensor_tensor = dequantize_per_tensor_default = _param_constant9 = None
    relu_2 = torch.ops.aten.relu.default(linear);  linear = None
    choose_qparams_tensor_1 = torch.ops.quantized_decomposed.choose_qparams.tensor(relu_2, -128, 127, 1.1920928955078125e-07, torch.int8)
    getitem_8 = choose_qparams_tensor_1[0]
    getitem_9 = choose_qparams_tensor_1[1];  choose_qparams_tensor_1 = None
    quantize_per_tensor_tensor_1 = torch.ops.quantized_decomposed.quantize_per_tensor.tensor(relu_2, getitem_8, getitem_9, -128, 127, torch.int8);  relu_2 = None
    dequantize_per_tensor_tensor_1 = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(quantize_per_tensor_tensor_1, getitem_8, getitem_9, -128, 127, torch.int8);  quantize_per_tensor_tensor_1 = getitem_8 = getitem_9 = None
    _frozen_param1 = self._frozen_param1
    dequantize_per_tensor_default_1 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(_frozen_param1, 0.004677255637943745, 0, -128, 127, torch.int8);  _frozen_param1 = None
    _param_constant11 = self.fc2_bias
    linear_1 = torch.ops.aten.linear.default(dequantize_per_tensor_tensor_1, dequantize_per_tensor_default_1, _param_constant11);  dequantize_per_tensor_tensor_1 = dequantize_per_tensor_default_1 = _param_constant11 = None
    return pytree.tree_unflatten([linear_1], self._out_spec)
    
# To see more debug info, please use `graph_module.print_readable()`
'''

test(pt2e_quantized_model, 'cpu', test_loader, criterion)
# Test set: Average loss: 0.0051, Accuracy: 9001/10000 (90%)
```


##### PT2E QAT static example

https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html
In progress...


```python
import torch
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig
from torch.ao.quantization.quantizer import QuantizationSpec
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch._export import capture_pre_autograd_graph

# data imports
import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transformsv2
from torch.utils.data import DataLoader

import copy



## Define training and testing cycles
def train(
        model,
        device,
        train_loader,
        optimizer,
        criterion,
        epoch,
        log_interval=1,
        dry_run=False,
    ):
    if hasattr(model, 'graph'):
        torch.ao.quantization.move_exported_model_to_train(model)
    else:
        model.train()
    n_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            item = batch_idx * len(data)
            item_percent = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{item}/{n_data} ({item_percent:.0f}%)]\tLoss: {loss.item():.6f}')
            if dry_run:
                break


def test(
        model,
        device,
        test_loader,
        criterion,
    ):
    if hasattr(model, 'graph'):
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()
    test_loss = 0
    correct = 0
    n_data = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # sum up batch loss
            pred = torch.argmax(output,dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= n_data
    test_percent = 100. * correct / n_data
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n_data} ({test_percent:.0f}%)\n')


# test a few batches for calibration
def calibrate_ptq(
        prepared_model,
        device,
        test_loader,
        criterion,
        batch_size,
        n_batches=32,
    ):
    if hasattr(model, 'graph'):
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()
    cal_loss = 0
    correct = 0
    n_data = (len(test_loader.dataset) / batch_size) * n_batches
    with torch.no_grad():
        for (i,(data, target)) in enumerate(test_loader):
            if i<=n_batches:
                data, target = data.to(device), target.to(device)
                output = prepared_model(data)
                cal_loss += criterion(output, target).item()
                # sum up batch loss
                pred = torch.argmax(output,dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                break
    cal_loss /= n_data
    cal_percent = 100. * correct / n_data
    print(f'\nCalibrate set: Average loss: {cal_loss:.4f}, Accuracy: {correct}/{n_data} ({cal_percent:.0f}%)\n')



class ExampleModel(torch.nn.Module):
    '''
    Expects mnist input of shape (batch,1,28,28)
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,32,1,1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.act1 = torch.nn.ReLU()
        # self.pool1 = torch.nn.AdaptiveMaxPool2d((14,14))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(32,64,1,1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.act2 = torch.nn.ReLU()
        # self.pool2 = torch.nn.AdaptiveMaxPool2d((7,7))
        # not supported once quantized, so replace with manual MaxPool2d
        self.pool2 = torch.nn.MaxPool2d(2,2)
        self.fl1 = torch.nn.Flatten(start_dim=1)
        self.fc1 = torch.nn.Linear(3136, 10)
        self.fc1act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.fc1act(x)
        x = self.fc2(x)
        return x



input_output_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(),
    )
weight_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False,
        observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
            ),
    )
bias_spec = None
quantization_config = QuantizationConfig(
        input_output_spec,
        input_output_spec,
        weight_spec,
        bias_spec,
        is_qat=True,
    )
quantizer = XNNPACKQuantizer().set_global(quantization_config)


model = ExampleModel()
print(model)
'''
ExampleModel(
  (conv1): Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fl1): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
  (fc1act): ReLU()
  (fc2): Linear(in_features=10, out_features=10, bias=True)
)
'''

# in the case of PT2E PTQ, it's easier to run into complications during exporting, so we train before tracing

# however, we can still train a PT2E model if we use
# # for pytorch 2.5+
# exported_model = torch.export.export_for_training(model, example_inputs).module()
# # for pytorch 2.4 and before
# # from torch._export import capture_pre_autograd_graph
# # exported_model = capture_pre_autograd_graph(model, example_inputs)

## Load datasets
batch_size = 64
transform=transformsv2.Compose([
        transformsv2.ToTensor(),
        transformsv2.Normalize((0.1307,), (0.3081,)),
        ])
dataset1 = datasets.MNIST(
    '../data',
    train=True,
    download=True,
    transform=transform,
    )
dataset2 = datasets.MNIST(
    '../data',
    train=False,
    transform=transform,
    )
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)


# device = 'cuda:0'
device='cpu'
model.to(device)
lr = 0.9
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
max_epochs = 20
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=int(((len(train_loader)-1)//batch_size + 1)),
    epochs=max_epochs,
    cycle_momentum=False)
log_interval = 60


# train float model
for epoch in range(1, max_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch, log_interval)
    test(model, device, test_loader, criterion)
    scheduler.step()


# Test set: Average loss: 0.0051, Accuracy: 9025/10000 (90%)


# in the case of PT2E PTQ, the batch size gets hard coded unless
# we add an exception as dynamic shapes
##  example_inputs = (torch.randn(1,1,28,28),)
##  # for pt2e ptq: model.eval()
##  # for pt2e qat: model.train()
##  pt2e_traced_model = capture_pre_autograd_graph(
##      model.eval(),
##      example_inputs,
##      dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
##      )
##  print(pt2e_traced_model)
# however, convolution and batch normalization don't allow for this.

example_inputs = (torch.randn(1,1,28,28),)
pt2e_traced_model = capture_pre_autograd_graph(
    model.eval(),
    example_inputs,
    )
print(pt2e_traced_model)
'''
GraphModule()



def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    arg0_1 = arg0
    _param_constant0 = self.conv1_weight
    _param_constant1 = self.conv1_bias
    conv2d = torch.ops.aten.conv2d.default(arg0_1, _param_constant0, _param_constant1);  arg0_1 = _param_constant0 = _param_constant1 = None
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
    _param_constant2 = self.bn1_weight
    _param_constant3 = self.bn1_bias
    _tensor_constant0 = self.bn1_running_mean
    _tensor_constant1 = self.bn1_running_var
    _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(conv2d, _param_constant2, _param_constant3, _tensor_constant0, _tensor_constant1, 0.1, 1e-05);  conv2d = _param_constant2 = _param_constant3 = _tensor_constant0 = _tensor_constant1 = None
    getitem = _native_batch_norm_legit_no_training[0]
    getitem_1 = _native_batch_norm_legit_no_training[1]
    getitem_2 = _native_batch_norm_legit_no_training[2];  _native_batch_norm_legit_no_training = None
    relu = torch.ops.aten.relu.default(getitem);  getitem = None
    max_pool2d = torch.ops.aten.max_pool2d.default(relu, [2, 2], [2, 2]);  relu = None
    _param_constant4 = self.conv2_weight
    _param_constant5 = self.conv2_bias
    conv2d_1 = torch.ops.aten.conv2d.default(max_pool2d, _param_constant4, _param_constant5);  max_pool2d = _param_constant4 = _param_constant5 = None
    empty_1 = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
    _param_constant6 = self.bn2_weight
    _param_constant7 = self.bn2_bias
    _tensor_constant2 = self.bn2_running_mean
    _tensor_constant3 = self.bn2_running_var
    _native_batch_norm_legit_no_training_1 = torch.ops.aten._native_batch_norm_legit_no_training.default(conv2d_1, _param_constant6, _param_constant7, _tensor_constant2, _tensor_constant3, 0.1, 1e-05);  conv2d_1 = _param_constant6 = _param_constant7 = _tensor_constant2 = _tensor_constant3 = None
    getitem_3 = _native_batch_norm_legit_no_training_1[0]
    getitem_4 = _native_batch_norm_legit_no_training_1[1]
    getitem_5 = _native_batch_norm_legit_no_training_1[2];  _native_batch_norm_legit_no_training_1 = None
    relu_1 = torch.ops.aten.relu.default(getitem_3);  getitem_3 = None
    max_pool2d_1 = torch.ops.aten.max_pool2d.default(relu_1, [2, 2], [2, 2]);  relu_1 = None
    flatten = torch.ops.aten.flatten.using_ints(max_pool2d_1, 1);  max_pool2d_1 = None
    _param_constant8 = self.fc1_weight
    _param_constant9 = self.fc1_bias
    linear = torch.ops.aten.linear.default(flatten, _param_constant8, _param_constant9);  flatten = _param_constant8 = _param_constant9 = None
    relu_2 = torch.ops.aten.relu.default(linear);  linear = None
    _param_constant10 = self.fc2_weight
    _param_constant11 = self.fc2_bias
    linear_1 = torch.ops.aten.linear.default(relu_2, _param_constant10, _param_constant11);  relu_2 = _param_constant10 = _param_constant11 = None
    return pytree.tree_unflatten([linear_1], self._out_spec)
    
# To see more debug info, please use `graph_module.print_readable()`
'''


# Prepare (and fuse) PT2E model
pt2e_prepared_model = prepare_pt2e(pt2e_traced_model, quantizer)
print(pt2e_prepared_model)
'''
GraphModule(
  (activation_post_process_0): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (activation_post_process_1): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (activation_post_process_2): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (activation_post_process_3): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (activation_post_process_4): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (activation_post_process_5): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (activation_post_process_6): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (activation_post_process_7): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (activation_post_process_8): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (activation_post_process_9): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (activation_post_process_10): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
  (activation_post_process_11): FusedMovingAvgObsFakeQuantize(
    fake_quant_enabled=tensor([1]), observer_enabled=tensor([1]), scale=tensor([1.]), zero_point=tensor([0], dtype=torch.int32), dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, reduce_range=False
    (activation_post_process): MovingAverageMinMaxObserver(min_val=inf, max_val=-inf)
  )
)



def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    arg0_1 = arg0
    activation_post_process_0 = self.activation_post_process_0(arg0_1);  arg0_1 = None
    _param_constant0 = self.conv1_weight
    activation_post_process_1 = self.activation_post_process_1(_param_constant0);  _param_constant0 = None
    _param_constant1 = self.conv1_bias
    conv2d = torch.ops.aten.conv2d.default(activation_post_process_0, activation_post_process_1, _param_constant1);  activation_post_process_0 = activation_post_process_1 = _param_constant1 = None
    relu = torch.ops.aten.relu.default(conv2d);  conv2d = None
    activation_post_process_2 = self.activation_post_process_2(relu);  relu = None
    max_pool2d = torch.ops.aten.max_pool2d.default(activation_post_process_2, [2, 2], [2, 2]);  activation_post_process_2 = None
    activation_post_process_3 = self.activation_post_process_3(max_pool2d);  max_pool2d = None
    _param_constant4 = self.conv2_weight
    activation_post_process_4 = self.activation_post_process_4(_param_constant4);  _param_constant4 = None
    _param_constant5 = self.conv2_bias
    conv2d_1 = torch.ops.aten.conv2d.default(activation_post_process_3, activation_post_process_4, _param_constant5);  activation_post_process_3 = activation_post_process_4 = _param_constant5 = None
    relu_1 = torch.ops.aten.relu.default(conv2d_1);  conv2d_1 = None
    activation_post_process_5 = self.activation_post_process_5(relu_1);  relu_1 = None
    max_pool2d_1 = torch.ops.aten.max_pool2d.default(activation_post_process_5, [2, 2], [2, 2]);  activation_post_process_5 = None
    activation_post_process_6 = self.activation_post_process_6(max_pool2d_1);  max_pool2d_1 = None
    flatten = torch.ops.aten.flatten.using_ints(activation_post_process_6, 1);  activation_post_process_6 = None
    activation_post_process_7 = self.activation_post_process_7(flatten);  flatten = None
    _param_constant8 = self.fc1_weight
    activation_post_process_8 = self.activation_post_process_8(_param_constant8);  _param_constant8 = None
    _param_constant9 = self.fc1_bias
    linear = torch.ops.aten.linear.default(activation_post_process_7, activation_post_process_8, _param_constant9);  activation_post_process_7 = activation_post_process_8 = _param_constant9 = None
    relu_2 = torch.ops.aten.relu.default(linear);  linear = None
    activation_post_process_9 = self.activation_post_process_9(relu_2);  relu_2 = None
    _param_constant10 = self.fc2_weight
    activation_post_process_10 = self.activation_post_process_10(_param_constant10);  _param_constant10 = None
    _param_constant11 = self.fc2_bias
    linear_1 = torch.ops.aten.linear.default(activation_post_process_9, activation_post_process_10, _param_constant11);  activation_post_process_9 = activation_post_process_10 = _param_constant11 = None
    activation_post_process_11 = self.activation_post_process_11(linear_1);  linear_1 = None
    return pytree.tree_unflatten([activation_post_process_11], self._out_spec)
    
# To see more debug info, please use `graph_module.print_readable()`
'''



# device = 'cuda:0'
device='cpu'
pt2e_prepared_model.to(device)
lr = 0.9
optimizer = torch.optim.Adadelta(pt2e_prepared_model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
max_epochs = 5
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=int(((len(train_loader)-1)//batch_size + 1)),
    epochs=max_epochs,
    cycle_momentum=False)
log_interval = 60

# train qat model
for epoch in range(1, max_epochs + 1):
    train(pt2e_prepared_model, device, train_loader, optimizer, criterion, epoch, log_interval)
    test(pt2e_prepared_model, device, test_loader, criterion)
    scheduler.step()


# Test set: Average loss: 0.0047, Accuracy: 9105/10000 (91%)

# freeze observers for a pre-conversion test
torch.ao.quantization.fake_quantize.disable_observer(pt2e_prepared_model)
test(pt2e_prepared_model.cpu(), 'cpu', test_loader, criterion)
# Test set: Average loss: 0.0047, Accuracy: 9092/10000 (91%)

pt2e_quantized_model = convert_pt2e(pt2e_prepared_model)
print(pt2e_quantized_model)
'''
GraphModule()



def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    arg0_1 = arg0
    quantize_per_tensor_default = torch.ops.quantized_decomposed.quantize_per_tensor.default(arg0_1, 0.012728233821690083, -95, -128, 127, torch.int8);  arg0_1 = None
    dequantize_per_tensor_default = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default, 0.012728233821690083, -95, -128, 127, torch.int8);  quantize_per_tensor_default = None
    _frozen_param0 = self._frozen_param0
    dequantize_per_tensor_default_1 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(_frozen_param0, 0.008397966623306274, 0, -128, 127, torch.int8);  _frozen_param0 = None
    _param_constant1 = self.conv1_bias
    conv2d = torch.ops.aten.conv2d.default(dequantize_per_tensor_default, dequantize_per_tensor_default_1, _param_constant1);  dequantize_per_tensor_default = dequantize_per_tensor_default_1 = _param_constant1 = None
    relu = torch.ops.aten.relu.default(conv2d);  conv2d = None
    quantize_per_tensor_default_2 = torch.ops.quantized_decomposed.quantize_per_tensor.default(relu, 0.013678223825991154, -128, -128, 127, torch.int8);  relu = None
    dequantize_per_tensor_default_2 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_2, 0.013678223825991154, -128, -128, 127, torch.int8);  quantize_per_tensor_default_2 = None
    max_pool2d = torch.ops.aten.max_pool2d.default(dequantize_per_tensor_default_2, [2, 2], [2, 2]);  dequantize_per_tensor_default_2 = None
    quantize_per_tensor_default_3 = torch.ops.quantized_decomposed.quantize_per_tensor.default(max_pool2d, 0.013678223825991154, -128, -128, 127, torch.int8);  max_pool2d = None
    dequantize_per_tensor_default_3 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_3, 0.013678223825991154, -128, -128, 127, torch.int8);  quantize_per_tensor_default_3 = None
    _frozen_param1 = self._frozen_param1
    dequantize_per_tensor_default_4 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(_frozen_param1, 0.058939602226018906, 0, -128, 127, torch.int8);  _frozen_param1 = None
    _param_constant5 = self.conv2_bias
    conv2d_1 = torch.ops.aten.conv2d.default(dequantize_per_tensor_default_3, dequantize_per_tensor_default_4, _param_constant5);  dequantize_per_tensor_default_3 = dequantize_per_tensor_default_4 = _param_constant5 = None
    relu_1 = torch.ops.aten.relu.default(conv2d_1);  conv2d_1 = None
    quantize_per_tensor_default_5 = torch.ops.quantized_decomposed.quantize_per_tensor.default(relu_1, 0.02328488975763321, -128, -128, 127, torch.int8);  relu_1 = None
    dequantize_per_tensor_default_5 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_5, 0.02328488975763321, -128, -128, 127, torch.int8);  quantize_per_tensor_default_5 = None
    max_pool2d_1 = torch.ops.aten.max_pool2d.default(dequantize_per_tensor_default_5, [2, 2], [2, 2]);  dequantize_per_tensor_default_5 = None
    quantize_per_tensor_default_6 = torch.ops.quantized_decomposed.quantize_per_tensor.default(max_pool2d_1, 0.02328488975763321, -128, -128, 127, torch.int8);  max_pool2d_1 = None
    dequantize_per_tensor_default_6 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_6, 0.02328488975763321, -128, -128, 127, torch.int8);  quantize_per_tensor_default_6 = None
    flatten = torch.ops.aten.flatten.using_ints(dequantize_per_tensor_default_6, 1);  dequantize_per_tensor_default_6 = None
    quantize_per_tensor_default_7 = torch.ops.quantized_decomposed.quantize_per_tensor.default(flatten, 0.02328488975763321, -128, -128, 127, torch.int8);  flatten = None
    dequantize_per_tensor_default_7 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_7, 0.02328488975763321, -128, -128, 127, torch.int8);  quantize_per_tensor_default_7 = None
    _frozen_param2 = self._frozen_param2
    dequantize_per_tensor_default_8 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(_frozen_param2, 0.0020467343274503946, 0, -128, 127, torch.int8);  _frozen_param2 = None
    _param_constant9 = self.fc1_bias
    linear = torch.ops.aten.linear.default(dequantize_per_tensor_default_7, dequantize_per_tensor_default_8, _param_constant9);  dequantize_per_tensor_default_7 = dequantize_per_tensor_default_8 = _param_constant9 = None
    relu_2 = torch.ops.aten.relu.default(linear);  linear = None
    quantize_per_tensor_default_9 = torch.ops.quantized_decomposed.quantize_per_tensor.default(relu_2, 0.1515197604894638, -128, -128, 127, torch.int8);  relu_2 = None
    dequantize_per_tensor_default_9 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_9, 0.1515197604894638, -128, -128, 127, torch.int8);  quantize_per_tensor_default_9 = None
    _frozen_param3 = self._frozen_param3
    dequantize_per_tensor_default_10 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(_frozen_param3, 0.0048540569841861725, 0, -128, 127, torch.int8);  _frozen_param3 = None
    _param_constant11 = self.fc2_bias
    linear_1 = torch.ops.aten.linear.default(dequantize_per_tensor_default_9, dequantize_per_tensor_default_10, _param_constant11);  dequantize_per_tensor_default_9 = dequantize_per_tensor_default_10 = _param_constant11 = None
    quantize_per_tensor_default_11 = torch.ops.quantized_decomposed.quantize_per_tensor.default(linear_1, 0.15816844999790192, 61, -128, 127, torch.int8);  linear_1 = None
    dequantize_per_tensor_default_11 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_11, 0.15816844999790192, 61, -128, 127, torch.int8);  quantize_per_tensor_default_11 = None
    return pytree.tree_unflatten([dequantize_per_tensor_default_11], self._out_spec)
    
# To see more debug info, please use `graph_module.print_readable()`
'''

test(pt2e_quantized_model, 'cpu', test_loader, criterion)
# Test set: Average loss: 0.0049, Accuracy: 9074/10000 (91%)
```



#### Export method conversion

In progress...


##### Eager mode observed to FX observed

In progress...

##### FX float to PT2E float

In progress...

##### FX quantized to PT2E quantized

In progress...

###### FX PTQ to PT2E PTQ

In progress...

###### FX QAT to PT2E QAT


In progress...

### ONNX


In progress...


### Tensorflow / Keras /TFLite

In progress...

#### TF float to PTQ

In progress...

#### TF Keras to QAT (tfmot)

In progress...

##### Sequential

In progress...

##### Functional


In progress...

## Backend conversion

### Pytorch float to ONNX float

In progress...
### ONNX float to Keras float

In progress...
### Tensorflow / Keras float to Quantized TFLite

In progress...
### Pytorch quantized to ONNX quantized

In progress...
### Pytorch observed QAT to ONNX quantized

In progress...
### Pytorch to TFLite (ai_edge_torch)

In progress...
#### Pytorch float PT2E to TFLite quantized (ai_edge_torch)

In progress...
#### Pytorch quantized PT2E to TFLite quantized (ai_edge_torch)

In progress...
#### Pytorch FX quantized to PT2E to TFLite quantized (ai_edge_torch)


In progress...


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

the above works but only if using the custom ai_edge_torch fx backend config I wrote.


Memos:

- Hailo8 backend:https://github.com/hailo-ai/hailo_model_zoo
    - updates with database?
