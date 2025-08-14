# AI Python Developer Reference Wiki

Personal collection of documentation and reference guides for AI development with python, from PC setup guides, recommended software to install, to anything I've learned about AI training, quantization, etc.

- Initial setup guides:
    - [Windows setup guide](./docs/setup_guides/Windows-Setup.md)
    - [Linux / WSL setup guide](./docs/setup_guides/Linux-WSL-Setup.md)
    - [MacOS setup guide](./docs/setup_guides/MacOS-Setup.md)
    - [Git Setup and Customization](./docs/setup_guides/Git-Setup-and-Customization.md)
    - [Suggested Tools and Setup](./docs/setup_guides/Suggested-Tools-and-Setup.md)
        - My preferred command line tools, developing environment managers, documentation tools, accessibility references
    - [Web Development tools and guides](./docs/Web-Development)
- [About Quantization](./docs/ai_development/About-Quantization.md)
    - Both a theoretic and practical introduction to quantization, as well as conversion guides for pytorch, onnx, and tensorflow, both for float and quantized models.
    - Particularly of interest is practical examples of the three methods of quantization in pytorch, and ways to export pytorch quantized models to onnx and tflite.    
    

## Code Block Style Guide

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

For example:

```sh
# @ environment::sub_environment::/.../path

echo "Example codeblock"
```

- `# @` will mean the place where the code is happening. Since it's a shell comment, a `#` is used but some other syntaxes use other comments
- the environment will be things like `shell` or `python` depending on where the command is being run
- the subsequent environments being for example, a docker container, a virtual environment, and so on. 
- the path, when relevant to the command being input, or where the file being written in the code block is saved.

This will make it more accessible to determine where something is run, even if revisiting the code block after a long time has passed.
