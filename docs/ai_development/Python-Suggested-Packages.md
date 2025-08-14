# Python suggested packages

!!!!!!!!!!!!
This section needs heavy editing, after years, some of the libraries are not recommended anymore.
!!!!!!!!!!!!

TODO:
- [ ] Update all python package suggestions (pretty outdated)
    - [ ] Add pyaudio stuff because it was complicated.

<!-- This is my generic fresh start install so I can work. Usually I'd install all of them in general, but recently I only install the necessary libraries under venv. There's more libraries with complicated installations in other repositories of mine, and you might not wanna run this particular piece of code without checking what I'm doing first. For example, you might have a specific version of Tensorflow that you want, or some of these you won't use. But I'll leave it here as reference.


#### Basic tasks:

```
pip install numpy scipy statsmodels \
pandas pathlib tqdm retry openpyxl
```


#### Plotting:
```
pip install matplotlib adjustText plotly kaleido
```


#### Basic data science and machine learning:
```
pip install sklearn sympy pyclustering
```


#### Data mining / text mining / crawling / scraping websites:
```
pip install beautifulsoup4 requests selenium
```


#### Natural language processing (NLP):
```
pip install gensim nltk langdetect
```

For Japanese NLP tools see my example repository: 
- [MeCab-python](https://github.com/elisa-aleman/MeCab-python)

For Chinese NLP tools and installation guides I developed see: 
- [StanfordCoreNLP_Chinese](https://github.com/elisa-aleman/StanfordCoreNLP_Chinese)

Both are pretty old repositories that I haven't looked that in forever, though.

#### Neural network and machine learning:
```
pip install tensorflow tflearn keras \
torch torchaudio torchvision \
optuna
```

#### XGBoost

To Install with CPU:
```
pip install xgboost
```

To Install with CUDA GPU integration:
```
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build
cd build
cmake .. -DUSE_CUDA=ON
make -j8
cd ../python-package
python setup.py install
```


#### LightGBM

To Install with CPU:

```
pip install lightgbm
```

Install dependencies:
```
apt-get install libboost-all-dev
apt install ocl-icd-libopencl1
apt install opencl-headers
apt install clinfo
apt install ocl-icd-opencl-dev
```
Install with CUDA GPU integration:

```
pip install lightgbm --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"
```


#### MINEPY / Maximal Information Coefficient

For Minepy / Maximal Information Coefficient, we need the Visual Studio C++ Build Tools as a dependency, so install it first:<br>
https://visualstudio.microsoft.com/visual-cpp-build-tools/

```
pip install minepy
```


#### Computer Vision (OpenCV)

**Note to self: re-write with poetry project use instead of venv**

with CPU and no extra options:

2024-09 note, PYPI opencv-python-headless is pretty much all that's needed for CPU builds, and has ffmpeg now.
https://github.com/opencv/opencv-python

```
python -m pip install -U opencv-python opencv-contrib-python
```



##### Install OpenCV with CUDA


The exception would be for CUDA, for which there are a few pre-built wheels 
