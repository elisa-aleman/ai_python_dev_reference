# Python suggested packages

!!!!!!!!!!!!
This section needs heavy editing, after years, some of the libraries are not recommended anymore.
!!!!!!!!!!!!

- [ ] Update all python package suggestions (pretty outdated)
    - [ ] Add pyaudio stuff because it was complicated.
    - [ ] Opencv with poetry

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

```
python -m pip install -U opencv-python opencv-contrib-python
```



##### Install OpenCV with ffmpeg

Install dependencies:

```
apt-get update
apt-get upgrade
apt-get install build-essential
apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
apt-get install libxvidcore-dev libx264-dev
apt-get install libgtk-3-dev
apt-get install libatlas-base-dev gfortran pylint
apt-get install python2.7-dev python3.5-dev python3.6-dev
apt-get install unzip
```

Now the ffmpeg dependency:
```
add-apt-repository ppa:jonathonf/ffmpeg-3
apt update
apt install ffmpeg libav-tools x264 x265
```

Check the version:
```
ffmpeg
```

Download and build opencv

```
wget https://github.com/opencv/opencv/archive/3.4.0.zip -O opencv-3.4.0.zip
wget https://github.com/opencv/opencv_contrib/archive/3.4.0.zip -O opencv_contrib-3.4.0.zip

unzip opencv-3.4.0.zip
unzip opencv_contrib-3.4.0.zip

cd  opencv-3.4.0
mkdir build_3.5
mkdir build
cd build
```

Make, but remember to replace Python versions:

```
which python
```

```
cmake -DCMAKE_BUILD_TYPE=Release \
    -D WITH_FFMPEG=ON \
    -D PYTHON3_EXECUTABLE=<path to your python> \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.0/modules \
    -D OPENCV_ENABLE_NONFREE=True ..

make -j8        #(where -j8 is for 8 cores in the server cpu)
make install
ldconfig
```


 -->