ARG CUDA="10.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++-5 \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev vim feh wget xterm

# Install Miniconda
RUN wget -O /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install ninja yacs cython matplotlib opencv-python tqdm sklearn comet_ml shapely pandas 

# Install PyTorch 1.0 Nightly and OpenCV
ARG CUDA
RUN conda install -y pytorch==1.1.0 cudatoolkit=${CUDA} -c pytorch \
 && conda clean -ya

RUN pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
# Install TorchVision master
RUN git clone --single-branch --branch v0.2.2_branch https://github.com/pytorch/vision.git \
 && cd vision \
 && python setup.py install

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI && git checkout aca78bcd6b4345d25405a64fdba1120dfa5da1ab \
 && python setup.py build_ext install

# install apex
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex && git checkout 4ff153cd50e4533b21dc1fd97c0ed609e19c4042 \
 && python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark && git checkout a44d65dcdb9c9098a25dd6b23ca3c36f1b887aba\
 && python setup.py build develop 

WORKDIR /code
