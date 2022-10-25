FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install alien
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt install libdb-dev -y
RUN apt install build-essential -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install python3-pip git
RUN pip3 install --upgrade pip
RUN pip3 install cython pycocotools
RUN pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install opencv-python==4.5.4.58 pyaml==21.10.1 mmdet==2.11.0 tqdm matplotlib
RUN pip3 install numpy scipy pytorch_ssim torchsummary
RUN pip3 install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip3 install -r requirements.txt
RUN apt-get install bash wget -y
RUN pip3 install git+https://github.com/elliottzheng/face-detection.git@master
RUN pip3 install seaborn

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
        python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
        python3 get-pip.py --user && \
        rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip3 install --user tensorboard cmake onnx   # cmake from apt-get is too old

RUN pip3 install 'git+https://github.com/facebookresearch/fvcore'
RUN pip3 install -U git+https://github.com/facebookresearch/fvcore.git
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip3 install -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
COPY . /opt/app
