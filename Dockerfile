FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.8
ARG WITH_TORCHVISION=1
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         python-pip \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

WORKDIR /pytorch

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy matplotlib && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# This must be done before pip so that requirements.txt is available

RUN pip install torch torchvision

COPY dcgan.py utils.py model.py /pytorch/
COPY GAN/data /pytorch/data

# RUN git submodule sync && git submodule update --init --recursive
# RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
#     CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
#     pip install -v .
#
# RUN if [ "$WITH_TORCHVISION" = "1" ] ; then git clone https://github.com/pytorch/vision.git && cd vision && pip install -v . ; else echo "building without torchvision" ; fi
#
# WORKDIR /workspace
# RUN chmod -R a+w .