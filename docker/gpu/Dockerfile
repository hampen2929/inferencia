FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

USER root
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTP_PROXY

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    git \
    zip \
    unzip \
    curl \
    xterm \
    vim \
    bzip2 \
    x11-apps \
    ca-certificates \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Add ubuntu user
ENV USERNAME ubuntu
RUN useradd -m -s /bin/bash $USERNAME
RUN gpasswd -a $USERNAME sudo
RUN echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN visudo --c

# Python
ENV PYTHON_VERSION 3.7.9
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv

ENV TZ Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libgl1-mesa-dev \
    && git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
    && $PYENV_ROOT/plugins/python-build/install.sh \
    && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
    && rm -rf $PYENV_ROOT

# Path
RUN sudo chown ubuntu:ubuntu /local/python-3.7.9/ -R

# CMAKE
RUN apt-get update
RUN sudo apt remove cmake -y
ARG DOWNLOAD_LINK=https://github.com/Kitware/CMake/releases/download/v3.16.2/cmake-3.16.2-Linux-x86_64.sh
ARG TEMP_DIR=/home/ubuntu/cmake_installer/cmake-3.16.2-Linux-x86_64/

RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR \
    && wget $DOWNLOAD_LINK \
    && chmod +x cmake-*-Linux-x86_64.sh \
    && sudo bash cmake-*-Linux-x86_64.sh --skip-license \
    && cd .. \
    && sudo mv cmake-*-Linux-x86_64 /opt \
    && sudo ln -s /opt/cmake-3.16.2-Linux-x86_64/bin/* /usr/bin

# Install python package
USER ${USERNAME}
ENV PROJECT_DIR /workspace/
WORKDIR ${PROJECT_DIR}

RUN pip install --upgrade pip

ADD ./requirements_prod.txt ${PROJECT_DIR}
RUN pip install -r requirements_prod.txt

ADD ./requirements_dev.txt ${PROJECT_DIR}
RUN pip install -r requirements_dev.txt

ADD ./requirements_other.txt ${PROJECT_DIR}
RUN pip install -r requirements_other.txt

ENV PATH $PATH:/home/${USERNAME}/.local/bin

# Jupyter notebook
RUN mkdir /home/${USERNAME}/.jupyter/
ADD ./jupyter_notebook_config.py /home/ubuntu/.jupyter/

