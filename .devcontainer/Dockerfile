FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN pip install -U pip \
    && pip install --no-cache-dir japanize_matplotlib \
    && pip install --no-cache-dir torchviz \
    && pip install --no-cache-dir torchinfo

RUN useradd -m -d /home/dev-user -s /bin/bash dev-user
USER dev-user

RUN apt-get dist-upgrade -y\
    && apt-get update \
    && apt-get install build-essential \
    # && apt-get install graphviz \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /usr/local/src/*
