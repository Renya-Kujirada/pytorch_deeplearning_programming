FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN pip install -U pip \
    && pip install --no-cache-dir japanize_matplotlib \
    && pip install --no-cache-dir torchviz \
    && pip install --no-cache-dir torchinfo

RUN conda install graphviz \
    && conda clean -i -t -y

RUN useradd -m -d /home/dev-user -s /bin/bash dev-user
USER dev-user


# pip install --no-cache-dir mlflow
# pip install --no-cache-dir autopep8
