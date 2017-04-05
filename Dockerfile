FROM gcr.io/tensorflow/tensorflow:1.0.0-py3

MAINTAINER Motoki Wu <tokestermw@gmail.com>

RUN apt-get update && apt-get install -y \
        git \
        wget \
        htop \
        && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app
# http://stackoverflow.com/a/34399661/2802468
# COPY requirements.txt /opt/app/requirements.txt
RUN git clone https://github.com/tokestermw/text-gan-tensorflow

WORKDIR /opt/app/text-gan-tensorflow
RUN pip install -r requirements.txt

# COPY . /opt/app

# EXPOSE 8888

CMD git pull && /bin/bash
# try the following
# docker run text-gan-tensorflow:0.0.1 python train.py
# docker run text-gan-tensorflow:0.0.1 tensorboard --logdir tmp/ --port 5000