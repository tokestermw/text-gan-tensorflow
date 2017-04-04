FROM gcr.io/tensorflow/tensorflow:1.0.0-py3

MAINTAINER Motoki Wu <tokestermw@gmail.com>

# minimal docker file
# python version is 3.4, 3.6 is used for development

# http://stackoverflow.com/a/34399661/2802468
COPY requirements.txt /opt/app/requirements.txt

WORKDIR /opt/app
RUN pip install -r requirements.txt

COPY . /opt/app

CMD /bin/bash
# try running python train.py