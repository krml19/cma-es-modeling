FROM ubuntu:16.04
RUN apt-get update \
  && apt-get install -y software-properties-common python-software-properties curl
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update \
  && apt-get install -y python3-pip docker.io python3.6 python3.6-dev  \
  && apt-get install -y software-properties-common python-software-properties build-essential wget unzip cmake \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3.6 python
RUN python -m pip install --upgrade pip

# display CPU info
RUN cat /proc/cpuinfo
ADD ./requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY . /app
WORKDIR /app
RUN export PYTHONPATH=$PYTHONPATH:/app

ENV PYTHONPATH /app
CMD ["bash"]
