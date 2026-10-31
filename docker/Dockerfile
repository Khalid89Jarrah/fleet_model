FROM ubuntu:noble AS base

ENV HOME=/root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade

RUN apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    gdb \
    vim \
    sudo \
    locales \
    build-essential \
    pkg-config \
    python3 \
    python3-dev \
    python3-venv

RUN locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN python3 -m venv /opt/virtual_env

RUN /opt/virtual_env/bin/pip install --upgrade pip

RUN /opt/virtual_env/bin/pip install --no-cache-dir \
    tensorflow==2.18.0 \
    PyYAML==6.0.2 \
    gymnasium==1.0.0 \
    matplotlib==3.9.2 \
    pandas==2.2.3

ENV PATH="/opt/virtual_env/bin:$PATH"

# run_docker.sh mounts the sources here, so python3 finds sensor_interaction
ENV PYTHONPATH="/opt/fleet_model_ws/src/sensor_interaction"

WORKDIR /opt

ENTRYPOINT ["/bin/bash"]