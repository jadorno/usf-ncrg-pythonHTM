# [Choice] Debian version (use bullseye on local arm64/Apple Silicon): bullseye, buster
ARG VARIANT=bullseye
FROM mcr.microsoft.com/vscode/devcontainers/base:${VARIANT}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y \
    && apt-get install -y --no-install-recommends software-properties-common curl git \
    build-essential ca-certificates libssl-dev openssl wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/

ENV LD_LIBRARY_PATH=/usr/local/lib

RUN wget "https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz" \
    && tar xzvf Python-3.6.9.tgz \
    && cd Python-3.6.9 \
    && ./configure --enable-shared \
    && make -j4 \
    && make install \
    && cd /usr/src/ \
    && wget https://bootstrap.pypa.io/pip/3.6/get-pip.py \
    && python3.6 get-pip.py

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

