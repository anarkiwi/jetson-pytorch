#!/bin/sh

exec docker build --build-arg PIP_OPTS="--index-url http://192.168.5.1:5001/index/ --trusted-host 192.168.5.1" --build-arg PYTORCH_VERSION=v2.3.0 -t anarkiwi/jetson-pytorch:v2.3.0 -f Dockerfile.pytorch .
