# jetson-pytorch

PyTorch wheel + image for NVIDIA Jetson Orin (linux/arm64, sm_87),
built from source against the JetPack 7.2 CUDA toolkit.

## Image

`anarkiwi/jetson-pytorch:${PYTORCH_VERSION}` -- e.g.
`anarkiwi/jetson-pytorch:v2.13.0`. Tag `latest` tracks `main`.

| | |
|---|---|
| Base | `nvcr.io/nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` |
| Host | JetPack 7.2 (Jetson Linux r39.2), Orin family |
| CUDA | 13.2.1 (Arm SBSA) |
| Arch | `TORCH_CUDA_ARCH_LIST=8.7` |

JetPack 7.2 is the first 7.x release covering AGX Orin / Orin NX /
Orin Nano, and CUDA 13.2 is the first toolkit to package Orin as
standard Arm SBSA, so the stock `nvidia/cuda` arm64 image replaces
`l4t-jetpack` (frozen at JetPack 6, `r36.4.0`). The release stage
keeps the `devel` base because `anarkiwi/jetson-triton` compiles
against it.

Triton is not included; see
[anarkiwi/jetson-triton](https://github.com/anarkiwi/jetson-triton).

## Build

```bash
docker buildx build --platform linux/arm64 \
    --build-arg PYTORCH_VERSION=v2.13.0 \
    -f Dockerfile.pytorch -t anarkiwi/jetson-pytorch:v2.13.0 .
```

Build args: `PYTORCH_VERSION` (pytorch git tag), `CUDA_BASE` (CUDA
base image tag), `PIP_OPTS` (extra pip flags, e.g. a local mirror).
An unpacked pytorch source tree at `./pytorch` is used if present,
otherwise the tag is cloned.

## Release

Push a `vX.Y.Z` tag matching the pytorch release; the
`docker-release` workflow builds and pushes to Docker Hub. Requires
`DOCKER_USERNAME` / `DOCKER_PASSWORD` in the `release` environment.

## Sanity check

```bash
docker run --rm --runtime nvidia anarkiwi/jetson-pytorch:v2.13.0 \
    python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
