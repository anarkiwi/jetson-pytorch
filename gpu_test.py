"""On-device check that CUDA actually executes, for a real Jetson.

Complements smoke_test.py, which deliberately runs without a driver so
GPU-less CI can gate packaging. Run this on an Orin under the nvidia
runtime; it fails on any host without a working GPU.
"""

import sys

import torch
from torch import nn

EXPECTED_CAPABILITY = (8, 7)  # Orin


def main() -> int:
    """Exercise CUDA and cuDNN against CPU references."""
    assert torch.cuda.is_available(), "no CUDA device visible"
    props = torch.cuda.get_device_properties(0)
    print(f"device  {props.name} sm_{props.major}{props.minor}")
    print(f"memory  {props.total_memory // 1024 ** 2} MB")
    print(f"driver  {torch.version.cuda}")

    assert (props.major, props.minor) == EXPECTED_CAPABILITY, props

    a, b = torch.randn(512, 512), torch.randn(512, 512)
    assert torch.allclose(a @ b, (a.cuda() @ b.cuda()).cpu(), atol=1e-3)

    conv = nn.Conv2d(3, 16, 3)
    x = torch.randn(8, 3, 64, 64)
    assert torch.backends.cudnn.is_available()
    assert torch.allclose(conv(x), conv.cuda()(x.cuda()).cpu(), atol=1e-4)

    half = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    assert (half @ half).shape == (256, 256)

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
