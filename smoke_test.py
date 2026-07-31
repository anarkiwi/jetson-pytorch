"""Post-build check that the image is usable off-Jetson.

Everything here runs without an NVIDIA driver, so CI on a GPU-less runner
catches packaging faults before the image ships. GPU execution needs real
hardware; torch.cuda.is_available() is deliberately not asserted.
"""

import sys

import numpy
import torch

EXPECTED_ARCHS = ["sm_87"]  # Orin
EXPECTED_CUDA_MAJOR = "13"


def main() -> int:
    """Report what the image carries and assert it is coherent."""
    print(f"python  {sys.version.split()[0]}")
    print(f"torch   {torch.__version__}")
    print(f"cuda    {torch.version.cuda}")
    print(f"cudnn   {torch.backends.cudnn.version()}")
    print(f"archs   {torch.cuda.get_arch_list()}")
    print(f"numpy   {numpy.__version__}")

    assert torch.cuda.get_arch_list() == EXPECTED_ARCHS, torch.cuda.get_arch_list()
    assert torch.version.cuda.split(".")[0] == EXPECTED_CUDA_MAJOR, torch.version.cuda
    assert torch.backends.cudnn.version() is not None

    # ATen kernels the wheel was built with, plus the numpy bridge.
    x = torch.randn(64, 64)
    assert x.mm(x).shape == (64, 64)
    assert numpy.allclose(x.numpy(), x.detach().cpu().numpy())
    assert torch.from_numpy(numpy.eye(4, dtype="float32")).sum().item() == 4.0

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
