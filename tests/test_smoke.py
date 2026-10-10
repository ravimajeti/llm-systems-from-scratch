import torch

import llmsys


def test_package_imports():
    assert llmsys.__version__


def test_torch_matmul_cpu():
    a = torch.randn(4, 8)
    b = torch.randn(8, 3)
    assert (a @ b).shape == (4, 3)
