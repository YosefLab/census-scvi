import torch


def test_pytorch_cuda():
    assert torch.cuda.is_available()


if __name__ == "__main__":
    test_pytorch_cuda()