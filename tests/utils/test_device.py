from unittest import mock

import numpy as np
import pytest
import torch

from multipinn.utils.device import set_device, set_device_and_seed, set_seed


@pytest.fixture
def mock_logger():
    with mock.patch("multipinn.utils.device.logger") as logger:
        yield logger


def test_set_seed():
    seed = 42
    set_seed(seed)
    # Check if seeds were set correctly
    assert torch.initial_seed() == seed
    assert np.random.get_state()[1][0] == seed  # The seed should be in the random state


def test_set_device_cpu(mock_logger):
    with mock.patch("torch.cuda.is_available", return_value=False):
        set_device()
        mock_logger.info.assert_called_once_with(
            mock.ANY
        )  # Device info should have been logged

        assert torch.tensor(0).device.type == "cpu"


def test_set_device_cuda(mock_logger):
    # Mock necessary torch functions
    with mock.patch("torch.cuda.is_available", return_value=True), \
     mock.patch("torch.cuda.set_device") as mock_cuda_set_device, \
     mock.patch("torch.set_default_device") as mock_set_device, \
     mock.patch("torch.set_default_dtype") as mock_set_dtype:
        set_device(gpu_id=0)
        mock_cuda_set_device.assert_called_once_with(torch.device("cuda:0"))
        mock_set_device.assert_called_once_with(torch.device("cuda:0"))
        mock_set_dtype.assert_called_once_with(torch.float32)

        # Verify logger was called
        mock_logger.info.assert_called_once_with(mock.ANY)


@pytest.mark.parametrize(
    "accelerator, gpu_id, cuda_available, expected_device",
    [
        (None, 0, False, "cpu"),
        (None, 0, True, "cuda:0"),
        ("cuda:0", 1, True, "cuda:0"),
    ],
)
def test_set_device_parametric(accelerator, gpu_id, cuda_available, expected_device, mock_logger):
    with mock.patch("torch.cuda.is_available", return_value=cuda_available), \
         mock.patch("torch.cuda.set_device") as mock_cuda_set_device, \
         mock.patch("torch.set_default_device") as mock_set_device, \
         mock.patch("torch.set_default_dtype") as mock_set_dtype:

        set_device(accelerator, gpu_id)

        logged_device = str(mock_logger.info.call_args[0][0]).split()[-1]
        assert logged_device == expected_device

        if expected_device.startswith("cuda"):
            expected = torch.device(expected_device)
            mock_cuda_set_device.assert_called_once_with(expected)
            mock_set_device.assert_called_once_with(expected)
            mock_set_dtype.assert_called_once_with(torch.float32)
        else:
            mock_cuda_set_device.assert_not_called()
            mock_set_device.assert_not_called()
            mock_set_dtype.assert_not_called()


@pytest.mark.parametrize("seed", [42, 0, 1000])
def test_set_device_and_seed(seed):
    set_device_and_seed(seed)
    assert torch.initial_seed() == seed
    assert np.random.get_state()[1][0] == seed  # Check if numpy's seed was also set
