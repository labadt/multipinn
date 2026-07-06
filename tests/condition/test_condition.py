from unittest.mock import Mock, create_autospec

import pytest
import torch
import numpy as np

from multipinn.condition import *
from multipinn.geometry.geometry import Geometry


# Fixtures
@pytest.fixture
def mock_geometry():
    geometry = create_autospec(Geometry)
    geometry.bbox = [torch.tensor([0.0]), torch.tensor([1.0])]
    geometry.boundary_normal = Mock(return_value=[[1.0, 0.0]])
    return geometry


@pytest.fixture
def mock_generator():
    generator = Mock()
    generator.n_points = 10
    generator.generate = Mock(return_value=torch.rand((10, 2)))
    return generator


@pytest.fixture
def mock_model():
    model = Mock()
    model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    return model


@pytest.fixture
def function():
    return Mock(return_value=torch.tensor([1.0]))


@pytest.fixture
def condition(mock_geometry, function, mock_generator):
    cond = Condition(function=function, geometry=mock_geometry)
    cond.generator = mock_generator
    return cond


@pytest.fixture
def condition_extra(mock_geometry, function, mock_generator):
    data_gen = [
        lambda points: torch.ones_like(points),
        "normals",
    ]
    cond_extra = ConditionExtra(
        function=function, geometry=mock_geometry, data_gen=data_gen
    )
    cond_extra.generator = mock_generator
    return cond_extra


# Tests for Condition class
def test_update_points(condition, mock_model):
    condition.update_points(model=mock_model)
    assert condition.points is not None
    condition.generator.generate.assert_called_once_with(condition, mock_model)


def test_select_batch(condition):
    condition.batch_size = 2
    condition.points = torch.arange(10).reshape((5, 2))
    condition.select_batch(1)
    assert torch.equal(condition.batch_points, torch.tensor([[4, 5], [6, 7]]))


def test_get_residual(condition, mock_model):
    condition.batch_points = torch.tensor([[0.5, 0.5]])
    residual = condition.get_residual(mock_model)
    condition.function.assert_called_once_with(mock_model, condition.batch_points)
    assert torch.equal(residual, torch.tensor([1.0]))


def test_set_batching(condition, mock_generator):
    condition.set_batching(2)
    assert condition.batch_size == 5


def test_init_output_len(condition, mock_model):
    condition.init_output_len(mock_model)
    assert condition.output_len == 1  # Assuming function returns tensor of size 1


# Tests for ConditionExtra class
def test_update_points_extra(condition_extra, mock_model):
    condition_extra.update_points(model=mock_model)
    assert condition_extra.data is not None
    condition_extra.generator.generate.assert_called_once_with(
        condition_extra, mock_model
    )


def test_select_batch_extra(condition_extra):
    condition_extra.batch_size = 2
    condition_extra.points = torch.arange(10).reshape((5, 2))
    condition_extra.data = torch.arange(10).reshape((5, 2))
    condition_extra.select_batch(1)
    assert torch.equal(condition_extra.batch_points, torch.tensor([[4, 5], [6, 7]]))
    assert torch.equal(condition_extra.batch_data, torch.tensor([[4, 5], [6, 7]]))


def test_get_residual_extra(condition_extra, mock_model):
    condition_extra.batch_points = torch.tensor([[0.5, 0.5]])
    condition_extra.batch_data = torch.tensor([[1.0, 1.0]])
    residual = condition_extra.get_residual(mock_model)
    condition_extra.function.assert_called_once_with(
        mock_model, condition_extra.batch_points, condition_extra.batch_data
    )
    assert torch.equal(residual, torch.tensor([1.0]))


def test_condition_extra_generator_for_normals(mock_geometry):
    points = torch.tensor([[0.1, 0.2]])
    gen = ConditionExtra.generator_for_normals(mock_geometry)

    normals = gen(points)

    mock_geometry.boundary_normal.assert_called_once()
    np.testing.assert_array_equal(
        mock_geometry.boundary_normal.call_args[0][0],
        points.detach().cpu().numpy(),
    )
    assert normals.device == points.device
