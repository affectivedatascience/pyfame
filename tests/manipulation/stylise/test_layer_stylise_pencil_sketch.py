import pytest

from pyfame.layer.manipulations.stylise._layer_stylise_pencil_sketch import (
    layer_stylise_pencil_sketch,
    LayerStylisePencilSketch,
)


def test_layer_stylise_pencil_sketch_defaults():
    layer = layer_stylise_pencil_sketch()

    assert isinstance(layer, LayerStylisePencilSketch)

    params = layer.get_layer_parameters()

    assert params["landmark_paths"] is None
    assert params["detail_level"] == 0.35
    assert params["threshold_bias"] == 7.0


def test_layer_stylise_pencil_sketch_valid_landmark_path(landmark_path):
    layer = layer_stylise_pencil_sketch(
        landmark_paths=landmark_path
    )

    assert isinstance(layer, LayerStylisePencilSketch)
    assert layer.get_layer_parameters()["landmark_paths"] == landmark_path


def test_layer_stylise_pencil_sketch_valid_landmark_paths(landmark_paths):
    layer = layer_stylise_pencil_sketch(
        landmark_paths=landmark_paths
    )

    assert isinstance(layer, LayerStylisePencilSketch)
    assert layer.get_layer_parameters()["landmark_paths"] == landmark_paths


@pytest.mark.parametrize(
    "detail_level",
    [0.0, 0.01, 0.25, 0.35, 0.5, 0.75, 1.0],
)
def test_layer_stylise_pencil_sketch_valid_detail_level(detail_level):
    layer = layer_stylise_pencil_sketch(
        detail_level=detail_level
    )

    assert isinstance(layer, LayerStylisePencilSketch)
    assert layer.get_layer_parameters()["detail_level"] == detail_level


@pytest.mark.parametrize(
    "threshold_bias",
    [0.0, 0.1, 1.0, 7.0, 15.0, 100.0],
)
def test_layer_stylise_pencil_sketch_valid_threshold_bias(threshold_bias):
    layer = layer_stylise_pencil_sketch(
        threshold_bias=threshold_bias
    )

    assert isinstance(layer, LayerStylisePencilSketch)
    assert layer.get_layer_parameters()["threshold_bias"] == threshold_bias


@pytest.mark.parametrize(
    "detail_level",
    [-1.0, -0.01, 1.01, 2.0],
)
def test_layer_stylise_pencil_sketch_invalid_detail_level(detail_level):
    with pytest.raises(ValueError):
        layer_stylise_pencil_sketch(
            detail_level=detail_level
        )


@pytest.mark.parametrize(
    "threshold_bias",
    [-100.0, -7.0, -0.01],
)
def test_layer_stylise_pencil_sketch_invalid_threshold_bias(threshold_bias):
    with pytest.raises(ValueError):
        layer_stylise_pencil_sketch(
            threshold_bias=threshold_bias
        )


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("landmark_paths", "invalid"),
        ("landmark_paths", 123),
        ("landmark_paths", [(0, "invalid")]),
        ("landmark_paths", [[(0, 1)], "invalid"]),
        ("detail_level", "invalid"),
        ("detail_level", None),
        ("detail_level", []),
        ("threshold_bias", "invalid"),
        ("threshold_bias", None),
        ("threshold_bias", []),
    ],
)
def test_layer_stylise_pencil_sketch_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_stylise_pencil_sketch(**{parameter: value})