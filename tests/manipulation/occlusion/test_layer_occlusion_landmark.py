import pytest

from pyfame.layer.manipulations.occlusion._layer_occlusion_landmark import (
    layer_occlusion_landmark,
    LayerOcclusionLandmark,
)

@pytest.mark.parametrize(
    "fill_method",
    [
        "black",
        "mean",
        "BLACK",
        "MEAN",
        8,
        9,
    ],
)
def test_layer_occlusion_landmark_valid_fill_method(fill_method):
    layer = layer_occlusion_landmark(fill_method=fill_method)

    assert isinstance(layer, LayerOcclusionLandmark)

def test_layer_occlusion_landmark_valid_landmark_path(landmark_path):
    layer = layer_occlusion_landmark(landmark_paths=landmark_path)

    assert isinstance(layer, LayerOcclusionLandmark)

def test_layer_occlusion_landmark_valid_landmark_paths(landmark_paths):
    layer = layer_occlusion_landmark(landmark_paths=landmark_paths)

    assert isinstance(layer, LayerOcclusionLandmark)

@pytest.mark.parametrize(
    "fill_method",
    [
        "invalid",
        "",
        "white",
        7,
        10,
    ],
)
def test_layer_occlusion_landmark_invalid_fill_method(fill_method):
    with pytest.raises(ValueError):
        layer_occlusion_landmark(fill_method=fill_method)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("landmark_paths", "invalid"),
        ("landmark_paths", 123),
        ("fill_method", 1.5),
    ],
)
def test_layer_occlusion_landmark_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_occlusion_landmark(**{parameter: value})

@pytest.mark.parametrize(
    "fill_method, expected",
    [
        ("black", "black"),
        ("mean", "mean"),
        ("BLACK", "black"),
        ("MEAN", "mean"),
        (8, "black"),
        (9, "mean"),
    ],
)
def test_layer_occlusion_landmark_normalizes_fill_method(
    fill_method, expected
):
    layer = layer_occlusion_landmark(fill_method=fill_method)

    params = layer.get_layer_parameters()

    assert params["fill_method"] == expected