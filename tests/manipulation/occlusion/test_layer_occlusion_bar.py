import pytest
from pyfame.layer.manipulations.occlusion._layer_occlusion_bar import layer_occlusion_bar, LayerOcclusionBar
from pyfame.landmark import CONCAVE_LANDMARKS

def test_layer_occlusion_bar_defaults():
    layer = layer_occlusion_bar()

    assert isinstance(layer, LayerOcclusionBar)


def test_layer_occlusion_bar_valid_landmark_path(landmark_path):
    layer = layer_occlusion_bar(landmark_paths=landmark_path)

    assert isinstance(layer, LayerOcclusionBar)


@pytest.mark.parametrize(
    "runtime_lm_paths",
    CONCAVE_LANDMARKS,
)
def test_layer_occlusion_bar_valid_runtime_landmarks(runtime_lm_paths):
    layer = layer_occlusion_bar(landmark_paths=runtime_lm_paths)

    assert isinstance(layer, LayerOcclusionBar)


@pytest.mark.parametrize(
    "bar_colour",
    [
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 128, 255),
    ],
)
def test_layer_occlusion_bar_valid_bar_colour(bar_colour):
    layer = layer_occlusion_bar(bar_colour=bar_colour)

    assert isinstance(layer, LayerOcclusionBar)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("vertical_bar_span", 0),
        ("vertical_bar_span", 100),
        ("y_offset", -50),
        ("y_offset", 0),
        ("y_offset", 50),
        ("horizontal_bar_margin", -10),
        ("horizontal_bar_margin", 0),
        ("horizontal_bar_margin", 30),
    ],
)
def test_layer_occlusion_bar_valid_dimensions(parameter, value):
    layer = layer_occlusion_bar(**{parameter: value})

    assert isinstance(layer, LayerOcclusionBar)

@pytest.mark.parametrize(
    "bar_colour",
    [
        (-1, 0, 0),
        (0, 256, 0),
        (0, 0, 300),
        (0, 0),
        (0, 0, 0, 0),
    ],
)
def test_layer_occlusion_bar_invalid_bar_colour(bar_colour):
    with pytest.raises(ValueError):
        layer_occlusion_bar(bar_colour=bar_colour)


@pytest.mark.parametrize(
    "landmark_paths",
    [
        [],
        [(42, 1)],
    ],
)
def test_layer_occlusion_bar_insufficient_landmarks(landmark_paths):
    with pytest.raises(ValueError):
        layer_occlusion_bar(landmark_paths=landmark_paths)


@pytest.mark.parametrize(
    "vertical_bar_span",
    [-1, -100],
)
def test_layer_occlusion_bar_invalid_vertical_span(vertical_bar_span):
    with pytest.raises(ValueError):
        layer_occlusion_bar(vertical_bar_span=vertical_bar_span)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("bar_colour", "red"),
        ("bar_colour", (0, "invalid", 0)),
        ("bar_colour", (0, 0, 1.5)),
        ("vertical_bar_span", "invalid"),
        ("vertical_bar_span", 1.5),
        ("y_offset", "invalid"),
        ("y_offset", 1.5),
        ("horizontal_bar_margin", "invalid"),
        ("horizontal_bar_margin", 1.5),
        ("landmark_paths", "invalid"),
        ("landmark_paths", 123),
        ("landmark_paths", [(0, "invalid"), (1, 2)]),
    ],
)
def test_layer_occlusion_bar_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_occlusion_bar(**{parameter: value})