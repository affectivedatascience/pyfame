import pytest
from pyfame.layer.manipulations.colour._layer_colour_recolour import layer_colour_recolour, LayerColourRecolour

@pytest.mark.parametrize("magnitude", [0.0, 50.0, 100.0])
def test_layer_colour_recolour_valid_magnitude(magnitude):
    layer = layer_colour_recolour(magnitude=magnitude)

    assert isinstance(layer, LayerColourRecolour)


@pytest.mark.parametrize(
    "colour_scale",
    ["relative", "absolute", "RELATIVE", "ABSOLUTE"],
)
def test_layer_colour_recolour_valid_scale(colour_scale):
    layer = layer_colour_recolour(colour_scale=colour_scale)

    assert isinstance(layer, LayerColourRecolour)


@pytest.mark.parametrize(
    "focus_colour",
    [
        "red",
        "green",
        "blue",
        "yellow",
        "RED",
        "GREEN",
        "BLUE",
        "YELLOW",
        4,
        5,
        6,
        7,
    ],
)
def test_layer_colour_recolour_valid_focus_colour(focus_colour):
    layer = layer_colour_recolour(focus_colour=focus_colour)

    assert isinstance(layer, LayerColourRecolour)


def test_layer_colour_recolour_valid_landmark_path(landmark_path):
    layer = layer_colour_recolour(landmark_paths=landmark_path)

    assert isinstance(layer, LayerColourRecolour)


def test_layer_colour_recolour_valid_landmark_paths(landmark_paths):
    layer = layer_colour_recolour(landmark_paths=landmark_paths)

    assert isinstance(layer, LayerColourRecolour)

@pytest.mark.parametrize("magnitude", [-0.1, 100.1])
def test_layer_colour_recolour_invalid_magnitude(magnitude):
    with pytest.raises(ValueError):
        layer_colour_recolour(magnitude=magnitude)


@pytest.mark.parametrize(
    "focus_colour",
    [
        "purple",
        "orange",
        "",
        3,
        8,
    ],
)
def test_layer_colour_recolour_invalid_focus_colour(focus_colour):
    with pytest.raises(ValueError):
        layer_colour_recolour(focus_colour=focus_colour)


@pytest.mark.parametrize(
    "colour_scale",
    ["invalid", ""],
)
def test_layer_colour_recolour_invalid_scale(colour_scale):
    with pytest.raises(ValueError):
        layer_colour_recolour(colour_scale=colour_scale)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("landmark_paths", "invalid"),
        ("landmark_paths", 123),
        ("magnitude", "invalid"),
        ("colour_scale", 123),
    ],
)
def test_layer_colour_recolour_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_colour_recolour(**{parameter: value})