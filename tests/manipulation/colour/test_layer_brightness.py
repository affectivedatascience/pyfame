import pytest
from pyfame.layer.manipulations.colour._layer_colour_brightness import LayerColourBrightness
from pyfame import layer_colour_brightness

@pytest.mark.parametrize("magnitude", [-100.0, 0.0, 100.0])
def test_layer_colour_brightness_valid_magnitude(magnitude):
    layer = layer_colour_brightness(magnitude=magnitude)
    assert isinstance(layer, LayerColourBrightness)


@pytest.mark.parametrize(
    "brightness_scale",
    ["relative", "absolute", "RELATIVE", "ABSOLUTE"],
)
def test_layer_colour_brightness_valid_scale(brightness_scale):
    layer = layer_colour_brightness(brightness_scale=brightness_scale)
    assert isinstance(layer, LayerColourBrightness)


def test_layer_colour_brightness_valid_landmark_path(landmark_path):
    layer = layer_colour_brightness(landmark_paths=landmark_path)
    assert isinstance(layer, LayerColourBrightness)


def test_layer_colour_brightness_valid_landmark_paths(landmark_paths):
    layer = layer_colour_brightness(landmark_paths=landmark_paths)
    assert isinstance(layer, LayerColourBrightness)


@pytest.mark.parametrize("magnitude", [-100.1, 100.1])
def test_layer_colour_brightness_invalid_magnitude(magnitude):
    with pytest.raises(ValueError):
        layer_colour_brightness(magnitude=magnitude)


@pytest.mark.parametrize("brightness_scale", ["invalid", ""])
def test_layer_colour_brightness_invalid_scale(brightness_scale):
    with pytest.raises(ValueError):
        layer_colour_brightness(brightness_scale=brightness_scale)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("landmark_paths", "invalid"),
        ("landmark_paths", 123),
        ("magnitude", "invalid"),
        ("brightness_scale", 123),
    ],
)
def test_layer_colour_brightness_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_colour_brightness(**{parameter: value})