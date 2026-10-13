import pytest
from pyfame.layer.manipulations.colour._layer_colour_saturation import LayerColourSaturation
from pyfame import layer_colour_saturation

@pytest.mark.parametrize("magnitude", [-100.0, 0.0, 100.0])
def test_layer_colour_saturation_valid_magnitude(magnitude):
    layer = layer_colour_saturation(magnitude=magnitude)

    assert isinstance(layer, LayerColourSaturation)


@pytest.mark.parametrize(
    "saturation_scale",
    ["relative", "absolute", "RELATIVE", "ABSOLUTE"],
)
def test_layer_colour_saturation_valid_scale(saturation_scale):
    layer = layer_colour_saturation(saturation_scale=saturation_scale)

    assert isinstance(layer, LayerColourSaturation)


def test_layer_colour_saturation_valid_landmark_path(landmark_path):
    layer = layer_colour_saturation(landmark_paths=landmark_path)

    assert isinstance(layer, LayerColourSaturation)


def test_layer_colour_saturation_valid_landmark_paths(landmark_paths):
    layer = layer_colour_saturation(landmark_paths=landmark_paths)

    assert isinstance(layer, LayerColourSaturation)


@pytest.mark.parametrize("magnitude", [-100.1, 100.1])
def test_layer_colour_saturation_invalid_magnitude(magnitude):
    with pytest.raises(ValueError):
        layer_colour_saturation(magnitude=magnitude)


@pytest.mark.parametrize(
    "saturation_scale",
    ["invalid", ""],
)
def test_layer_colour_saturation_invalid_scale(saturation_scale):
    with pytest.raises(ValueError):
        layer_colour_saturation(saturation_scale=saturation_scale)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("landmark_paths", "invalid"),
        ("landmark_paths", 123),
        ("magnitude", "invalid"),
        ("saturation_scale", 123),
    ],
)
def test_layer_colour_saturation_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_colour_saturation(**{parameter: value})