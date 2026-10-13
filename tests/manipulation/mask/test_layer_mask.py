import pytest
from pyfame.layer.manipulations.mask._layer_mask import layer_mask, LayerMask


def test_layer_mask_defaults():
    layer = layer_mask()

    assert isinstance(layer, LayerMask)


def test_layer_mask_valid_landmark_path(landmark_path):
    layer = layer_mask(landmark_paths=landmark_path)

    assert isinstance(layer, LayerMask)


def test_layer_mask_valid_landmark_paths(landmark_paths):
    layer = layer_mask(landmark_paths=landmark_paths)

    assert isinstance(layer, LayerMask)


@pytest.mark.parametrize(
    "background_colour",
    [
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (128, 64, 32),
    ],
)
def test_layer_mask_valid_background_colour(background_colour):
    layer = layer_mask(background_colour=background_colour)

    assert isinstance(layer, LayerMask)

@pytest.mark.parametrize(
    "background_colour",
    [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (256, 0, 0),
        (0, 256, 0),
        (0, 0, 256),
        (0, 0),
        (0, 0, 0, 0),
    ],
)
def test_layer_mask_invalid_background_colour(background_colour):
    with pytest.raises(ValueError):
        layer_mask(background_colour=background_colour)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("background_colour", "black"),
        ("background_colour", 123),
        ("background_colour", (0, "invalid", 0)),
        ("background_colour", (0, 0, 1.5)),
        ("landmark_paths", "invalid"),
        ("landmark_paths", 123),
        ("landmark_paths", [(0, "invalid")]),
    ],
)
def test_layer_mask_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_mask(**{parameter: value})