import pytest
from pyfame.layer.manipulations.overlay._layer_overlay import layer_overlay, LayerOverlay


def test_layer_overlay_defaults():
    layer = layer_overlay()

    assert isinstance(layer, LayerOverlay)

    params = layer.get_layer_parameters()
    assert params["overlay"] == "sunglasses"
    assert params["overlay_scale_factor"] is None
    assert params["x_offset"] == 0
    assert params["y_offset"] == 10
    assert params["pupil_scale_factor"] == 0.25


@pytest.mark.parametrize(
    "overlay, expected",
    [
        ("sunglasses", "sunglasses"),
        ("glasses", "glasses"),
        ("teardrop", "teardrop"),
        ("face_mask", "face_mask"),
        ("pupils", "pupils"),
        ("SUNGLASSES", "sunglasses"),
        ("GLASSES", "glasses"),
        ("TEARDROP", "teardrop"),
        ("FACE_MASK", "face_mask"),
        ("PUPILS", "pupils"),
        (43, "sunglasses"),
        (44, "glasses"),
        (45, "teardrop"),
        (46, "face_mask"),
        (47, "pupils"),
    ],
)
def test_layer_overlay_valid_overlay(overlay, expected):
    layer = layer_overlay(overlay=overlay)

    params = layer.get_layer_parameters()

    assert isinstance(layer, LayerOverlay)
    assert params["overlay"] == expected


@pytest.mark.parametrize(
    "overlay_scale_factor",
    [None, 0.1, 0.5, 1.0, 2.0],
)
def test_layer_overlay_valid_scale_factor(overlay_scale_factor):
    layer = layer_overlay(
        overlay_scale_factor=overlay_scale_factor
    )

    assert isinstance(layer, LayerOverlay)


@pytest.mark.parametrize(
    "x_offset",
    [-100, -10, 0, 10, 100],
)
def test_layer_overlay_valid_x_offset(x_offset):
    layer = layer_overlay(x_offset=x_offset)

    assert isinstance(layer, LayerOverlay)


@pytest.mark.parametrize(
    "y_offset",
    [-100, -10, 0, 10, 100],
)
def test_layer_overlay_valid_y_offset(y_offset):
    layer = layer_overlay(y_offset=y_offset)

    assert isinstance(layer, LayerOverlay)


@pytest.mark.parametrize(
    "pupil_scale_factor",
    [0.01, 0.13, 0.25, 0.5, 1.0],
)
def test_layer_overlay_valid_pupil_scale_factor(pupil_scale_factor):
    layer = layer_overlay(
        pupil_scale_factor=pupil_scale_factor
    )

    assert isinstance(layer, LayerOverlay)


@pytest.mark.parametrize(
    "overlay_scale_factor",
    [-1.0, -0.1, 0.0],
)
def test_layer_overlay_invalid_scale_factor(overlay_scale_factor):
    with pytest.raises(ValueError):
        layer_overlay(overlay_scale_factor=overlay_scale_factor)


@pytest.mark.parametrize(
    "overlay",
    [
        "invalid",
        "",
        "mask",
        "hat",
        "custom_overlay.png",
        42,
        48,
        -1,
    ],
)
def test_layer_overlay_invalid_overlay(overlay):
    with pytest.raises(ValueError):
        layer_overlay(overlay=overlay)


@pytest.mark.parametrize(
    "pupil_scale_factor",
    [-1.0, -0.1, 0.0, 1.01, 2.0],
)
def test_layer_overlay_invalid_pupil_scale_factor(pupil_scale_factor):
    with pytest.raises(ValueError):
        layer_overlay(pupil_scale_factor=pupil_scale_factor)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("overlay", 1.5),
        ("overlay", []),
        ("overlay", None),
        ("overlay_scale_factor", "invalid"),
        ("overlay_scale_factor", []),
        ("x_offset", "invalid"),
        ("x_offset", 1.5),
        ("y_offset", "invalid"),
        ("y_offset", 1.5),
        ("pupil_scale_factor", "invalid"),
        ("pupil_scale_factor", []),
        ("pupil_scale_factor", None),
    ],
)
def test_layer_overlay_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_overlay(**{parameter: value})