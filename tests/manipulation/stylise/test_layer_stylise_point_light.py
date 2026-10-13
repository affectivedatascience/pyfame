import pytest

from pyfame.layer.manipulations.stylise._layer_stylise_point_light import (
    layer_stylise_point_light,
    LayerStylisePointLight,
)


def test_layer_stylise_point_light_defaults():
    layer = layer_stylise_point_light()

    assert isinstance(layer, LayerStylisePointLight)

    params = layer.get_layer_parameters()

    assert params["point_density"] == 1.0
    assert params["point_colour"] == (255, 255, 255)
    assert params["point_radius"] == 3
    assert params["display_history_vectors"] is False
    assert params["history_window_msec"] == 500
    assert params["history_method"] == "origin"
    assert params["history_vector_colour"] == (0, 0, 255)
    assert params["maintain_background"] is False
    assert params["invert_colours"] is False


@pytest.mark.parametrize(
    "history_method, expected",
    [
        ("origin", "origin"),
        ("relative", "relative"),
        ("ORIGIN", "origin"),
        ("RELATIVE", "relative"),
        (32, "origin"),
        (33, "relative"),
    ],
)
def test_layer_stylise_point_light_valid_history_method(history_method, expected):
    layer = layer_stylise_point_light(
        history_method=history_method
    )

    assert isinstance(layer, LayerStylisePointLight)
    assert layer.get_layer_parameters()["history_method"] == expected


@pytest.mark.parametrize(
    "point_density",
    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
)
def test_layer_stylise_point_light_valid_density(point_density):
    layer = layer_stylise_point_light(
        point_density=point_density
    )

    assert isinstance(layer, LayerStylisePointLight)
    assert layer.get_layer_parameters()["point_density"] == point_density


@pytest.mark.parametrize(
    "point_radius",
    [1, 2, 3, 10, 25],
)
def test_layer_stylise_point_light_valid_radius(point_radius):
    layer = layer_stylise_point_light(
        point_radius=point_radius
    )

    assert isinstance(layer, LayerStylisePointLight)
    assert layer.get_layer_parameters()["point_radius"] == point_radius


@pytest.mark.parametrize(
    "point_colour",
    [
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (128, 64, 32),
    ],
)
def test_layer_stylise_point_light_valid_point_colour(point_colour):
    layer = layer_stylise_point_light(
        point_colour=point_colour
    )

    assert isinstance(layer, LayerStylisePointLight)
    assert layer.get_layer_parameters()["point_colour"] == point_colour


@pytest.mark.parametrize(
    "history_vector_colour",
    [
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (128, 64, 32),
    ],
)
def test_layer_stylise_point_light_valid_history_colour(history_vector_colour):
    layer = layer_stylise_point_light(
        history_vector_colour=history_vector_colour
    )

    assert isinstance(layer, LayerStylisePointLight)
    assert layer.get_layer_parameters()["history_vector_colour"] == history_vector_colour


@pytest.mark.parametrize(
    "history_window_msec",
    [1, 100, 500, 1000, 5000],
)
def test_layer_stylise_point_light_valid_history_window(history_window_msec):
    layer = layer_stylise_point_light(
        history_window_msec=history_window_msec
    )

    assert isinstance(layer, LayerStylisePointLight)
    assert layer.get_layer_parameters()["history_window_msec"] == history_window_msec


@pytest.mark.parametrize(
    "parameter",
    [
        "display_history_vectors",
        "maintain_background",
        "invert_colours",
    ],
)
@pytest.mark.parametrize("value", [True, False])
def test_layer_stylise_point_light_valid_booleans(parameter, value):
    layer = layer_stylise_point_light(**{parameter: value})

    assert isinstance(layer, LayerStylisePointLight)
    assert layer.get_layer_parameters()[parameter] is value


def test_layer_stylise_point_light_valid_landmark_path(landmark_path):
    layer = layer_stylise_point_light(
        landmark_paths=landmark_path
    )

    assert isinstance(layer, LayerStylisePointLight)
    assert layer.get_layer_parameters()["landmark_paths"] == landmark_path


def test_layer_stylise_point_light_valid_landmark_paths(landmark_paths):
    layer = layer_stylise_point_light(
        landmark_paths=landmark_paths
    )

    assert isinstance(layer, LayerStylisePointLight)
    assert layer.get_layer_parameters()["landmark_paths"] == landmark_paths


@pytest.mark.parametrize(
    "history_method",
    [
        "invalid",
        "",
        "history",
        31,
        34,
        -1,
        True,
        False,
    ],
)
def test_layer_stylise_point_light_invalid_history_method(history_method):
    with pytest.raises(ValueError):
        layer_stylise_point_light(history_method=history_method)


@pytest.mark.parametrize(
    "point_density",
    [-1.0, -0.1, 0.0, 1.01, 2.0],
)
def test_layer_stylise_point_light_invalid_density(point_density):
    with pytest.raises(ValueError):
        layer_stylise_point_light(point_density=point_density)


@pytest.mark.parametrize(
    "point_radius",
    [-10, -1, 0, 26, 100],
)
def test_layer_stylise_point_light_invalid_radius(point_radius):
    with pytest.raises(ValueError):
        layer_stylise_point_light(point_radius=point_radius)


@pytest.mark.parametrize(
    "history_window_msec",
    [-1000, -1, 0],
)
def test_layer_stylise_point_light_invalid_history_window(history_window_msec):
    with pytest.raises(ValueError):
        layer_stylise_point_light(
            history_window_msec=history_window_msec
        )


@pytest.mark.parametrize(
    "parameter",
    ["point_colour", "history_vector_colour"],
)
@pytest.mark.parametrize(
    "colour",
    [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (256, 0, 0),
        (0, 256, 0),
        (0, 0, 256),
        (255, 255),
        (255, 255, 255, 255),
    ],
)
def test_layer_stylise_point_light_invalid_colours(parameter, colour):
    with pytest.raises(ValueError):
        layer_stylise_point_light(**{parameter: colour})


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("landmark_paths", None),
        ("landmark_paths", "invalid"),
        ("landmark_paths", 123),
        ("landmark_paths", [(0, "invalid")]),
        ("point_density", "invalid"),
        ("point_density", None),
        ("point_radius", "invalid"),
        ("point_radius", 1.5),
        ("point_radius", None),
        ("point_colour", "invalid"),
        ("point_colour", (1.5, 0, 0)),
        ("history_vector_colour", "invalid"),
        ("history_vector_colour", (0, 1.5, 0)),
        ("history_window_msec", "invalid"),
        ("history_window_msec", 1.5),
        ("history_method", 1.5),
        ("history_method", None),
        ("history_method", []),
        ("display_history_vectors", "invalid"),
        ("maintain_background", "invalid"),
        ("invert_colours", "invalid"),
    ],
)
def test_layer_stylise_point_light_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_stylise_point_light(**{parameter: value})