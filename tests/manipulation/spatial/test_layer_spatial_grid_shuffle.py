import pytest

from pyfame.layer.manipulations.spatial._layer_spatial_grid_shuffle import (
    layer_spatial_grid_shuffle,
    LayerSpatialGridShuffle,
)


def test_layer_spatial_grid_shuffle_defaults():
    layer = layer_spatial_grid_shuffle()

    assert isinstance(layer, LayerSpatialGridShuffle)

    params = layer.get_layer_parameters()
    assert params["random_seed"] is None
    assert params["shuffle_method"] == "random"
    assert params["grid_square_size"] == 40
    assert params["grid_dimensions"] is None
    assert params["mask_overlap_threshold"] == 0.5
    assert params["cyclic_shift_amount"] == 1


@pytest.mark.parametrize(
    "shuffle_method, expected",
    [
        ("random", "random"),
        ("cyclic shift", "cyclic shift"),
        ("none", "none"),
        ("RANDOM", "random"),
        ("CYCLIC SHIFT", "cyclic shift"),
        ("NONE", "none"),
        (27, "random"),
        (28, "cyclic shift"),
        (48, "none"),
    ],
)
def test_layer_spatial_grid_shuffle_valid_method(shuffle_method, expected):
    layer = layer_spatial_grid_shuffle(shuffle_method=shuffle_method)

    params = layer.get_layer_parameters()

    assert isinstance(layer, LayerSpatialGridShuffle)
    assert params["shuffle_method"] == expected


@pytest.mark.parametrize(
    "random_seed",
    [None, 0, 1, 42, 1000],
)
def test_layer_spatial_grid_shuffle_valid_random_seed(random_seed):
    layer = layer_spatial_grid_shuffle(random_seed=random_seed)

    assert isinstance(layer, LayerSpatialGridShuffle)


@pytest.mark.parametrize(
    "grid_square_size",
    [1, 10, 40, 100],
)
def test_layer_spatial_grid_shuffle_valid_square_size(grid_square_size):
    layer = layer_spatial_grid_shuffle(
        grid_square_size=grid_square_size
    )

    assert isinstance(layer, LayerSpatialGridShuffle)


@pytest.mark.parametrize(
    "grid_dimensions",
    [
        None,
        (1, 1),
        (2, 2),
        (4, 6),
        (10, 10),
    ],
)
def test_layer_spatial_grid_shuffle_valid_dimensions(grid_dimensions):
    layer = layer_spatial_grid_shuffle(
        grid_dimensions=grid_dimensions
    )

    assert isinstance(layer, LayerSpatialGridShuffle)


@pytest.mark.parametrize(
    "mask_overlap_threshold",
    [0.01, 0.25, 0.5, 0.75, 1.0],
)
def test_layer_spatial_grid_shuffle_valid_overlap(mask_overlap_threshold):
    layer = layer_spatial_grid_shuffle(
        mask_overlap_threshold=mask_overlap_threshold
    )

    assert isinstance(layer, LayerSpatialGridShuffle)


@pytest.mark.parametrize(
    "cyclic_shift_amount",
    [1, 2, 5, 100],
)
def test_layer_spatial_grid_shuffle_valid_shift_amount(cyclic_shift_amount):
    layer = layer_spatial_grid_shuffle(
        cyclic_shift_amount=cyclic_shift_amount
    )

    assert isinstance(layer, LayerSpatialGridShuffle)


def test_layer_spatial_grid_shuffle_valid_landmark_path(landmark_path):
    layer = layer_spatial_grid_shuffle(
        landmark_paths=landmark_path
    )

    assert isinstance(layer, LayerSpatialGridShuffle)


def test_layer_spatial_grid_shuffle_valid_landmark_paths(landmark_paths):
    layer = layer_spatial_grid_shuffle(
        landmark_paths=landmark_paths
    )

    assert isinstance(layer, LayerSpatialGridShuffle)


@pytest.mark.parametrize(
    "shuffle_method",
    [
        "invalid",
        "",
        "shuffle",
        "cyclic-shift",
        26,
        29,
        47,
        49,
        -1,
    ],
)
def test_layer_spatial_grid_shuffle_invalid_method(shuffle_method):
    with pytest.raises(ValueError):
        layer_spatial_grid_shuffle(shuffle_method=shuffle_method)


@pytest.mark.parametrize(
    "random_seed",
    [-100, -1],
)
def test_layer_spatial_grid_shuffle_invalid_random_seed(random_seed):
    with pytest.raises(ValueError):
        layer_spatial_grid_shuffle(random_seed=random_seed)


@pytest.mark.parametrize(
    "grid_square_size",
    [-10, -1, 0],
)
def test_layer_spatial_grid_shuffle_invalid_square_size(grid_square_size):
    with pytest.raises(ValueError):
        layer_spatial_grid_shuffle(grid_square_size=grid_square_size)


@pytest.mark.parametrize(
    "grid_dimensions",
    [
        (0, 1),
        (1, 0),
        (-1, 5),
        (5, -1),
        (0, 0),
        (-1, -1),
        (1,),
        (1, 2, 3),
    ],
)
def test_layer_spatial_grid_shuffle_invalid_dimensions(grid_dimensions):
    with pytest.raises(ValueError):
        layer_spatial_grid_shuffle(grid_dimensions=grid_dimensions)


@pytest.mark.parametrize(
    "mask_overlap_threshold",
    [-1.0, -0.1, 0.0, 1.01, 2.0],
)
def test_layer_spatial_grid_shuffle_invalid_overlap(mask_overlap_threshold):
    with pytest.raises(ValueError):
        layer_spatial_grid_shuffle(
            mask_overlap_threshold=mask_overlap_threshold
        )


@pytest.mark.parametrize(
    "cyclic_shift_amount",
    [-10, -1, 0],
)
def test_layer_spatial_grid_shuffle_invalid_shift_amount(cyclic_shift_amount):
    with pytest.raises(ValueError):
        layer_spatial_grid_shuffle(
            cyclic_shift_amount=cyclic_shift_amount
        )


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("random_seed", "invalid"),
        ("random_seed", 1.5),
        ("shuffle_method", 1.5),
        ("shuffle_method", []),
        ("shuffle_method", None),
        ("grid_square_size", "invalid"),
        ("grid_square_size", 1.5),
        ("grid_dimensions", "invalid"),
        ("grid_dimensions", (1, "invalid")),
        ("grid_dimensions", (1.5, 2)),
        ("mask_overlap_threshold", "invalid"),
        ("cyclic_shift_amount", "invalid"),
        ("cyclic_shift_amount", 1.5),
        ("landmark_paths", "invalid"),
        ("landmark_paths", 123),
    ],
)
def test_layer_spatial_grid_shuffle_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_spatial_grid_shuffle(**{parameter: value})