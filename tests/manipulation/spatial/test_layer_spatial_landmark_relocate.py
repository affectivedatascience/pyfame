import pytest

from pyfame.layer.manipulations.spatial._layer_spatial_landmark_relocate import (
    layer_spatial_landmark_relocate,
    LayerSpatialLandmarkRelocate,
    LandmarkRelocateSpec
)

from pyfame.utils import FaceAnchor


def test_layer_spatial_landmark_relocate_defaults():
    layer = layer_spatial_landmark_relocate()

    assert isinstance(layer, LayerSpatialLandmarkRelocate)

    params = layer.get_layer_parameters()

    assert params["max_random_offset"] == 0.15
    assert params["out_greyscale"] is True


@pytest.mark.parametrize(
    "random_seed",
    [None, 1, 42, 1000],
)
def test_layer_spatial_landmark_relocate_valid_random_seed(random_seed):
    layer = layer_spatial_landmark_relocate(random_seed=random_seed)

    assert isinstance(layer, LayerSpatialLandmarkRelocate)


@pytest.mark.parametrize(
    "max_random_offset",
    [0.0, 0.1, 0.15, 0.5, 1.0],
)
def test_layer_spatial_landmark_relocate_valid_max_offset(max_random_offset):
    layer = layer_spatial_landmark_relocate(
        max_random_offset=max_random_offset
    )

    assert isinstance(layer, LayerSpatialLandmarkRelocate)


@pytest.mark.parametrize(
    "out_greyscale",
    [True, False],
)
def test_layer_spatial_landmark_relocate_valid_greyscale(out_greyscale):
    layer = layer_spatial_landmark_relocate(
        out_greyscale=out_greyscale
    )

    assert isinstance(layer, LayerSpatialLandmarkRelocate)


@pytest.mark.parametrize("anchor", list(FaceAnchor))
def test_layer_spatial_landmark_relocate_valid_anchors(anchor):
    spec = LandmarkRelocateSpec(anchor=anchor)

    layer = layer_spatial_landmark_relocate(
        landmark_relocate_specs={0: spec}
    )

    assert isinstance(layer, LayerSpatialLandmarkRelocate)


@pytest.mark.parametrize(
    "rotation_deg",
    [-360.0, -90.0, 0.0, 90.0, 360.0],
)
def test_layer_spatial_landmark_relocate_valid_rotation(rotation_deg):
    spec = LandmarkRelocateSpec(
        anchor=FaceAnchor.SUBJECT_CENTER,
        rotation_deg=rotation_deg,
    )

    layer = layer_spatial_landmark_relocate(
        landmark_relocate_specs={0: spec}
    )

    assert isinstance(layer, LayerSpatialLandmarkRelocate)


@pytest.mark.parametrize(
    "offsets",
    [
        (-1.0, -1.0),
        (-0.5, 0.5),
        (0.0, 0.0),
        (0.5, -0.5),
        (1.0, 1.0),
    ],
)
def test_layer_spatial_landmark_relocate_valid_offsets(offsets):
    spec = LandmarkRelocateSpec(
        anchor=FaceAnchor.SUBJECT_CENTER,
        offsets=offsets,
    )

    layer = layer_spatial_landmark_relocate(
        landmark_relocate_specs={0: spec}
    )

    assert isinstance(layer, LayerSpatialLandmarkRelocate)


@pytest.mark.parametrize("key", [0, 1, 2, 3])
def test_layer_spatial_landmark_relocate_valid_spec_keys(key):
    spec = LandmarkRelocateSpec(
        anchor=FaceAnchor.SUBJECT_CENTER
    )

    layer = layer_spatial_landmark_relocate(
        landmark_relocate_specs={key: spec}
    )

    assert isinstance(layer, LayerSpatialLandmarkRelocate)
    assert layer.relocate_keys == [key]


def test_layer_spatial_landmark_relocate_valid_partial_specs():
    specs = {
        0: LandmarkRelocateSpec(
            anchor=FaceAnchor.SUBJECT_UPPER_RIGHT
        ),
        3: LandmarkRelocateSpec(
            anchor=FaceAnchor.SUBJECT_LOWER_CENTER
        ),
    }

    layer = layer_spatial_landmark_relocate(
        landmark_relocate_specs=specs
    )

    assert isinstance(layer, LayerSpatialLandmarkRelocate)
    assert set(layer.relocate_keys) == {0, 3}


def test_layer_spatial_landmark_relocate_valid_all_specs():
    specs = {
        0: LandmarkRelocateSpec(
            anchor=FaceAnchor.SUBJECT_UPPER_RIGHT
        ),
        1: LandmarkRelocateSpec(
            anchor=FaceAnchor.SUBJECT_UPPER_LEFT
        ),
        2: LandmarkRelocateSpec(
            anchor=FaceAnchor.SUBJECT_CENTER
        ),
        3: LandmarkRelocateSpec(
            anchor=FaceAnchor.SUBJECT_LOWER_CENTER
        ),
    }

    layer = layer_spatial_landmark_relocate(
        landmark_relocate_specs=specs
    )

    assert isinstance(layer, LayerSpatialLandmarkRelocate)
    assert set(layer.relocate_keys) == {0, 1, 2, 3}


@pytest.mark.parametrize(
    "random_seed",
    [-100, -1, 0],
)
def test_layer_spatial_landmark_relocate_invalid_random_seed(random_seed):
    with pytest.raises(ValueError):
        layer_spatial_landmark_relocate(random_seed=random_seed)


@pytest.mark.parametrize(
    "max_random_offset",
    [-1.0, -0.1, 1.01, 2.0],
)
def test_layer_spatial_landmark_relocate_invalid_max_offset(max_random_offset):
    with pytest.raises(ValueError):
        layer_spatial_landmark_relocate(
            max_random_offset=max_random_offset
        )


@pytest.mark.parametrize(
    "offsets",
    [
        (-1.01, 0.0),
        (1.01, 0.0),
        (0.0, -1.01),
        (0.0, 1.01),
    ],
)
def test_layer_spatial_landmark_relocate_invalid_offsets(offsets):
    with pytest.raises(ValueError):
        layer_spatial_landmark_relocate(
            landmark_relocate_specs={
                0: {
                    "anchor": FaceAnchor.SUBJECT_CENTER,
                    "offsets": offsets,
                }
            }
        )


@pytest.mark.parametrize(
    "invalid_key",
    [-1, 4, 5, 100],
)
def test_layer_spatial_landmark_relocate_invalid_spec_keys(invalid_key):
    with pytest.raises(ValueError):
        layer_spatial_landmark_relocate(
            landmark_relocate_specs={
                invalid_key: {
                    "anchor": FaceAnchor.SUBJECT_CENTER,
                }
            }
        )


def test_layer_spatial_landmark_relocate_empty_specs():
    with pytest.raises(ValueError):
        layer_spatial_landmark_relocate(
            landmark_relocate_specs={}
        )


@pytest.mark.parametrize(
    "anchor",
    ["invalid", None, 123, (0.75, 0.75)],
)
def test_layer_spatial_landmark_relocate_invalid_anchor(anchor):
    with pytest.raises(ValueError):
        layer_spatial_landmark_relocate(
            landmark_relocate_specs={
                0: {"anchor": anchor}
            }
        )


@pytest.mark.parametrize(
    "offsets",
    [
        "invalid",
        (0.0,),
        (0.0, 0.0, 0.0),
        ("invalid", 0.0),
        (0.0, None),
    ],
)
def test_layer_spatial_landmark_relocate_invalid_offset_types(offsets):
    with pytest.raises(ValueError):
        layer_spatial_landmark_relocate(
            landmark_relocate_specs={
                0: {
                    "anchor": FaceAnchor.SUBJECT_CENTER,
                    "offsets": offsets,
                }
            }
        )


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("random_seed", "invalid"),
        ("random_seed", 1.5),
        ("max_random_offset", "invalid"),
        ("out_greyscale", "invalid"),
        ("landmark_relocate_specs", "invalid"),
        ("landmark_relocate_specs", []),
        ("landmark_relocate_specs", {0: "invalid"}),
        ("landmark_relocate_specs", {0: {"rotation_deg": 30.0}}),
        (
            "landmark_relocate_specs",
            {
                0: {
                    "anchor": FaceAnchor.SUBJECT_CENTER,
                    "rotation_deg": "invalid",
                }
            },
        ),
    ],
)
def test_layer_spatial_landmark_relocate_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_spatial_landmark_relocate(**{parameter: value})