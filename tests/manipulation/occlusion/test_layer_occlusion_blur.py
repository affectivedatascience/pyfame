import pytest
from pyfame.layer.manipulations.occlusion._layer_occlusion_blur import layer_occlusion_blur, LayerOcclusionBlur


def test_layer_occlusion_blur_defaults():
    layer = layer_occlusion_blur()

    assert isinstance(layer, LayerOcclusionBlur)


@pytest.mark.parametrize(
    "blur_method, expected",
    [
        ("average", "average"),
        ("gaussian", "gaussian"),
        ("median", "median"),
        ("AVERAGE", "average"),
        ("GAUSSIAN", "gaussian"),
        ("MEDIAN", "median"),
        (11, "average"),
        (12, "gaussian"),
        (13, "median"),
    ],
)
def test_layer_occlusion_blur_valid_method(blur_method, expected):
    layer = layer_occlusion_blur(blur_method=blur_method)

    params = layer.get_layer_parameters()

    assert isinstance(layer, LayerOcclusionBlur)
    assert params["blur_method"] == expected


@pytest.mark.parametrize(
    "max_kernel_size",
    [3, 5, 15, 29, 31],
)
def test_layer_occlusion_blur_valid_kernel_size(max_kernel_size):
    layer = layer_occlusion_blur(max_kernel_size=max_kernel_size)

    assert isinstance(layer, LayerOcclusionBlur)


def test_layer_occlusion_blur_valid_landmark_path(landmark_path):
    layer = layer_occlusion_blur(landmark_paths=landmark_path)

    assert isinstance(layer, LayerOcclusionBlur)


def test_layer_occlusion_blur_valid_landmark_paths(landmark_paths):
    layer = layer_occlusion_blur(landmark_paths=landmark_paths)

    assert isinstance(layer, LayerOcclusionBlur)


def test_layer_occlusion_blur_parameter_passthrough():
    layer = layer_occlusion_blur(
        blur_method="median",
        max_kernel_size=21,
    )

    params = layer.get_layer_parameters()

    assert params["blur_method"] == "median"
    assert params["max_kernel_size"] == 21
    

@pytest.mark.parametrize(
    "blur_method",
    [
        "invalid",
        "",
        "bilateral",
        10,
        14,
    ],
)
def test_layer_occlusion_blur_invalid_method(blur_method):
    with pytest.raises(ValueError):
        layer_occlusion_blur(blur_method=blur_method)


@pytest.mark.parametrize(
    "max_kernel_size",
    [
        -1,
        0,
        1,
        2,
        4,
        30,
        32,
        33,
    ],
)
def test_layer_occlusion_blur_invalid_kernel_size(max_kernel_size):
    with pytest.raises(ValueError):
        layer_occlusion_blur(max_kernel_size=max_kernel_size)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("blur_method", 1.5),
        ("blur_method", []),
        ("max_kernel_size", "invalid"),
        ("max_kernel_size", 3.5),
        ("landmark_paths", "invalid"),
        ("landmark_paths", 123),
    ],
)
def test_layer_occlusion_blur_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_occlusion_blur(**{parameter: value})