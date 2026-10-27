import pytest
from pyfame.layer.manipulations.occlusion._layer_occlusion_noise import layer_occlusion_noise, LayerOcclusionNoise


def test_layer_occlusion_noise_defaults():
    layer = layer_occlusion_noise()

    assert isinstance(layer, LayerOcclusionNoise)


@pytest.mark.parametrize(
    "noise_method, expected",
    [
        ("pixelate", "pixelate"),
        ("salt and pepper", "salt and pepper"),
        ("gaussian", "gaussian"),
        ("PIXELATE", "pixelate"),
        ("SALT AND PEPPER", "salt and pepper"),
        ("GAUSSIAN", "gaussian"),
        (18, "pixelate"),
        (19, "salt and pepper"),
        (20, "gaussian"),
    ],
)
def test_layer_occlusion_noise_valid_method(noise_method, expected):
    layer = layer_occlusion_noise(noise_method=noise_method)

    params = layer.get_layer_parameters()

    assert isinstance(layer, LayerOcclusionNoise)
    assert params["noise_method"] == expected


@pytest.mark.parametrize(
    "noise_probability",
    [0.0, 0.25, 0.5, 1.0],
)
def test_layer_occlusion_noise_valid_probability(noise_probability):
    layer = layer_occlusion_noise(noise_probability=noise_probability)

    assert isinstance(layer, LayerOcclusionNoise)


@pytest.mark.parametrize(
    "pixel_size",
    [4, 16, 32, 64],
)
def test_layer_occlusion_noise_valid_pixel_size(pixel_size):
    layer = layer_occlusion_noise(pixel_size=pixel_size)

    assert isinstance(layer, LayerOcclusionNoise)


@pytest.mark.parametrize(
    "gaussian_mean",
    [-10.0, 0.0, 10.0],
)
def test_layer_occlusion_noise_valid_mean(gaussian_mean):
    layer = layer_occlusion_noise(gaussian_mean=gaussian_mean)

    assert isinstance(layer, LayerOcclusionNoise)


@pytest.mark.parametrize(
    "gaussian_std",
    [0.0, 0.5, 1.0, 10.0],
)
def test_layer_occlusion_noise_valid_standard_deviation(gaussian_std):
    layer = layer_occlusion_noise(gaussian_std=gaussian_std)

    assert isinstance(layer, LayerOcclusionNoise)


@pytest.mark.parametrize(
    "random_seed",
    [None, 0, 1, 42, 1000],
)
def test_layer_occlusion_noise_valid_random_seed(random_seed):
    layer = layer_occlusion_noise(random_seed=random_seed)

    assert isinstance(layer, LayerOcclusionNoise)


def test_layer_occlusion_noise_valid_landmark_path(landmark_path):
    layer = layer_occlusion_noise(landmark_paths=landmark_path)

    assert isinstance(layer, LayerOcclusionNoise)


def test_layer_occlusion_noise_valid_landmark_paths(landmark_paths):
    layer = layer_occlusion_noise(landmark_paths=landmark_paths)

    assert isinstance(layer, LayerOcclusionNoise)


@pytest.mark.parametrize(
    "noise_method",
    [
        "invalid",
        "",
        "speckle",
        17,
        21,
    ],
)
def test_layer_occlusion_noise_invalid_method(noise_method):
    with pytest.raises(ValueError):
        layer_occlusion_noise(noise_method=noise_method)


@pytest.mark.parametrize(
    "noise_probability",
    [
        -1.0,
        -0.1,
        1.01,
        2.0,
    ],
)
def test_layer_occlusion_noise_invalid_probability(noise_probability):
    with pytest.raises(ValueError):
        layer_occlusion_noise(noise_probability=noise_probability)


@pytest.mark.parametrize(
    "pixel_size",
    [-1, 0, 1, 3],
)
def test_layer_occlusion_noise_invalid_pixel_size(pixel_size):
    with pytest.raises(ValueError):
        layer_occlusion_noise(pixel_size=pixel_size)


@pytest.mark.parametrize(
    "gaussian_std",
    [-0.1, -1.0],
)
def test_layer_occlusion_noise_invalid_standard_deviation(gaussian_std):
    with pytest.raises(ValueError):
        layer_occlusion_noise(gaussian_std=gaussian_std)


@pytest.mark.parametrize(
    "random_seed",
    [-100, -1],
)
def test_layer_occlusion_noise_invalid_random_seed(random_seed):
    with pytest.raises(ValueError):
        layer_occlusion_noise(random_seed=random_seed)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("noise_method", 1.5),
        ("noise_method", []),
        ("noise_probability", "invalid"),
        ("pixel_size", "invalid"),
        ("pixel_size", 3.5),
        ("gaussian_mean", "invalid"),
        ("gaussian_std", "invalid"),
        ("random_seed", "invalid"),
        ("random_seed", 1.5),
        ("landmark_paths", "invalid"),
        ("landmark_paths", 123),
    ],
)
def test_layer_occlusion_noise_invalid_types(parameter, value):
    with pytest.raises(ValueError):
        layer_occlusion_noise(**{parameter: value})