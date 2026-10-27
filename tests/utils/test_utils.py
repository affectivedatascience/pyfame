import pytest

from pyfame.utils import get_variable_name, get_landmark_names
from pyfame.landmark.facial_landmarks import *

def test_get_variable_name_local():
    test_variable = 12345

    result = get_variable_name(test_variable, locals())

    assert result == "test_variable"


def test_get_variable_name_global():
    namespace = {"global_variable": 12345}

    result = get_variable_name(12345, namespace)

    assert result == "global_variable"


@pytest.mark.parametrize(
    "value",
    [
        42,
        "hello",
        [1, 2, 3],
        {"key": "value"},
    ],
)
def test_get_variable_name_valid_types(value):
    namespace = {"test_variable": value}

    result = get_variable_name(value, namespace)

    assert result == "test_variable"


@pytest.mark.parametrize(
    "namespace",
    [
        None,
        "invalid",
        123,
        [],
        (),
    ],
)
def test_get_variable_name_invalid_namespace(namespace):
    with pytest.raises(TypeError):
        get_variable_name(42, namespace)


def test_get_variable_name_missing_variable():
    namespace = {"existing_variable": 123}

    with pytest.raises(IndexError):
        get_variable_name(456, namespace)


# -------------------------
# get_landmark_names()
# -------------------------

def test_get_landmark_names_single_path():
    result = get_landmark_names(
        LANDMARK_FACE_OVAL
    )

    assert result == "LANDMARK_FACE_OVAL"


def test_get_landmark_names_multiple_paths():
    landmarks = [
        LANDMARK_FACE_OVAL,
        LANDMARK_LEFT_EYE,
    ]

    result = get_landmark_names(landmarks)

    assert result == [
        "LANDMARK_FACE_OVAL",
        "LANDMARK_LEFT_EYE",
    ]


def test_get_landmark_names_mixed_paths():
    landmarks = [
        LANDMARK_FACE_OVAL,
        [(9999, 9998), (9998, 9997)],
    ]

    result = get_landmark_names(landmarks)

    assert result == [
        "LANDMARK_FACE_OVAL",
        [(9999, 9998), (9998, 9997)],
    ]


@pytest.mark.parametrize(
    "landmarks",
    [
        None,
        "invalid",
        123,
        {},
        (),
    ],
)
def test_get_landmark_names_invalid_input_type(landmarks):
    with pytest.raises(TypeError):
        get_landmark_names(landmarks)


@pytest.mark.parametrize(
    "landmarks",
    [
        [(0, "invalid")],
        [("invalid", 1)],
        [[(0, "invalid")]],
    ],
)
def test_get_landmark_names_invalid_landmark_types(landmarks):
    with pytest.raises(TypeError):
        get_landmark_names(landmarks)


@pytest.mark.parametrize(
    "landmarks",
    [
        [],
        [[]],
    ],
)
def test_get_landmark_names_empty_paths(landmarks):
    with pytest.raises(ValueError):
        get_landmark_names(landmarks)
