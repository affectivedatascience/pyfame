import pytest
import pandas as pd
from pyfame import analyse_facial_colour_means

@pytest.mark.parametrize(
    "file_paths",
    [
        None,
        [],
        {},
        "file1.mp4"
    ],
)
def test_analyse_facial_colour_means_file_paths_invalid_type(file_paths):
    with pytest.raises(TypeError):
        analyse_facial_colour_means(
            file_paths=file_paths
        )


@pytest.mark.parametrize(
    "cols",
    [
        [],
        ["absolute path"],
        ["relative path"],
        ["wrong", "columns"]
    ],
)
def test_analyse_facial_colour_means_file_paths_invalid_columns(cols):
    file_paths = pd.DataFrame(columns=cols)

    with pytest.raises(ValueError):
        analyse_facial_colour_means(
            file_paths=file_paths
        )

@pytest.mark.parametrize(
    "colour_space",
    [
        0,
        "lab",
        None,
        [],
        {},
        47.25
    ]
)
def test_analyse_facial_colour_means_colour_space_invalid(colour_space, file_paths_valid_file):
    with pytest.raises(ValueError):
        analyse_facial_colour_means(
            file_paths=file_paths_valid_file,
            colour_space=colour_space
        )

@pytest.mark.parametrize(
    "frame_step",
    [
        0.0,
        "test",
        [],
        {},
        None,
        -2
    ]
)
def test_analyse_facial_colour_means_frame_step_invalid(frame_step, file_paths_valid_file):
    with pytest.raises(ValueError):
        analyse_facial_colour_means(
            file_paths=file_paths_valid_file,
            frame_step=frame_step
        )

def test_analyse_facial_colour_means_result(file_paths_valid_file):
    result = analyse_facial_colour_means(
        file_paths=file_paths_valid_file,
        frame_step=20
    )

    assert isinstance(result, dict)
    assert all(isinstance(key, str) for key in result)
    assert all(isinstance(value, pd.DataFrame) for value in result.values())