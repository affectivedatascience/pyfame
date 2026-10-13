import pytest
import pandas as pd
from pyfame import analyse_optical_flow_dense

# Testing file_paths handling
@pytest.mark.parametrize(
    "file_paths",
    [
        None,
        [],
        {},
        "file1.mp4"
    ],
)
def test_analyse_optical_flow_dense_file_paths_invalid_type(file_paths):
    with pytest.raises(TypeError):
        analyse_optical_flow_dense(
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
def test_analyse_optical_flow_dense_file_paths_invalid_columns(cols):
    file_paths = pd.DataFrame(columns=cols)
    with pytest.raises(ValueError):
        analyse_optical_flow_dense(
            file_paths=file_paths
        )

@pytest.mark.parametrize(
    "parameter, value",
    [
        ("frame_step", "test"),
        ("output_visualization", "test"),
        ("precise_colour_scale", "test"),
        ("display_legend", "test"),
        ("legend_position", bool),
        ("legend_position", "invalid str")
    ]
)
def test_analyse_optical_flow_dense_invalid_parameter_values(file_paths_valid_file, parameter, value):
    with pytest.raises(ValueError):
        kwargs = {parameter : value}

        analyse_optical_flow_dense(
            file_paths=file_paths_valid_file,
            **kwargs
        )

def test_analyse_optical_flow_dense_return(file_paths_valid_file):
    result = analyse_optical_flow_dense(
        file_paths_valid_file,
        frame_step=25
    )

    analysis, viz = result

    assert isinstance(analysis, dict)
    assert all(isinstance(key, str) for key in analysis)
    assert all(isinstance(value, pd.DataFrame) for value in analysis.values())