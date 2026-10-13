import pytest
import pandas as pd
from pyfame import analyse_optical_flow_sparse

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
def test_analyse_optical_flow_sparse_file_paths_invalid_type(file_paths):
    with pytest.raises(TypeError):
        analyse_optical_flow_sparse(
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
def test_analyse_optical_flow_sparse_file_paths_invalid_columns(cols):
    file_paths = pd.DataFrame(columns=cols)
    with pytest.raises(ValueError):
        analyse_optical_flow_sparse(
            file_paths=file_paths
        )

@pytest.mark.parametrize(
    "parameter, value",
    [
        ("landmark_idx_to_track", "test"),
        ("max_points", "test"),
        ("flow_accuracy_threshold", "test"),
        ("arrow_thickness", "test"),
        ("display_legend", "test"),
        ("legend_position", 2.0),
        ("precise_colour_scale", "test"),
        ("stats_detail_level", 2.0),
        ("frame_step", "test")
        ("output_visualization", "test")
    ]
)
def test_analyse_optical_flow_sparse_invalid_parameter_values(file_paths_valid_file, parameter, value):
    with pytest.raises(ValueError):
        kwargs = {parameter : value}
        analyse_optical_flow_sparse(
            file_paths=file_paths_valid_file
            **kwargs
        )

@pytest.mark.parametrize(
        "parameter, value",
        [
            ("landmark_idx_to_track", [-1]),
            ("landmark_idx_to_track", [500]),
            ("landmark_idx_to_track", []),
            ("flow_accuracy_threshold", 2.0),
            ("legend_position", "invalid str"),
            ("stats_detail_level", "invalid str"),
        ]
)
def test_analyse_optical_flow_sparse_parameter_bounds(file_paths_valid_file, parameter, value):
    with pytest.raises(ValueError):
        kwargs = {parameter : value}
        analyse_optical_flow_sparse(
            file_paths=file_paths_valid_file,
            **kwargs
        )

def test_analyse_optical_flow_sparse_return(file_paths_valid_file):
    result = analyse_optical_flow_sparse(
        file_paths=file_paths_valid_file,
        max_points=10,
        frame_step=20
    )

    analysis, output = result

    assert isinstance(analysis, dict)
    assert all(isinstance(key, str) for key in analysis)
    assert all(isinstance(value, pd.DataFrame) for value in analysis.values())