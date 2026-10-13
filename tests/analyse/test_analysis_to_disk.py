import pytest
import pandas as pd
from pathlib import Path
from pyfame import analysis_to_disk

@pytest.mark.parametrize(
    "parameter, value",
    [
        ("analysis_dictionary", -1),
        ("analysis_label", -1),
        ("working_directory_path", -1),
        ("working_directory_path", "./invalid/path/file.txt"),
        ("output_folder_name", -1)
    ]
)
def test_analysis_to_disk_invalid_parameters(data_paths, parameter, value):
    sample_df = pd.DataFrame(
        {
            "x" : [1,2,3,4],
            "y" : [3,4,5,6]
        }
    )

    sample_analysis_result = {
        "file1" : sample_df
    }

    kwargs = {
        "analysis_dictionary" : sample_analysis_result,
        "analysis_label" : "test",
        "working_directory_path" : Path.cwd() / "data"
    }
    kwargs[parameter] = value

    with pytest.raises(ValueError):
        analysis_to_disk(
            **kwargs
        )

def test_analysis_to_disk_write(data_paths):
    sample_df = pd.DataFrame(
        {
            "timestamp" : [0.0, 0.1, 0.2, 0.3],
            "x" : [1,2,3,4],
            "y" : [3,4,5,6]
        }
    )

    sample_analysis_result = {
        "file1" : sample_df
    }

    analysis_to_disk(
        analysis_dictionary=sample_analysis_result,
        analysis_label="test",
        working_directory_path=Path.cwd() / "data",
        output_folder_name="dummy"
    )

    test_path = data_paths["analysis"] / "dummy" / "file1.json"
    assert test_path.exists()
    assert test_path.is_file()