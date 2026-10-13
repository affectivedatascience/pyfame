import pytest
import pandas as pd
from pyfame.file_access.conversion import standardise_image_dimensions, apply_conversion_image_to_video, apply_conversion_remux_to_mp4

@pytest.mark.parametrize(
    "parameter, value",
    [
        ("input_directory", -1),
        ("method", -1),
        ("pad_colour", {}),
        ("pad_colour", ("sum", "inv", "alid"))
    ]
)
def test_standardise_image_dims_invalid_param_types(tmp_path, parameter, value):
    kwargs = {
        "input_directory" : tmp_path
    }
    kwargs[parameter] = value

    with pytest.raises(TypeError):
        standardise_image_dimensions(
            **kwargs
        )

def test_standardise_image_dims_invalid_dir_format():
    with pytest.raises(OSError):
        standardise_image_dimensions(
            input_directory="./inval?d/d#r^ctr:y"
        )

@pytest.mark.parametrize(
    "param, value",
    [
        ("file_paths", []),
        ("output_filename", -1),
        ("frame_rate", []),
        ("repeat_duration_msec", "test"),
        ("blend_transition", {}),
        ("blended_frames_proportion", "test")
    ]
)
def test_apply_conversion_image_to_video_invalid_param_types(file_paths_valid_file, param, value):
    kwargs = {
        "file_paths" : file_paths_valid_file,
        "output_filename" : "test"
    }
    kwargs[param] = value

    with pytest.raises(TypeError):
        apply_conversion_image_to_video(
            **kwargs
        )

@pytest.mark.parametrize(
    "param, value",
    [
        ("file_paths", pd.DataFrame(columns=["test1", "test2"])),
        ("frame_rate", -5),
        ("frame_rate", 200),
        ("repeat_duration_msec", 50),
        ("blended_frames_proportion", 2.0)
    ]
)
def test_apply_conversion_image_to_video_invalid_param_values(file_paths_valid_file, param, value):
    kwargs = {
        "file_paths" : file_paths_valid_file,
        "output_filename" : "test"
    }
    kwargs[param] = value

    with pytest.raises(ValueError):
        apply_conversion_image_to_video(
            **kwargs
        )

def test_apply_conversion_remux_to_mp4_invalid_param():
    invalid_df = pd.DataFrame(
        columns=["tst", "xyx"]
    )

    with pytest.raises(TypeError):
        apply_conversion_remux_to_mp4(
            file_paths = []
        )

    with pytest.raises(ValueError):
        apply_conversion_remux_to_mp4(
            file_paths=invalid_df
        )