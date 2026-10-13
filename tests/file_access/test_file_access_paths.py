import pytest
import pandas as pd
from pathlib import Path
from pyfame import make_paths, load_sample_data, load_user_data

def test_make_paths_invalid_params(valid_face_photo):
    with pytest.raises(TypeError):
        make_paths(root_folder=-1)

    with pytest.raises(OSError):
        make_paths(root_folder="./+_:inval*d^symbols?")

    with pytest.raises(OSError):
        make_paths(root_folder=valid_face_photo)

def test_make_paths_folder_creation(tmp_path):
    make_paths(
        root_folder=tmp_path
    )

    subdirs = ["raw", "processed", "analysis", "logs", "conversion"]

    for dir in subdirs:
        dir_path = tmp_path / dir
        assert dir_path.exists()

def test_load_user_data_invalid_params():
    with pytest.raises(TypeError):
        load_user_data(
            data_folder_name=1
        )

    with pytest.raises(TypeError):
        load_user_data(
            include_folders=[
                -1,
                2
            ]
        )

def test_load_user_data_return(file_paths_valid_file):
    df = load_user_data()

    assert isinstance(df, pd.DataFrame)
    assert all((Path(my_path).exists()) for my_path in df["Absolute Path"])

def test_load_sample_data_invalid_params():
    with pytest.raises(TypeError):
        load_user_data(
            data_folder_name=1
        )

def test_load_sample_data_return(tmp_path, data_paths):
    df = load_sample_data()

    sample_path = data_paths["raw"] / "samples"

    assert sample_path.exists()
    assert all(my_path.is_file() for my_path in sample_path.iterdir())
    assert isinstance(df, pd.DataFrame)
    assert all((Path(my_path).exists()) for my_path in df["Absolute Path"])
