from pyfame.utils.exceptions import *
from typing import Any
import os
from pathlib import Path
import pandas as pd

def check_type(param:Any, accepted_types:list[type], iterable:bool=False) -> None:
    if not iterable:
        if not any(isinstance(param, t) for t in accepted_types):
            accepted_names = ", ".join(t.__name__ for t in accepted_types)
            raise TypeError(f"parameter of type {type(param).__name__} is invalid, expected one of: {accepted_names}.")
    else:
        for i in param:
            if not any(isinstance(i, t) for t in accepted_types):
                accepted_names = ", ".join(t.__name__ for t in accepted_types)
                raise TypeError(f"parameter of type {type(i).__name__} is invalid, expected one of: {accepted_names}.")

def check_value(param:Any, expected_values:list[Any]=None, **kwargs) -> None:
    minimum = None
    maximum = None
    full_range = False

    if kwargs.get("minimum") is not None:
        minimum = kwargs.get("minimum")
    if kwargs.get("maximum") is not None:
        maximum = kwargs.get("maximum")
    if all((val is not None) for val in [minimum, maximum]):
        full_range = True
    
    if expected_values is not None:
        if not any(param == val for val in expected_values):
            raise ValueError(f"Unrecognized value for input parameter.")
    elif full_range:
        if param < minimum or param > maximum:
            raise ValueError(f"Parameter must fall in the range [{minimum},{maximum}].")
    elif minimum is not None:
        if param < minimum:
            raise ValueError(f"Parameter must be >{minimum}.")
    elif maximum is not None:
        if param > maximum:
            raise ValueError(f"Parameter must be <{maximum}.")

def check_valid_path(path:str | Path) -> None:
    tst_path = Path(path)

    if not tst_path.exists():
        raise OSError("Provided path is invalid or does not exist in your file system.")

def check_valid_output_path(path: str | Path):
    my_path = Path(path)

    if not my_path.parent.exists():
        raise OSError("Provided output path is invalid or does not exist in your file system.")

    if not my_path.parent.is_dir():
        raise OSError("Provided parent path is not a directory.")
    
def check_is_dir(path:str | Path) -> None:
    if not os.path.isdir(path):
        raise OSError("Path string must be a path to a directory.")

def check_is_file(path:str | Path) -> None:
    if not os.path.isfile(path):
        raise OSError("Path string must be a path to a file.")

def check_valid_file_extension(extension:str, allowed_extensions:list[str] = ['.mp4', '.mov', '.jpg', '.jpeg', '.png', '.bmp']) -> None:
    if extension not in allowed_extensions:
        raise UnrecognizedExtensionError(f"Unrecognized file extension {extension}.")

def check_file_paths(file_paths: pd.DataFrame) -> None:
    if not isinstance(file_paths, pd.DataFrame):
        raise TypeError(
            "file_paths must be a pandas DataFrame."
        )

    required_columns = {"Absolute Path", "Relative Path"}

    if not required_columns.issubset(file_paths.columns):
        raise ValueError(
            "file_paths must contain 'Absolute Path' and 'Relative Path' columns."
        )