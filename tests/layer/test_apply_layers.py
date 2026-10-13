import pytest
import pandas as pd
from pyfame import apply_layers, layer_mask

@pytest.mark.parametrize(
    "param, val",
    [
        ("file_paths", {}),
        ("layers", {}),
        ("layers", ["test", "test2"]),
        ("min_face_detection_confidence", "str"),
        ("min_face_presence_confidence", "str"),
        ("min_face_tracking_confidence", "str")
    ]
)
def test_apply_layers_invalid_param_types(file_paths_valid_file, param, val):
    mask = layer_mask()
    kwargs = {
        "file_paths" : file_paths_valid_file,
        "layers" : [mask]
    }
    kwargs[param] = val

    with pytest.raises(TypeError):
        apply_layers(
            **kwargs
        )

@pytest.mark.parametrize(
    "param, val",
    [
        ("file_paths", pd.DataFrame(columns=["xyx", "yzy"])),
        ("min_face_detection_confidence", -1.0),
        ("min_face_presence_confidence", 2.0),
        ("min_face_tracking_confidence", 2.0)
    ]
)
def test_apply_layers_invalid_param_values(file_paths_valid_file, param, val):
    mask = layer_mask()
    kwargs = {
        "file_paths" : file_paths_valid_file,
        "layers" : [mask]
    }
    kwargs[param] = val

    with pytest.raises(ValueError):
        apply_layers(
            **kwargs
        )

def test_apply_layer_write_outputs(data_paths, file_paths_valid_file):
    mask = layer_mask()
    apply_layers(file_paths_valid_file, [mask])

    assert any(p.is_file() for p in data_paths["processed"].rglob("*.mp4"))
    assert any(p.is_file() for p in data_paths["logs"].rglob("*.json"))