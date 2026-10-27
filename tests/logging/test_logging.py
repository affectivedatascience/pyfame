import json
import pytest
from pyfame.logging.read_experiment_log import read_experiment_log
from pyfame.logging.write_experiment_log import write_experiment_log
from pyfame.layer._layer import Layer
from pyfame.layer.manipulations.mask import layer_mask


def test_read_experiment_log_invalid_path(tmp_path):
    """A nonexistent log file should raise ValueError."""
    invalid_path = tmp_path / "nonexistent.json"

    with pytest.raises(ValueError):
        read_experiment_log(str(invalid_path))


def test_read_experiment_log_directory_path(tmp_path):
    """An existing directory should raise IsADirectoryError."""
    with pytest.raises(IsADirectoryError):
        read_experiment_log(str(tmp_path))


@pytest.mark.parametrize("extension", [
    ".txt",
    ".csv",
    ".log",
    ".xml",
])
def test_read_experiment_log_invalid_extension(tmp_path, extension):
    """Existing files without a JSON extension should raise ValueError."""
    invalid_file = tmp_path / f"experiment_log{extension}"
    invalid_file.touch()

    with pytest.raises(ValueError):
        read_experiment_log(str(invalid_file))


def test_read_experiment_log_invalid_schema(tmp_path):
    invalid_log = tmp_path / "invalid_log.json"
    invalid_log.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        read_experiment_log(str(invalid_log))


def test_read_experiment_log_return(valid_log_file):
    layers = read_experiment_log(valid_log_file)

    assert isinstance(layers, list)
    assert len(layers) == 1
    assert isinstance(layers[0], Layer)


def test_write_experiment_log_invalid_path(tmp_path):
    invalid_path = tmp_path / "nonexistent"

    with pytest.raises(OSError):
        write_experiment_log(
            layers=layer_mask(),
            timestamp_id="test_invalid",
            working_directory_path=str(invalid_path)
        )


def test_write_experiment_log_output(data_paths):
    layer = layer_mask()
    timestamp_id = "test_log"

    write_experiment_log(
        layers=layer,
        timestamp_id=timestamp_id,
        working_directory_path=str(data_paths["data"])
    )

    log_path = data_paths["logs"] / f"{timestamp_id}.json"

    assert log_path.is_file()

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    parameters = log_data["layers"][type(layer).__name__]

    assert isinstance(parameters["background_colour"], list)
    assert len(parameters["background_colour"]) == 3
    assert parameters["background_colour"][0] == 0