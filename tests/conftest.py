from pathlib import Path
import pandas as pd
import pytest
import shutil

ASSETS_DIR = Path(__file__).parent / "assets"

@pytest.fixture
def data_paths(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    """Create a temporary PyFAME data directory structure."""
    monkeypatch.chdir(tmp_path)

    data = tmp_path / "data"

    paths = {
        "data": data,
        "raw": data / "raw",
        "processed": data / "processed",
        "logs": data / "logs",
        "analysis": data / "analysis",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths

@pytest.fixture
def invalid_ext() -> Path:
    """Return the path to a test image with an unsupported extension."""
    return ASSETS_DIR / "invalid_ext.webp"


@pytest.fixture
def no_face_photo() -> Path:
    """Return the path to a valid image containing no detectable face."""
    return ASSETS_DIR / "no_face.png"

@pytest.fixture
def no_face_video() -> Path:
    """Return the path to a valid video containing no detectable face."""
    return ASSETS_DIR / "no_face.mp4"

@pytest.fixture
def valid_face_photo() -> Path:
    """Return the path to a valid photo containing a detectable face."""
    return ASSETS_DIR / "valid_face.jpg"

@pytest.fixture
def valid_face_video() -> Path:
    """Return the path to a valid photo containing a detectable face."""
    return ASSETS_DIR / "valid_face.mp4"

@pytest.fixture
def file_paths_valid_file(valid_face_video, data_paths) -> pd.DataFrame:
    """Creates valid dir structure and returns a valid PyFAME file_paths object."""
    destination = data_paths["raw"] / valid_face_video.name
    shutil.copy(valid_face_video, destination)

    return pd.DataFrame(
        {
            "Absolute Path" : [
                destination
            ],
            "Relative Path" : [
                destination.relative_to(data_paths["data"])
            ]
        }
    )

@pytest.fixture
def landmark_path():
    """A valid single landmark path."""
    return [
        (0, 1),
        (1, 2),
        (2, 0),
    ]


@pytest.fixture
def landmark_paths():
    """A valid collection of multiple landmark paths."""
    return [
        [(0, 1), (1, 2), (2, 0)],
        [(3, 4), (4, 5), (5, 3)],
    ]