import pytest

from pyfame.layer.manipulations.temporal import apply_temporal_shuffle
from pyfame.utils.constants import TEMPORAL_SHUFFLE_RANDOM
from pyfame.utils.exceptions import FileReadError, UnrecognizedExtensionError


@pytest.mark.parametrize(
    "shuffle_method",
    ["random", 1.5, None, [], {}],
)
def test_apply_temporal_shuffle_invalid_method_type(
    file_paths_valid_file, shuffle_method
):
    with pytest.raises(TypeError):
        apply_temporal_shuffle(
            file_paths_valid_file,
            shuffle_method=shuffle_method,
        )


@pytest.mark.parametrize(
    "rand_seed",
    ["invalid", 1.5, [], {}],
)
def test_apply_temporal_shuffle_invalid_seed_type(
    file_paths_valid_file, rand_seed
):
    with pytest.raises(TypeError):
        apply_temporal_shuffle(
            file_paths_valid_file,
            rand_seed=rand_seed,
        )


@pytest.mark.parametrize(
    "block_size",
    ["invalid", 2.5, None, [], {}],
)
def test_apply_temporal_shuffle_invalid_block_size_type(
    file_paths_valid_file, block_size
):
    with pytest.raises(TypeError):
        apply_temporal_shuffle(
            file_paths_valid_file,
            block_size=block_size,
        )


@pytest.mark.parametrize(
    "drop_last_block",
    ["invalid", 0, 1, None, [], {}],
)
def test_apply_temporal_shuffle_invalid_drop_last_block_type(
    file_paths_valid_file, drop_last_block
):
    with pytest.raises(TypeError):
        apply_temporal_shuffle(
            file_paths_valid_file,
            drop_last_block=drop_last_block,
        )


@pytest.mark.parametrize(
    "shuffle_method",
    [-1, 0, 33, 41, 100],
)
def test_apply_temporal_shuffle_invalid_method_value(
    file_paths_valid_file, shuffle_method
):
    with pytest.raises(ValueError):
        apply_temporal_shuffle(
            file_paths_valid_file,
            shuffle_method=shuffle_method,
        )


@pytest.mark.parametrize(
    "block_size",
    [-100, -1, 0, 1],
)
def test_apply_temporal_shuffle_invalid_block_size_value(
    file_paths_valid_file, block_size
):
    with pytest.raises(ValueError):
        apply_temporal_shuffle(
            file_paths_valid_file,
            block_size=block_size,
        )

def test_apply_temporal_shuffle_passthrough(
    file_paths_valid_file, tmp_path
):
    apply_temporal_shuffle(
        file_paths=file_paths_valid_file,
        shuffle_method=TEMPORAL_SHUFFLE_RANDOM,
        rand_seed=42,
        block_size=10,
        drop_last_block=False,
    )

    output_dir = tmp_path / "data" / "processed"
    output_files = list(output_dir.rglob("*.mp4"))

    assert len(output_files) >= 1
    assert all(file.is_file() for file in output_files)
    assert all(file.stat().st_size > 0 for file in output_files)

# -------------------------
# File path validation tests
# -------------------------

@pytest.mark.parametrize(
    "file_paths",
    [
        None,
        "invalid",
        123,
        [],
        {},
    ],
)
def test_apply_temporal_shuffle_invalid_file_paths_type(file_paths):
    with pytest.raises(TypeError):
        apply_temporal_shuffle(file_paths)


def test_apply_temporal_shuffle_empty_dataframe(file_paths_valid_file):
    file_paths = file_paths_valid_file.iloc[0:0].copy()

    with pytest.raises(FileReadError):
        apply_temporal_shuffle(file_paths)


@pytest.mark.parametrize(
    "column",
    ["Absolute Path", "Relative Path"],
)
def test_apply_temporal_shuffle_missing_dataframe_column(
    file_paths_valid_file, column
):
    file_paths = file_paths_valid_file.drop(columns=[column])

    with pytest.raises((ValueError, KeyError)):
        apply_temporal_shuffle(file_paths)


def test_apply_temporal_shuffle_nonexistent_file(file_paths_valid_file, tmp_path):
    file_paths = file_paths_valid_file.copy(deep=True)

    file_paths.loc[0, "Absolute Path"] = str(
        tmp_path / "data" / "raw" / "nonexistent.mp4"
    )

    with pytest.raises(FileReadError):
        apply_temporal_shuffle(file_paths)


def test_apply_temporal_shuffle_invalid_extension(file_paths_valid_file, tmp_path):
    file_paths = file_paths_valid_file.copy(deep=True)

    invalid_file = tmp_path / "data" / "raw" / "invalid.webp"
    invalid_file.touch()

    file_paths.loc[0, "Absolute Path"] = str(invalid_file)
    file_paths.loc[0, "Relative Path"] = "raw/invalid.webp"

    with pytest.raises(UnrecognizedExtensionError):
        apply_temporal_shuffle(file_paths)