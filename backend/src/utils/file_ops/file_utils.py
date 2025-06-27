import os
import logging
import pathlib
from typing import Union, Optional


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def find_file_dir(file_path: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Returns the directory of the given file path.

    Args:
        file_path (str or Path): Full path to the file.

    Returns:
        Path: The directory containing the file.
    """
    return pathlib.Path(file_path).parent


def create_dir(dir_name: str, base_path: Optional[Union[str, pathlib.Path]] = None) -> pathlib.Path:
    """
    Creates a directory under the given base path. If base_path is None,
    the current working directory is used.

    Args:
        dir_name (str): The name of the directory to create.
        base_path (Optional[str or Path]): The base path where the directory should be created.

    Returns:
        Path: The full path to the created directory.
    """
    base = pathlib.Path(base_path) if base_path else pathlib.Path.cwd()
    full_path = base / dir_name
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

def find_file_name(file_path: Union[str, pathlib.Path]) -> str:
    """Return the file name without its extension."""
    return pathlib.Path(file_path).stem

def find_file_extension(file_path: Union[str, pathlib.Path]) -> str:
    """Return the file extension, including the dot."""
    return pathlib.Path(file_path).suffix


def file_save(content, filename: str, dir_path: pathlib.Path | str) -> pathlib.Path:
    """
    Save bytes content to a file inside dir_path with given filename.

    Args:
        content : The binary content to save.
        filename (str): The name of the file to save.
        dir_path (Path or str): Directory where file will be saved.

    Returns:
        Path: Full path to the saved file.
    """
    dir_path = pathlib.Path(dir_path).resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / filename

    with file_path.open("wb") as f:
        f.write(content)

    return file_path