import pathlib
import posixpath
import contextual_failure
from pathlib import Path

pkg_dir_str = posixpath.dirname(contextual_failure.__file__)

def find_repo_data_dir() -> Path | None:
    here = Path(contextual_failure.__file__).resolve().parent
    for parent in (here, *here.parents):
        # repo root typically has pyproject.toml
        if (parent / "pyproject.toml").exists() and (parent / "data").exists():
            return parent / "data"
    return None

DATA_DIR = find_repo_data_dir()  # Path or None