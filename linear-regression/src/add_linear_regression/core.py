from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

FOLDER_NAME = "線性回歸"
NOTEBOOK_NAME = "線性回歸_濃縮草案.ipynb"
DATA_FILES = ("ex1data1.txt", "ex1data2.txt")
PACKAGES = ("numpy", "matplotlib", "scikit-learn", "ipykernel")


@dataclass(frozen=True)
class MaterializeResult:
    status: str
    message: str


def default_source_dir() -> Path:
    packaged = Path(__file__).resolve().parent / "experiment"
    if (packaged / NOTEBOOK_NAME).is_file():
        return packaged
    raise FileNotFoundError("找不到實驗本。")


def run_uv(args: list[str]) -> None:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or f"指令結束碼 {completed.returncode}")


def materialize(
    project_root: Path,
    *,
    source_dir: Path,
    run_uv: Callable[[list[str]], None],
) -> MaterializeResult:
    root = project_root.resolve()
    destination = root / FOLDER_NAME
    wrote = False
    if not destination.exists():
        try:
            _place_experiment(root, source_dir)
        except OSError as exc:
            return MaterializeResult("failed", f"沒有留下線性回歸實驗夾：{exc}")
        wrote = True
    try:
        _install_dependencies(root, run_uv)
    except RuntimeError as exc:
        if wrote:
            return MaterializeResult("failed", f"實驗夾已寫入，但相依沒裝上：{exc}")
        return MaterializeResult("failed", f"未覆寫實驗夾。相依沒裝上：{exc}")
    if wrote:
        return MaterializeResult("written", "已寫入線性回歸實驗夾，並裝上相依。")
    return MaterializeResult("skipped", "已有線性回歸實驗夾，未覆寫。已補上相依。")


def _install_dependencies(root: Path, run_uv: Callable[[list[str]], None]) -> None:
    directory = str(root)
    if not (root / "pyproject.toml").is_file():
        run_uv(
            [
                "uv",
                "init",
                "--bare",
                "--vcs",
                "none",
                "--no-workspace",
                "--directory",
                directory,
            ]
        )
    run_uv(["uv", "add", "--directory", directory, *PACKAGES])


def _place_experiment(root: Path, source_dir: Path) -> None:
    destination = root / FOLDER_NAME
    staging = root / f".{FOLDER_NAME}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        notebook = source_dir / NOTEBOOK_NAME
        if not notebook.is_file():
            raise FileNotFoundError(notebook)
        shutil.copy2(notebook, staging / NOTEBOOK_NAME)
        data_dir = staging / "data"
        data_dir.mkdir()
        for name in DATA_FILES:
            source = source_dir / "data" / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, data_dir / name)
        if not _is_complete(staging):
            raise FileNotFoundError("實驗夾不完整")
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if destination.exists() and not _is_complete(destination):
            shutil.rmtree(destination, ignore_errors=True)
        raise


def _is_complete(folder: Path) -> bool:
    return (folder / NOTEBOOK_NAME).is_file() and all(
        (folder / "data" / name).is_file() for name in DATA_FILES
    )
