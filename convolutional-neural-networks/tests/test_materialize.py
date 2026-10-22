from pathlib import Path

from add_convolutional_neural_networks.core import default_source_dir, materialize

NOTEBOOK = "卷積神經網路.ipynb"
PACKAGES = ["numpy", "matplotlib", "tensorflow", "ipykernel"]


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    (source / NOTEBOOK).write_text("notebook", encoding="utf-8")
    (source / "data" / "mnist_small.npz").write_bytes(b"npz")
    return source


def _runner(calls: list[list[str]]):
    def run(args: list[str]) -> None:
        calls.append(args)

    return run


def test_writes_experiment_folder_and_adds_packages(tmp_path: Path) -> None:
    project = tmp_path / "student"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='student'\n", encoding="utf-8")
    calls: list[list[str]] = []

    result = materialize(project, source_dir=_source(tmp_path), run_uv=_runner(calls))

    folder = project / "卷積神經網路"
    assert result.status == "written"
    assert (folder / NOTEBOOK).read_text(encoding="utf-8") == "notebook"
    assert (folder / "data" / "mnist_small.npz").read_bytes() == b"npz"
    assert calls == [
        ["uv", "add", "--directory", str(project), *PACKAGES],
    ]


def test_inits_uv_project_when_pyproject_is_missing(tmp_path: Path) -> None:
    project = tmp_path / "student"
    project.mkdir()
    calls: list[list[str]] = []

    materialize(project, source_dir=_source(tmp_path), run_uv=_runner(calls))

    assert calls[0] == [
        "uv",
        "init",
        "--bare",
        "--vcs",
        "none",
        "--no-workspace",
        "--directory",
        str(project),
    ]
    assert calls[1][0:3] == ["uv", "add", "--directory"]
    assert calls[1][3] == str(project)


def test_existing_folder_is_left_alone_while_packages_are_added(tmp_path: Path) -> None:
    project = tmp_path / "student"
    folder = project / "卷積神經網路"
    folder.mkdir(parents=True)
    (project / "pyproject.toml").write_text("x", encoding="utf-8")
    (folder / NOTEBOOK).write_text("student edit", encoding="utf-8")
    calls: list[list[str]] = []

    result = materialize(project, source_dir=_source(tmp_path), run_uv=_runner(calls))

    assert result.status == "skipped"
    assert (folder / NOTEBOOK).read_text(encoding="utf-8") == "student edit"
    assert not (folder / "data").exists()
    assert calls[0][1] == "add"


def test_incomplete_write_removes_the_folder(tmp_path: Path) -> None:
    project = tmp_path / "student"
    project.mkdir()
    (project / "pyproject.toml").write_text("x", encoding="utf-8")
    source = _source(tmp_path)
    (source / "data" / "mnist_small.npz").unlink()
    calls: list[list[str]] = []

    result = materialize(project, source_dir=source, run_uv=_runner(calls))

    assert result.status == "failed"
    assert not (project / "卷積神經網路").exists()
    assert calls == []


def test_failed_package_install_keeps_a_complete_folder(tmp_path: Path) -> None:
    project = tmp_path / "student"
    project.mkdir()
    (project / "pyproject.toml").write_text("x", encoding="utf-8")

    def run_uv(args: list[str]) -> None:
        raise RuntimeError("uv add failed")

    result = materialize(project, source_dir=_source(tmp_path), run_uv=run_uv)

    assert result.status == "failed"
    assert (project / "卷積神經網路" / NOTEBOOK).is_file()
    assert (project / "卷積神經網路" / "data" / "mnist_small.npz").is_file()


def test_uses_the_given_directory_not_a_parent_project(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    project = parent / "child"
    project.mkdir(parents=True)
    (parent / "pyproject.toml").write_text("parent", encoding="utf-8")
    calls: list[list[str]] = []

    materialize(project, source_dir=_source(tmp_path), run_uv=_runner(calls))

    assert all(str(parent) not in arg or str(project) in arg for call in calls for arg in call)
    assert calls[0][-1] == str(project)


def test_does_not_touch_other_experiment_folders(tmp_path: Path) -> None:
    project = tmp_path / "student"
    network = project / "神經網路"
    network.mkdir(parents=True)
    (network / "神經網路.ipynb").write_text("keep-network", encoding="utf-8")
    (project / "pyproject.toml").write_text("x", encoding="utf-8")

    result = materialize(project, source_dir=_source(tmp_path), run_uv=_runner([]))

    assert result.status == "written"
    assert (network / "神經網路.ipynb").read_text(encoding="utf-8") == "keep-network"
    assert (project / "卷積神經網路" / NOTEBOOK).is_file()


def test_packaged_source_has_the_notebook_and_data() -> None:
    source = default_source_dir()
    assert (source / NOTEBOOK).is_file()
    assert (source / "data" / "mnist_small.npz").is_file()
