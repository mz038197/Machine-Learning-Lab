import json
from pathlib import Path

from add_linear_regression.core import materialize

NOTEBOOK = "線性回歸.ipynb"
PACKAGES = ["numpy", "matplotlib", "scikit-learn", "ipykernel"]


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    (source / NOTEBOOK).write_text("notebook", encoding="utf-8")
    (source / "data" / "ex1data1.csv").write_text("1,2\n", encoding="utf-8")
    (source / "data" / "ex1data2.csv").write_text("3,4,5\n", encoding="utf-8")
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

    folder = project / "線性回歸"
    assert result.status == "written"
    assert (folder / NOTEBOOK).read_text(encoding="utf-8") == "notebook"
    assert (folder / "data" / "ex1data1.csv").read_text(encoding="utf-8") == "1,2\n"
    assert (folder / "data" / "ex1data2.csv").read_text(encoding="utf-8") == "3,4,5\n"
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
    folder = project / "線性回歸"
    (folder / "data").mkdir(parents=True)
    (project / "pyproject.toml").write_text("x", encoding="utf-8")
    (folder / NOTEBOOK).write_text("student edit", encoding="utf-8")
    calls: list[list[str]] = []

    result = materialize(project, source_dir=_source(tmp_path), run_uv=_runner(calls))

    assert result.status == "skipped"
    assert (folder / NOTEBOOK).read_text(encoding="utf-8") == "student edit"
    assert not (folder / "data" / "ex1data1.csv").exists()
    assert calls[0][1] == "add"


def test_incomplete_write_removes_the_folder(tmp_path: Path) -> None:
    project = tmp_path / "student"
    project.mkdir()
    (project / "pyproject.toml").write_text("x", encoding="utf-8")
    source = _source(tmp_path)
    (source / "data" / "ex1data2.csv").unlink()
    calls: list[list[str]] = []

    result = materialize(project, source_dir=source, run_uv=_runner(calls))

    assert result.status == "failed"
    assert not (project / "線性回歸").exists()
    assert calls == []


def test_failed_package_install_keeps_a_complete_folder(tmp_path: Path) -> None:
    project = tmp_path / "student"
    project.mkdir()
    (project / "pyproject.toml").write_text("x", encoding="utf-8")

    def run_uv(args: list[str]) -> None:
        raise RuntimeError("uv add failed")

    result = materialize(project, source_dir=_source(tmp_path), run_uv=run_uv)

    assert result.status == "failed"
    assert (project / "線性回歸" / NOTEBOOK).is_file()
    assert (project / "線性回歸" / "data" / "ex1data2.csv").is_file()


def test_uses_the_given_directory_not_a_parent_project(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    project = parent / "child"
    project.mkdir(parents=True)
    (parent / "pyproject.toml").write_text("parent", encoding="utf-8")
    calls: list[list[str]] = []

    materialize(project, source_dir=_source(tmp_path), run_uv=_runner(calls))

    assert all(str(parent) not in arg or str(project) in arg for call in calls for arg in call)
    assert calls[0][-1] == str(project)


def _load_find_data_dir():
    notebook = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "add_linear_regression"
        / "experiment"
        / NOTEBOOK
    )
    document = json.loads(notebook.read_text(encoding="utf-8"))
    for cell in document["cells"]:
        source = "".join(cell.get("source", []))
        if "def find_data_dir" in source:
            start = source.index("def find_data_dir")
            end = source.index("\nDATA_DIR")
            namespace: dict[str, object] = {}
            exec("from pathlib import Path\n\n" + source[start:end], namespace)
            return namespace
    raise AssertionError("notebook 裡沒有 find_data_dir")


def test_notebook_uses_data_beside_itself_not_a_root_decoy(tmp_path: Path, monkeypatch) -> None:
    namespace = _load_find_data_dir()
    find_data_dir = namespace["find_data_dir"]
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ex1data1.csv").write_text("old", encoding="utf-8")
    experiment = tmp_path / "線性回歸"
    (experiment / "data").mkdir(parents=True)
    (experiment / NOTEBOOK).write_text("notebook", encoding="utf-8")
    (experiment / "data" / "ex1data1.csv").write_text("student", encoding="utf-8")
    namespace["__vsc_ipynb_file__"] = str(experiment / NOTEBOOK)
    monkeypatch.chdir(tmp_path)

    assert find_data_dir() == (experiment / "data").resolve()
    assert (find_data_dir() / "ex1data1.csv").read_text(encoding="utf-8") == "student"


def test_notebook_in_tool_directory_ignores_repo_root_data(tmp_path: Path, monkeypatch) -> None:
    namespace = _load_find_data_dir()
    find_data_dir = namespace["find_data_dir"]
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ex1data1.csv").write_text("course", encoding="utf-8")
    tool = tmp_path / "linear-regression"
    (tool / "data").mkdir(parents=True)
    (tool / NOTEBOOK).write_text("notebook", encoding="utf-8")
    (tool / "data" / "ex1data1.csv").write_text("tool", encoding="utf-8")
    namespace["__vsc_ipynb_file__"] = str(tool / NOTEBOOK)
    monkeypatch.chdir(tmp_path)

    assert (find_data_dir() / "ex1data1.csv").read_text(encoding="utf-8") == "tool"


def test_editor_notebook_path_uses_that_folder(tmp_path: Path, monkeypatch) -> None:
    namespace = _load_find_data_dir()
    find_data_dir = namespace["find_data_dir"]
    decoy = tmp_path / "線性回歸"
    (decoy / "data").mkdir(parents=True)
    (decoy / NOTEBOOK).write_text("decoy", encoding="utf-8")
    (decoy / "data" / "ex1data1.csv").write_text("decoy", encoding="utf-8")
    opened = tmp_path / "opened"
    (opened / "data").mkdir(parents=True)
    (opened / NOTEBOOK).write_text("opened", encoding="utf-8")
    (opened / "data" / "ex1data1.csv").write_text("opened", encoding="utf-8")
    namespace["__vsc_ipynb_file__"] = str(opened / NOTEBOOK)
    monkeypatch.chdir(tmp_path)

    assert (find_data_dir() / "ex1data1.csv").read_text(encoding="utf-8") == "opened"
