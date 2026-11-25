import re
from pathlib import Path

from add_linear_regression.core import DATA_FILES, ROOT_FILES, default_source_dir

BACKUP = Path(__file__).resolve().parents[1] / "備案" / "尖峰負載"
DATA = "taipower_peak.csv"
WORKBOOK = "尖峰負載.ipynb"


def test_backup_brief_embeds_the_same_data_as_the_csv() -> None:
    html = (BACKUP / "委託.html").read_text(encoding="utf-8")
    match = re.search(r'<script id="csv" type="text/plain">\n(.*?)\n</script>', html, re.S)
    assert match, "備案的委託.html 裡找不到嵌入的資料"
    csv = (BACKUP / "data" / DATA).read_text(encoding="utf-8").strip()
    assert match.group(1) == csv, f"備案的委託.html 嵌入的資料跟 data/{DATA} 不一致，請重跑 uv run python tools/build_briefs.py"


def test_backup_folder_has_brief_workbook_and_data() -> None:
    assert (BACKUP / "委託.html").is_file()
    assert (BACKUP / WORKBOOK).is_file()
    assert (BACKUP / "data" / DATA).is_file()


def test_backup_does_not_ship_with_the_experiment() -> None:
    assert WORKBOOK not in ROOT_FILES
    assert DATA not in DATA_FILES
    source = default_source_dir()
    assert not (source / WORKBOOK).exists()
    assert not (source / "data" / DATA).exists()
