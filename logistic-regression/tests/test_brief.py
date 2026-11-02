import re

from add_logistic_regression.core import BRIEF_NAME, default_source_dir

DATA = "mlb_games.csv"


def test_brief_embeds_the_same_data_as_the_csv() -> None:
    source = default_source_dir()
    html = (source / BRIEF_NAME).read_text(encoding="utf-8")
    match = re.search(r'<script id="csv" type="text/plain">\n(.*?)\n</script>', html, re.S)
    assert match, "委託.html 裡找不到嵌入的資料"
    csv = (source / "data" / DATA).read_text(encoding="utf-8").strip()
    assert match.group(1) == csv, f"委託.html 嵌入的資料跟 data/{DATA} 不一致，請重新產生"
