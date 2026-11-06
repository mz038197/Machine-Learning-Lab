import json

from add_convolutional_neural_networks.core import NOTEBOOK_NAME, default_source_dir

ROAD_TEST_SITE = "https://road-test.fly.dev/"


def test_last_cell_points_to_the_road_test_site() -> None:
    notebook = json.loads((default_source_dir() / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    last = notebook["cells"][-1]
    text = "".join(last["source"])

    assert last["cell_type"] == "markdown", "最後一格是說明文字"
    assert "下一步：路考" in text, "實驗本最後一格要接到委託：路考"
    assert ROAD_TEST_SITE in text, f"最後一格要附上路考網站的網址 {ROAD_TEST_SITE}"
