from __future__ import annotations

import argparse
import sys
from pathlib import Path

from add_logistic_regression.core import default_source_dir, materialize, run_uv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把邏輯回歸實驗夾寫進學生專案，並裝上相依。"
    )
    parser.add_argument(
        "-C",
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="學生專案目錄（預設為目前目錄）",
    )
    args = parser.parse_args(argv)
    try:
        source_dir = default_source_dir()
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    result = materialize(args.project_root, source_dir=source_dir, run_uv=run_uv)
    stream = sys.stderr if result.status == "failed" else sys.stdout
    print(result.message, file=stream)
    return 1 if result.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
