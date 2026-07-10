"""Download IU X-Ray via Kaggle CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def download_iu_xray(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pip", "install", "-q", "kaggle",
    ]
    subprocess.run(cmd, check=True)

    kaggle_cmd = [
        "kaggle", "datasets", "download",
        "-d", "raddar/chest-xrays-indiana-university",
        "-p", str(output_dir),
        "--unzip",
    ]
    subprocess.run(kaggle_cmd, check=True)
    return output_dir / "chest-xrays-indiana-university"
