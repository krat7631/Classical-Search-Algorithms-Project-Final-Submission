"""
Download the Moving AI Lab benchmark sets used by evaluation/benchmark.py (~6.5 MB).

Sets: Dragon Age: Origins game maps ("dao") and real city street maps ("street").
Files are cached in benchmarks/ (gitignored). Please cite:
    N. Sturtevant, "Benchmarks for Grid-Based Pathfinding,"
    IEEE Transactions on Computational Intelligence and AI in Games, 4(2), 2012.

Usage:
    python scripts/download_benchmarks.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://movingai.com/benchmarks"
SETS = ["dao", "street"]


def main() -> None:
    out = ROOT / "benchmarks"
    for name in SETS:
        for kind in ("map", "scen"):
            target = out / name / kind
            if target.exists() and any(target.iterdir()):
                print(f"{name}-{kind}: cached")
                continue
            url = f"{BASE_URL}/{name}/{name}-{kind}.zip"
            print(f"Downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
            target.mkdir(parents=True, exist_ok=True)
            for member in archive.infolist():
                filename = Path(member.filename).name
                if member.is_dir() or not filename.endswith((".map", ".scen")):
                    continue
                (target / filename).write_bytes(archive.read(member))
            print(f"{name}-{kind}: {sum(1 for _ in target.iterdir())} files")


if __name__ == "__main__":
    main()
