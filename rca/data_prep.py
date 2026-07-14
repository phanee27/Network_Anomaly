from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from rca.config import get_settings

ZENODO_RECORD = "https://zenodo.org/records/14590730/files/{name}.zip?download=1"
DATASETS = [
    "RE1-OB", "RE1-SS", "RE1-TT",
    "RE2-OB", "RE2-SS", "RE2-TT",
    "RE3-OB", "RE3-SS", "RE3-TT",
]


def _release_dir(name: str) -> str:
    return name.split("-", 1)[0]


def download(name: str, dest_root: Path) -> Path:
    release = _release_dir(name)
    dest = dest_root / release
    target = dest / name
    if target.exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{name}.zip"
    url = ZENODO_RECORD.format(name=name)

    def _hook(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100.0, 100.0 * block * block_size / total)
            print(f"\r{name}: {pct:5.1f}%", end="")

    urllib.request.urlretrieve(url, zip_path, _hook)
    print()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    zip_path.unlink()
    return target


def curate(src: Path, dst: Path, n: int) -> list[Path]:
    cases = sorted(p.parent for p in Path(src).rglob("inject_time.txt"))
    copied: list[Path] = []
    for case in cases[:n]:
        rel = case.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(case, out)
        copied.append(out)
    return copied


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Download / curate RCAEval datasets (offline data prep).")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download + extract one RCAEval dataset from Zenodo")
    dl.add_argument("--dataset", required=True, choices=DATASETS)
    dl.add_argument("--dest", default=str(settings.rcaeval_root))

    cur = sub.add_parser("curate", help="copy a few cases into an offline demo subset")
    cur.add_argument("--src", required=True, help="e.g. data/rcaeval/RE2/RE2-OB")
    cur.add_argument("--dst", required=True, help="e.g. data/curated/RE2-OB")
    cur.add_argument("--n", type=int, default=5)

    args = parser.parse_args()
    if args.command == "download":
        path = download(args.dataset, Path(args.dest))
        print(f"ready: {path}")
    elif args.command == "curate":
        copied = curate(Path(args.src), Path(args.dst), args.n)
        print(f"curated {len(copied)} cases into {args.dst}")


if __name__ == "__main__":
    main()
