#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download small benchmark datasets (TSP, Coloring, SAT, Knapsack)
- Сучасні HTTPS-посилання + резервні дзеркала
- Авто-розпакування .gz і .tar.gz
- Ідемпотентність (не перетерти існуюче без --force)
- Фільтрування категорій (--only tsp,coloring,sat,knapsack)
"""

from __future__ import annotations
import argparse
import gzip
import io
import os
import shutil
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence
import requests


# ---------- Каталог цільових елементів ----------

@dataclass
class DatasetItem:
    path: str                         # куди зберігати (відносно data/)
    urls: Sequence[str]               # список URL (primary -> fallbacks)
    decompress: str | None = None     # None | 'gz' | 'tar.gz'
    note: str = ""                    # опціонально

    def category(self) -> str:
        return self.path.split("/", 1)[0]


# TSP із резервним дзеркалом (Rice) + оригінал (Heidelberg).
# Rice віддає .gz -> розпаковуємо у .tsp / .opt.tour
TSP_ITEMS: List[DatasetItem] = [
    DatasetItem(
        path="tsp/gr17.tsp",
        urls=[
            "https://softlib.rice.edu/pub/tsplib/tsp/gr17.tsp.gz",
            "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/gr17.tsp",
        ],
        decompress="gz",
    ),
    DatasetItem(
        path="tsp/gr17.opt.tour",
        urls=[
            "https://softlib.rice.edu/pub/tsplib/tsp/gr17.opt.tour.gz",
            "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/gr17.opt.tour",
        ],
        decompress="gz",
    ),
    DatasetItem(
        path="tsp/fri26.tsp",
        urls=[
            "https://softlib.rice.edu/pub/tsplib/tsp/fri26.tsp.gz",
            "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/fri26.tsp",
        ],
        decompress="gz",
    ),
    DatasetItem(
        path="tsp/att48.tsp",
        urls=[
            "https://softlib.rice.edu/pub/tsplib/tsp/att48.tsp.gz",
            "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/att48.tsp",
        ],
        decompress="gz",
    ),
    DatasetItem(
        path="tsp/att48.opt.tour",
        urls=[
            "https://softlib.rice.edu/pub/tsplib/tsp/att48.opt.tour.gz",
            "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/att48.opt.tour",
        ],
        decompress="gz",
    ),
]

# DIMACS Coloring (дрібні зручні інстанси)
COLORING_ITEMS: List[DatasetItem] = [
    DatasetItem(
        path="coloring/anna.col",
        urls=["https://mat.tepper.cmu.edu/COLOR/instances/anna.col"],
    ),
    DatasetItem(
        path="coloring/queen5_5.col",
        urls=["https://mat.tepper.cmu.edu/COLOR/instances/queen5_5.col"],
    ),
    DatasetItem(
        path="coloring/myciel3.col",
        urls=["https://mat.tepper.cmu.edu/COLOR/instances/myciel3.col"],
    ),
]

# SATLIB: класичний набір uf20-91 (тарбол з 1000 інстансів)
SAT_ITEMS: List[DatasetItem] = [
    DatasetItem(
        path="maxsat/uf20-91",  # буде створено як директорію і розпаковано сюди
        urls=["https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz"],
        decompress="tar.gz",
        note="Uniform Random-3-SAT (uf20-91), 1000 інстансів",
    ),
]

# OR-Library: багатовимірний/класичний knapsack (текстові файли)
KNAPSACK_ITEMS: List[DatasetItem] = [
    DatasetItem(
        path="knapsack/mknap1",
        urls=["https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknap1"],
        note="MKP: 7 задач (Petersen 1967)",
    ),
    DatasetItem(
        path="knapsack/mknap2",
        urls=["https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknap2"],
        note="MKP: 48 задач (з літератури)",
    ),
]

ALL_ITEMS: List[DatasetItem] = TSP_ITEMS + COLORING_ITEMS + SAT_ITEMS + KNAPSACK_ITEMS


# ---------- Завантаження/розпакування ----------

UA = "blended5-datasets/1.0 (+github.com/your-org/your-repo)"


def http_get(url: str, timeout: int = 60) -> bytes:
    with requests.get(url, timeout=timeout, allow_redirects=True, stream=True, headers={"User-Agent": UA}) as r:
        r.raise_for_status()
        buf = io.BytesIO()
        for chunk in r.iter_content(chunk_size=1 << 14):
            if chunk:
                buf.write(chunk)
        return buf.getvalue()


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def save_bytes(b: bytes, dest: Path, force: bool) -> None:
    ensure_parent(dest)
    tmp = dest.with_suffix(dest.suffix + ".download")
    with open(tmp, "wb") as f:
        f.write(b)
    if force and dest.exists():
        if dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    tmp.rename(dest)


def decompress_gz(src_bytes: bytes, dest: Path) -> None:
    ensure_parent(dest)
    with gzip.GzipFile(fileobj=io.BytesIO(src_bytes)) as gz, open(dest, "wb") as out:
        shutil.copyfileobj(gz, out)


def safe_extract_tar_gz(src_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    def is_within_directory(directory: str, target: str) -> bool:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

    with tarfile.open(fileobj=io.BytesIO(src_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            target_path = dest_dir / member.name
            if not is_within_directory(str(dest_dir), str(target_path)):
                raise RuntimeError("Blocked potential path traversal in tarball")
        tf.extractall(dest_dir)


def already_present(item: DatasetItem, data_dir: Path) -> bool:
    target = data_dir / item.path
    if item.decompress == "tar.gz":
        return target.exists() and target.is_dir() and any(target.iterdir())
    return target.exists()


def download_item(item: DatasetItem, data_dir: Path, force: bool, timeout: int) -> bool:
    target = data_dir / item.path
    # Визначаємо "фінальну" ціль для запису
    for url in item.urls:
        try:
            print(f"  → {item.path}  [{url}]")
            blob = http_get(url, timeout=timeout)

            if item.decompress == "gz":
                decompress_gz(blob, target)
            elif item.decompress == "tar.gz":
                # path трактується як директорія призначення
                safe_extract_tar_gz(blob, target)
            else:
                save_bytes(blob, target, force)
            print("    ✓ ok")
            return True
        except requests.exceptions.RequestException as e:
            print(f"    ✗ HTTP error: {e}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    return False


# ---------- CLI / main ----------

def iter_items(selected: set[str] | None) -> Iterable[DatasetItem]:
    if not selected:
        yield from ALL_ITEMS
        return
    for it in ALL_ITEMS:
        if it.category() in selected:
            yield it


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download small benchmark datasets to ./data")
    p.add_argument("--dir", default="data", help="Базова директорія для даних (default: data)")
    p.add_argument("--only", default="", help="Коми-розділений список категорій: tsp,coloring,sat,knapsack")
    p.add_argument("--force", action="store_true", help="Перезаписати існуючі файли/каталоги")
    p.add_argument("--timeout", type=int, default=90, help="HTTP timeout сек (default: 90)")
    p.add_argument("--list", action="store_true", help="Лише показати перелік без завантаження")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    data_dir = Path(args.dir)
    selected = {s.strip() for s in args.only.split(",") if s.strip()} or None

    items = list(iter_items(selected))

    if args.list:
        print("Available items:")
        for it in items:
            print(f"  - {it.path:30s}  [{it.category()}]  {'; '.join(it.urls[:1])}")
        return 0

    data_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading benchmark datasets")
    print("=" * 60)
    ok = 0
    skip = 0
    for it in items:
        if already_present(it, data_dir) and not args.force:
            print(f"  ↷ Skip {it.path} (exists)")
            skip += 1
            continue
        if download_item(it, data_dir, force=args.force, timeout=args.timeout):
            ok += 1

    total = len(items)
    print("=" * 60)
    print(f"OK: {ok}/{total}  |  Skipped: {skip}  |  Failed: {total - ok - skip}")
    if ok + skip == total:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
