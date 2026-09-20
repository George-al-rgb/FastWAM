from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


FILES = {
    "libero_spatial_no_noops_lerobot.tar.gz": 968_794_594,
    "libero_object_no_noops_lerobot.tar.gz": 1_352_739_510,
    "libero_goal_no_noops_lerobot.tar.gz": 837_172_899,
    "libero_10_no_noops_lerobot.tar.gz": 1_533_534_552,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo", default="yuanty/LIBERO-fastwam")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--file", action="append", dest="files")
    return parser.parse_args()


def url_for(repo: str, name: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    return f"{endpoint}/datasets/{repo}/resolve/main/{name}"


def download_part(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    retries: int = 6,
) -> None:
    expected = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected:
        return

    part_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        temporary = part_path.with_suffix(part_path.suffix + ".tmp")
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(45, 240),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(
                        f"unexpected Content-Range for {start}-{end}: {content_range!r}"
                    )
                with temporary.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            if temporary.stat().st_size != expected:
                raise RuntimeError(
                    f"part has {temporary.stat().st_size} bytes, expected {expected}"
                )
            temporary.replace(part_path)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 20))


def download_file(repo: str, output_dir: Path, name: str, size: int, workers: int, chunk_size: int) -> None:
    target = output_dir / name
    if target.exists() and target.stat().st_size == size:
        print(f"[done] {name}", flush=True)
        return

    part_dir = output_dir / ".parts" / name
    ranges = [
        (start, min(start + chunk_size, size) - 1)
        for start in range(0, size, chunk_size)
    ]
    print(f"[start] {name}: {size / 1024**3:.2f} GiB, {len(ranges)} parts", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {
            pool.submit(
                download_part,
                url_for(repo, name),
                part_dir / f"{index:06d}.part",
                start,
                end,
            ): index
            for index, (start, end) in enumerate(ranges)
        }
        completed = 0
        for job in as_completed(jobs):
            job.result()
            completed += 1
            print(f"[part] {name}: {completed}/{len(ranges)}", flush=True)

    temporary = target.with_suffix(target.suffix + ".partial")
    with temporary.open("wb") as output:
        for index in range(len(ranges)):
            part = part_dir / f"{index:06d}.part"
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
    if temporary.stat().st_size != size:
        raise RuntimeError(f"assembled {temporary} has the wrong size")
    temporary.replace(target)
    print(f"[done] {name}", flush=True)


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        raise SystemExit("workers and chunk-size must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = args.files or list(FILES)
    unknown = [name for name in names if name not in FILES]
    if unknown:
        raise SystemExit(f"unknown archive: {unknown}")
    for name in names:
        download_file(
            args.repo,
            output_dir,
            name,
            FILES[name],
            args.workers,
            args.chunk_size,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
