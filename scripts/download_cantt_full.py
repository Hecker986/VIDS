"""Download the full public Bitbucket can-train-and-test repository tree.

This intentionally targets the public original repository only:
https://bitbucket.org/brooke-lampe/can-train-and-test
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


API_ROOT = "https://api.bitbucket.org/2.0/repositories/brooke-lampe/can-train-and-test/src/master"
RAW_ROOT = "https://bitbucket.org/brooke-lampe/can-train-and-test/raw/master"


def api_json(url: str) -> dict:
    with urlopen(url, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def list_tree(path: str = "") -> list[str]:
    files: list[str] = []
    url = f"{API_ROOT}/{quote(path)}/?pagelen=100" if path else f"{API_ROOT}/?pagelen=100"
    while url:
        payload = api_json(url)
        for item in payload.get("values", []):
            item_path = item.get("path", "")
            item_type = item.get("type", "")
            if item_type == "commit_directory":
                files.extend(list_tree(item_path))
            elif item_type == "commit_file":
                files.append(item_path)
        url = payload.get("next")
    return files


def download(path: str, out_root: Path) -> Path:
    out = out_root / path
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"{RAW_ROOT}/{quote(path)}"
    subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--output", str(out), url], check=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="data/raw/can-train-and-test")
    parser.add_argument("--manifest", default="data/raw/can-train-and-test_full_manifest.txt")
    parser.add_argument("--suffixes", nargs="*", default=[".csv", ".md", ".txt", ".xlsx", ".json"])
    args = parser.parse_args()

    out_root = Path(args.out_root)
    paths = list_tree("")
    selected = [p for p in paths if any(p.lower().endswith(s.lower()) for s in args.suffixes)]
    downloaded = []
    for i, path in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {path}", flush=True)
        downloaded.append(download(path, out_root))
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(p) for p in downloaded) + "\n", encoding="utf-8")
    print(f"downloaded {len(downloaded)} files; wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
