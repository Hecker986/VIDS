"""Download selected CSV files from can-train-and-test Bitbucket repo."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen


API_ROOT = "https://api.bitbucket.org/2.0/repositories/brooke-lampe/can-train-and-test/src/master"
RAW_ROOT = "https://bitbucket.org/brooke-lampe/can-train-and-test/raw/master"


def api_json(url: str) -> dict:
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def list_csv_files(folder: str) -> list[str]:
    files = []
    url = f"{API_ROOT}/{folder}/?pagelen=100"
    while url:
        payload = api_json(url)
        for item in payload.get("values", []):
            if item.get("type") == "commit_file" and item.get("path", "").endswith(".csv"):
                files.append(item["path"])
        url = payload.get("next")
    return files


def download_file(path: str, out_root: Path) -> Path:
    out = out_root / path
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"{RAW_ROOT}/{path}"
    subprocess.run(["curl", "-L", "--fail", "--output", str(out), url], check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="data/raw/can-train-and-test")
    ap.add_argument("--folders", nargs="+", default=[
        "set_01/train_01",
        "set_01/test_01_known_vehicle_known_attack",
        "set_01/test_02_unknown_vehicle_known_attack",
        "set_01/test_03_known_vehicle_unknown_attack",
        "set_01/test_04_unknown_vehicle_unknown_attack",
    ])
    ap.add_argument("--manifest", default="data/raw/can-train-and-test_sources.txt")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    downloaded = []
    for folder in args.folders:
        files = list_csv_files(folder)
        print(f"{folder}: {len(files)} csv files", flush=True)
        for file_path in files:
            downloaded.append(download_file(file_path, out_root))
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(p) for p in downloaded) + "\n")
    print(f"wrote {manifest} with {len(downloaded)} files", flush=True)


if __name__ == "__main__":
    main()
