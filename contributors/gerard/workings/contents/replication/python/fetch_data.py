"""Fetch the competition parquet files into data/ and check them against the pinned hashes.

The data is never committed (competition rules section 5b). Two sources are supported:
  python python/fetch_data.py                     # export from the Databricks workspace (needs `databricks auth login`)
  python python/fetch_data.py --from-dir <path>   # copy from an existing local copy
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", type=Path, help="Copy from this local folder instead of Databricks.")
    args = ap.parse_args()

    cfg = common.load_config()
    data_dir = common.PACK_ROOT / cfg["paths"]["data_dir"]
    data_dir.mkdir(exist_ok=True)
    remote = cfg["source"]["databricks_data_path"]
    if not args.from_dir and "{user}" in remote:
        me = subprocess.run(["databricks", "current-user", "me", "-o", "json"], capture_output=True, text=True, check=True)
        remote = remote.replace("{user}", json.loads(me.stdout)["userName"])
    for t in cfg["source"]["tables"]:
        dest = data_dir / f"{t}.parquet"
        if args.from_dir:
            shutil.copy2(args.from_dir / f"{t}.parquet", dest)
        else:
            src = f"{remote}/{t}.parquet"
            subprocess.run(["databricks", "workspace", "export", src, "--format", "AUTO",
                            "--file", str(dest)], check=True)
        print(f"fetched {dest.name}")
    common.verify_inputs(cfg)
    print("All input hashes match audit/expected_inputs.sha256")


if __name__ == "__main__":
    main()
