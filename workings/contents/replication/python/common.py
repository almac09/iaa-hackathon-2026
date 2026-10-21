"""Shared plumbing for the Python runner: config, hashing, the SQL layer and the run log.

The R runner (R/run_all.R) mirrors these functions so both produce comparable evidence.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import uuid
from pathlib import Path

import duckdb

PACK_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = PACK_ROOT / "config.json") -> dict:
    cfg = json.loads(path.read_text())
    if not cfg["decisions"]["D06_missing_as_level"]["apply"]:
        raise ValueError("D06 apply=false is not implemented: sql/08 always treats missing as a level.")
    for key in ("D04_exclude_contacts", "D05_exclude_written_premium"):
        d = cfg["decisions"][key]
        if d["apply"] and d["excluded_feature"] in cfg["model"]["factors"]:
            raise ValueError(f"{key} is applied but {d['excluded_feature']} is in model.factors.")
    return cfg


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_expected_hashes(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def verify_inputs(cfg: dict) -> dict[str, str]:
    """Hash every input parquet file and stop if any differs from the pinned expected hash."""
    data_dir = PACK_ROOT / cfg["paths"]["data_dir"]
    expected = read_expected_hashes(PACK_ROOT / cfg["paths"]["expected_inputs"])
    actual = {}
    for t in cfg["source"]["tables"]:
        name = f"{t}.parquet"
        actual[name] = sha256_file(data_dir / name)
        if expected.get(name) != actual[name]:
            raise RuntimeError(f"Input hash mismatch for {name}: expected {expected.get(name)}, got {actual[name]}")
    return actual


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(PACK_ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sql_literal(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str) and len(value) == 10 and value[4] == "-" and value[7] == "-":
        return f"DATE '{value}'"
    return "'" + str(value).replace("'", "''") + "'"


def open_duckdb(cfg: dict) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads = {int(cfg['duckdb_threads'])}")
    data_dir = PACK_ROOT / cfg["paths"]["data_dir"]
    for t in cfg["source"]["tables"]:
        path = (data_dir / f"{t}.parquet").as_posix()
        con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet('{path}')")
    set_sql_variables(con, cfg)
    return con


def set_sql_variables(con: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    """Parameters and decision switches from config.json, readable in SQL via getvariable()."""
    variables = dict(cfg["sql_variables"])
    for d in cfg["decisions"].values():
        if "sql_variable" in d:
            variables[d["sql_variable"]] = bool(d["apply"])
    for name, value in variables.items():
        con.execute(f"SET VARIABLE {name} = {sql_literal(value)}")


def run_sql_files(con: duckdb.DuckDBPyConnection, cfg: dict) -> dict[str, str]:
    hashes = {}
    for rel in cfg["sql_files"]:
        path = PACK_ROOT / rel
        hashes[rel] = sha256_file(path)
        con.execute(path.read_text())
    return hashes


def export_sql_evidence(con: duckdb.DuckDBPyConnection, out_dir: Path, prefixes: tuple[str, ...] = ("ev",)) -> list[Path]:
    """Write every ev_* / evr_* table (or those starting with `prefixes`) to CSV, rows sorted on all columns so
    files are deterministic."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [r[0] for r in con.execute(
        "SELECT table_name FROM duckdb_tables() WHERE table_name LIKE 'ev%' ORDER BY 1").fetchall()
        if r[0].startswith(prefixes)]
    paths = []
    for n in names:
        p = out_dir / f"{n}.csv"
        con.execute(f"COPY (SELECT * FROM {n} ORDER BY ALL) TO '{p.as_posix()}' (HEADER, DELIMITER ',')")
        paths.append(p)
    return paths


def append_run_log(cfg: dict, record: dict) -> None:
    path = PACK_ROOT / cfg["paths"]["run_log"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def new_run_record(runner: str, cfg: dict) -> dict:
    # Deliberately no host name, user name or absolute paths: the log is meant to be shareable.
    return {
        "run_id": str(uuid.uuid4()),
        "runner": runner,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.system(),
        "git_commit": git_commit(),
        "config_sha256": sha256_file(PACK_ROOT / "config.json"),
        "code_sha256": {p.relative_to(PACK_ROOT).as_posix(): sha256_file(p)
                        for d in ("python", "reconcile") for p in sorted((PACK_ROOT / d).glob("*.py"))},
        "decisions": {k: {"apply": v["apply"], "signed_off": v["signed_off"], "signed_off_by": v["signed_off_by"]}
                      for k, v in cfg["decisions"].items()},
    }
