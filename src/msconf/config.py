"""Configuration loading and provenance stamping.

Every result file written by the pipeline carries: script name, config hash,
seed, git commit (and dirty flag), and UTC timestamp, so that each reported
number can be traced back to code + config + seed + commit.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_hash"] = config_hash(path)
    return cfg


def config_hash(path: str | os.PathLike | None = None) -> str:
    path = Path(path) if path else ROOT / "config.yaml"
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def git_commit() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        # "dirty" refers to code and configuration only; regenerated outputs (results/, figures/, reports/, models/,
        # REPORT.md, WETLAB_PREREG.md, cosmic/) are expected to change when the pipeline is re-run.
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "scripts",
                                              "config.yaml", "Makefile", "requirements.txt", "reports/templates"],
                                             cwd=ROOT).decode().strip())
    except Exception:
        sha, dirty = "unknown", True
    return {"commit": sha, "dirty": dirty}


def provenance(cfg: dict, script: str | None = None, seed: int | None = None, extra: dict | None = None) -> dict:
    p = {
        "script": script or Path(sys.argv[0]).name,
        "config_hash": cfg.get("_config_hash"),
        "seed": seed if seed is not None else cfg["project"]["seed"],
        **git_commit(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        p.update(extra)
    return p


def path(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_result(cfg: dict, name: str, payload: dict, script: str | None = None, seed: int | None = None) -> Path:
    out = path(cfg, "results") / f"{name}.json"
    payload = {"_provenance": provenance(cfg, script=script, seed=seed), **payload}
    with open(out, "w") as f:
        json.dump(payload, f, indent=1, default=_json_default)
    return out


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
