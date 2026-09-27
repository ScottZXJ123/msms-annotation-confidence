"""ClassyFire lookup by InChIKey with a local cache (§4 'family').

Order: (1) local cache; (2) existing ClassyFire results by InChIKey via the Fiehn-lab ClassyFire
cache service (cfb.fiehnlab.ucdavis.edu) and the ClassyFire entity endpoint; (3) only for structures
still missing, a new ClassyFire classification request (public MassSpecGym structures only — lab
data are never sent anywhere). Molecules that remain unclassified go to the 'unclassified' group.
Cache: data/cache/classyfire.jsonl (committed; public structures only).
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import ROOT, load_config, path  # noqa: E402

cfg = load_config()
cache_f = path(cfg, "cache") / "classyfire.jsonl"
molfile = Path(sys.argv[1]) if len(sys.argv) > 1 else path(cfg, "derived") / "molecules.parquet"
mols = pd.read_parquet(molfile)
if "lab" in str(Path(cfg["paths"]["derived"])):
    raise SystemExit("refusing: lab data must never be sent to ClassyFire")

done = {}
if cache_f.exists():
    for line in open(cache_f):
        r = json.loads(line)
        done[r["inchikey"]] = r
todo = [ik for ik in mols.inchikey_full if ik not in done or done[ik].get("status") == "error"]
print("cached", len(done), "todo", len(todo), flush=True)
lock = threading.Lock()
ENDPOINTS = ["https://cfb.fiehnlab.ucdavis.edu/entities/{ik}.json", "http://classyfire.wishartlab.com/entities/{ik}.json"]


def name(d, k):
    v = d.get(k)
    return v.get("name") if isinstance(v, dict) else None


def fetch(ik):
    rec = {"inchikey": ik, "status": "missing"}
    for ep in ENDPOINTS:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(ep.format(ik=ik), timeout=40) as r:
                    d = json.loads(r.read())
                if isinstance(d, dict) and (d.get("kingdom") or d.get("superclass")):
                    rec = {"inchikey": ik, "status": "ok", "source": ep.split("/")[2], "returned_id": d.get("_id") or d.get("inchikey"),
                           **{k: name(d, k) for k in ["kingdom", "superclass", "class", "subclass", "direct_parent"]}}
                    return rec
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
                time.sleep(2 * (attempt + 1))
            except Exception:
                time.sleep(2 * (attempt + 1))
                rec = {"inchikey": ik, "status": "error"}
    return rec


def work(ik):
    rec = fetch(ik)
    with lock:
        with open(cache_f, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec["status"]


t0 = time.time()
n = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    for st in ex.map(work, todo):
        n += 1
        if n % 1000 == 0:
            print(n, st, f"{time.time()-t0:.0f}s", flush=True)
print("finished", n, flush=True)

# (3) new classification requests for structures still missing (public structures only)
done = {}
for line in open(cache_f):
    r = json.loads(line)
    done[r["inchikey"]] = r
missing = mols[mols.inchikey_full.map(lambda k: done.get(k, {}).get("status") != "ok")]
submit_log = {"n_missing_after_lookup": int(len(missing))}
if len(missing):
    smi_col = "std_smiles" if "std_smiles" in missing else "smiles"
    body = {"label": "msconf_public_massspecgym", "query_input": "\n".join(missing[smi_col].astype(str).tolist()[:1000]),
            "query_type": "STRUCTURE"}
    try:
        req = urllib.request.Request("http://classyfire.wishartlab.com/queries.json", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            submit_log["submission"] = json.loads(r.read())
    except Exception as e:  # the ClassyFire query service answered HTTP 500 on 2026-09-26 (see LOG)
        submit_log["submission_error"] = str(e)
print(submit_log)

