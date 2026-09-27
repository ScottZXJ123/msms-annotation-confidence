"""Download public MassSpecGym files (v1.5) from Hugging Face into data/raw/. Raw files are never modified."""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msconf.config import load_config, path  # noqa: E402

cfg = load_config()
raw = path(cfg, "raw") / "massspecgym"
raw.mkdir(parents=True, exist_ok=True)
base = f"https://huggingface.co/datasets/{cfg['data']['hf_repo']}/resolve/main/"
for key, rel in cfg["data"]["files"].items():
    out = raw / Path(rel).name
    if out.exists() and out.stat().st_size > 0:
        print("exists", out)
        continue
    print("downloading", base + rel)
    urllib.request.urlretrieve(base + rel, out)
print("done")
