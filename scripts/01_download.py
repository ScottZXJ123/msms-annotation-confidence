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
# external source for the data-source shift test: MassBank-data 2026.03 (Zenodo record 19590880)
mb = path(cfg, "raw") / "massbank"
mb.mkdir(parents=True, exist_ok=True)
zf = mb / "MassBank-data-2026.03.zip"
if not zf.exists():
    print("downloading MassBank-data 2026.03 from Zenodo")
    urllib.request.urlretrieve("https://zenodo.org/api/records/19590880/files/MassBank/MassBank-data-2026.03.zip/content", zf)
print("done")
