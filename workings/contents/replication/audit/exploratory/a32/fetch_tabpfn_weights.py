"""Download only the approved TabPFN-3.5 weights (tabpfn-v3.5-20260909.safetensors, 876 MB) into the scratch cache.
Run it yourself: a browser opens for the Prior Labs login and licence acceptance on first use."""
import hashlib
import os
from pathlib import Path

from tabpfn.constants import ModelVersion
from tabpfn.model_loading import download_model

NAME = "tabpfn-v3.5-20260909.safetensors"
to = Path(os.environ["TABPFN_MODEL_CACHE_DIR"]) / NAME
result = download_model(to, version=ModelVersion.V3_5, which="classifier", model_name=NAME)
if result != "ok":
    raise SystemExit(f"download failed: {result}")
h = hashlib.sha256(to.read_bytes()).hexdigest()
print(f"downloaded {to.name}: {to.stat().st_size / 1e6:.1f} MB, sha256 {h[:16]}")
