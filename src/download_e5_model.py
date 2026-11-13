"""Download public E5 files only; no model inference or training."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HOME"] = str(ROOT / "models" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(ROOT / "models" / "huggingface" / "hub")

from huggingface_hub import snapshot_download

TARGET = ROOT / "models" / "multilingual-e5-small"
REPO = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
FILES = ["onnx/model.onnx", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "README.md"]


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    provenance = TARGET / "model_provenance.json"
    if provenance.exists():
        revision = json.loads(provenance.read_text(encoding="utf-8"))["revision"]
        assert revision == REVISION
    else:
        revision = REVISION
        provenance.write_text(json.dumps({"repo": REPO, "revision": revision,
                              "license": "MIT, declared in upstream model card",
                              "download_started_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    print(f"Downloading {REPO} revision {revision}; FP32 ONNX only", flush=True)
    snapshot_download(REPO, revision=revision, allow_patterns=FILES, local_dir=TARGET,
                      cache_dir=ROOT / "models" / "huggingface" / "hub", token=False, max_workers=3)
    record = json.loads(provenance.read_text(encoding="utf-8"))
    record["download_completed_utc"] = datetime.now(timezone.utc).isoformat()
    record["files"] = {}
    for name in FILES:
        path = TARGET / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        record["files"][name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved model and provenance to {TARGET}", flush=True)


if __name__ == "__main__":
    main()
