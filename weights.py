"""Resolve model snapshots from persistent disk, app.nz mirror, or Hugging Face."""

import fnmatch
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_REPO = "Wan-AI/Wan2.2-S2V-14B"
LORA_REPO = "Quark-Vision/Live-Avatar"
DEFAULT_BASE = "https://appstatic.app.nz/models"

BASE_ALLOW_PATTERNS = (
    "config.json",
    "configuration.json",
    "diffusion_pytorch_model-*.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "google/umt5-xxl/*",
    "wav2vec2-large-xlsr-53-english/*.json",
    "wav2vec2-large-xlsr-53-english/model.safetensors",
)


def weights_dir() -> Path:
    configured = os.environ.get("WEIGHTS_DIR")
    if configured:
        return Path(configured)
    if Path("/runpod-volume").exists():
        return Path("/runpod-volume/models")
    return Path("/weights")


def _allowed(path: str, patterns: tuple[str, ...] | None) -> bool:
    return patterns is None or any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _fetch_manifest(base: str, repo: str, timeout: int = 30) -> list[dict]:
    with urllib.request.urlopen(f"{base}/{repo}/manifest.json", timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, list):
        raise ValueError("model manifest must be a list")
    return value


def _download_file(base: str, repo: str, entry: dict, destination: Path, retries: int = 3) -> None:
    relative = entry["path"]
    expected_size = int(entry["size"])
    output = destination / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            existing = output.stat().st_size if output.exists() else 0
            if existing == expected_size:
                return
            if existing > expected_size:
                output.unlink()
                existing = 0
            request = urllib.request.Request(f"{base}/{repo}/{relative}")
            mode = "wb"
            if existing:
                request.add_header("Range", f"bytes={existing}-")
                mode = "ab"
            with urllib.request.urlopen(request, timeout=120) as response:
                if existing and response.status == 200:
                    mode = "wb"
                with output.open(mode) as file:
                    while chunk := response.read(1 << 20):
                        file.write(chunk)
            if output.stat().st_size != expected_size:
                raise IOError(f"size mismatch for {relative}")
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def _download_from_mirror(
    base: str,
    repo: str,
    destination: Path,
    patterns: tuple[str, ...] | None,
    workers: int,
) -> bool:
    try:
        manifest = [entry for entry in _fetch_manifest(base, repo) if _allowed(entry["path"], patterns)]
    except Exception:
        return False
    if not manifest:
        return False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download_file, base, repo, entry, destination) for entry in manifest]
        for future in futures:
            future.result()
    return all(
        (destination / entry["path"]).is_file()
        and (destination / entry["path"]).stat().st_size == int(entry["size"])
        for entry in manifest
    )


def ensure_weights(
    repo: str,
    *,
    allow_patterns: tuple[str, ...] | None = None,
    base: str | None = None,
    workers: int = 12,
) -> Path:
    destination = weights_dir() / repo.split("/")[-1]
    marker = destination / ".incomplete"
    if destination.is_dir() and any(destination.iterdir()) and not marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    marker.touch()
    mirror = base or os.environ.get("APPNZ_MODELS_BASE", DEFAULT_BASE)
    try:
        mirrored = _download_from_mirror(mirror, repo, destination, allow_patterns, workers)
        if not mirrored:
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo,
                local_dir=str(destination),
                allow_patterns=list(allow_patterns) if allow_patterns else None,
                max_workers=workers,
            )
        marker.unlink()
        return destination
    except Exception:
        marker.touch()
        raise
