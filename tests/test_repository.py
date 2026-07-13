from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_cog_contract_is_pinned_and_runtime_weighted():
    config = yaml.safe_load((ROOT / "cog.yaml").read_text())
    assert config["run"] == "predict.py:Predictor"
    assert config["build"]["gpu"] is True
    assert config["build"]["cuda"] == "12.8"
    commands = "\n".join(config["build"]["run"])
    assert "76489bc1c6718edbf610009bcd5e5436b0dc8459" in commands
    assert "huggingface-cli download" not in commands
    assert config["image"] == "ghcr.io/lee101/liveavatar-app-nz:latest"


def test_repo_contains_no_large_artifacts():
    ignored = {".git", ".venv", ".pytest_cache", "__pycache__"}
    files = [
        path for path in ROOT.rglob("*")
        if path.is_file() and not ignored.intersection(path.relative_to(ROOT).parts)
    ]
    largest = max(path.stat().st_size for path in files)
    assert largest < 2_000_000
