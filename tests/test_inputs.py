import json
import subprocess

import pytest
from PIL import Image

from inputs import generation_options, media_duration, validate_image, validate_prompt


def test_generation_options_are_bounded_and_dmd_configured(tmp_path):
    options = generation_options(
        prompt="  a painted fox speaking  ",
        image=tmp_path / "face.png",
        audio=tmp_path / "voice.wav",
        quality="standard",
        num_clips=2,
        steps=4,
        seed=12,
        start_from_reference=True,
        offload_model=True,
    )
    assert options["input_prompt"] == "a painted fox speaking"
    assert options["max_area"] == 704 * 384
    assert options["num_repeat"] == 2
    assert options["infer_frames"] == 48
    assert options["sample_solver"] == "euler"
    assert options["guide_scale"] == 0.0


def test_prompt_rejects_empty_and_oversized_values():
    with pytest.raises(ValueError, match="empty"):
        validate_prompt("  ")
    with pytest.raises(ValueError, match="2000"):
        validate_prompt("x" * 2001)


def test_image_validation(tmp_path):
    valid = tmp_path / "valid.png"
    Image.new("RGB", (128, 96)).save(valid)
    validate_image(valid)
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (32, 32)).save(tiny)
    with pytest.raises(ValueError, match="64x64"):
        validate_image(tiny)


def test_audio_duration_validation(monkeypatch):
    def completed(duration):
        return subprocess.CompletedProcess([], 0, json.dumps({"format": {"duration": duration}}), "")

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed("12.5"))
    assert media_duration("voice.wav") == 12.5
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed("301"))
    with pytest.raises(ValueError, match="300"):
        media_duration("voice.wav")
