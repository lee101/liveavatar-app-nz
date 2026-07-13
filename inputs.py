import json
import subprocess
from pathlib import Path

MAX_AUDIO_SECONDS = 300.0
MAX_IMAGE_PIXELS = 40_000_000
RESOLUTION_AREAS = {
    "preview": "384*256",
    "standard": "704*384",
    "high": "1024*704",
}


def validate_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    if len(prompt) > 2_000:
        raise ValueError("prompt must be at most 2000 characters")
    return prompt


def validate_image(path: str | Path) -> None:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    if width < 64 or height < 64:
        raise ValueError("image dimensions must be at least 64x64")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("image exceeds the 40 megapixel limit")


def media_duration(path: str | Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("could not determine audio duration") from error
    if duration <= 0:
        raise ValueError("audio must have a positive duration")
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(f"audio must be at most {int(MAX_AUDIO_SECONDS)} seconds")
    return duration


def generation_options(
    prompt: str,
    image: str | Path,
    audio: str | Path,
    quality: str,
    num_clips: int,
    steps: int,
    seed: int,
    start_from_reference: bool,
    offload_model: bool,
) -> dict:
    if quality not in RESOLUTION_AREAS:
        raise ValueError(f"unknown quality: {quality}")
    return {
        "input_prompt": validate_prompt(prompt),
        "ref_image_path": str(image),
        "audio_path": str(audio),
        "num_repeat": num_clips,
        "generate_size": RESOLUTION_AREAS[quality],
        "max_area": _area(RESOLUTION_AREAS[quality]),
        "infer_frames": 48,
        "shift": 3.0,
        "sample_solver": "euler",
        "sampling_steps": steps,
        "guide_scale": 0.0,
        "seed": seed,
        "offload_model": offload_model,
        "init_first_frame": start_from_reference,
        "num_gpus_dit": 1,
        "enable_vae_parallel": False,
        "enable_online_decode": False,
    }


def _area(value: str) -> int:
    width, height = value.split("*")
    return int(width) * int(height)
