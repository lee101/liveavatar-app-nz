# LiveAvatar for Cog and app.nz

Audio-driven, autoregressive character animation from a reference image, driving audio, and scene prompt. This adapter packages [Alibaba-Quark/LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar) as a warm, reusable [Cog](https://github.com/replicate/cog) predictor for scale-to-zero, pay-per-inference serving.

[![Deploy on app.nz](https://app.nz/deploy-badge.svg)](https://app.nz/deploy?template=liveavatar)

## Why this model

LiveAvatar is a 4-step, block-autoregressive Wan2.2-S2V model designed for long-lived character animation. It handles photoreal people, stylized characters, singing, expressive body motion, and optional scene/camera guidance. The upstream code and both model components are Apache-2.0.

This Cog intentionally serves bounded offline MP4 jobs. LiveAvatar's multi-GPU streaming mode belongs in an always-on service with a streaming transport, not a scale-to-zero request/response Cog.

## Hardware

- Recommended: H100 or H200 80GB.
- `LIVEAVATAR_FP8=1` is the default and follows upstream's FP8 path.
- The upstream single-GPU path requires 80GB without FP8; upstream reports FP8 operation on 48GB hardware, but this adapter does not claim L40S support until an end-to-end test passes there.
- Persistent disk: 64GB minimum; 80GB recommended.

## Predict

```sh
cog predict \
  -i image=@character.png \
  -i audio=@voice.wav \
  -i prompt="A hand-painted fox explains a map, warm tavern light" \
  -i quality=standard \
  -i num_clips=2 \
  -i steps=4
```

Inputs:

- `image`: portrait, full-body person, animal, anime, or other character image.
- `audio`: driving speech or singing, up to 300 seconds.
- `prompt`: character action, expression, scene, lighting, and camera guidance.
- `quality`: `preview` (384×256 area), `standard` (704×384 area), or `high` (1024×704 area). Input aspect ratio is preserved.
- `num_clips`: 1–8 autoregressive clips, additionally capped by audio duration.
- `steps`: 4–8; the distilled LoRA is tuned for 4.
- `seed`, `start_from_reference`.

Output: MP4 with the original driving audio muxed as AAC.

## Weights and disk use

No checkpoints are stored in Git or baked into the container. At worker setup, `weights.py` resolves:

1. an existing persistent directory (`WEIGHTS_DIR`, then `/runpod-volume/models`, then `/weights`);
2. the app.nz model mirror (`APPNZ_MODELS_BASE`, default `https://appstatic.app.nz/models`), with parallel range-resumable downloads and size checks;
3. Hugging Face via `snapshot_download`.

The adapter selects only the Wan files needed for inference, omitting demos and duplicate Flax/PyTorch Wav2Vec checkpoints. Current upstream payloads are approximately 46GB after filtering plus the 1.35GB LiveAvatar LoRA.

The image reuses CUDA, cuDNN, cuBLAS, and NCCL from Cog's CUDA 12.8 base instead of installing duplicate `nvidia-*` wheels with PyTorch. CI imports the complete GPU Python stack from the built image before publishing. This matters for cold starts: the first full-dependency image was 11.5GB compressed and exceeded app.nz's 24-minute community-host pull window.

## Runtime configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `LIVEAVATAR_FP8` | `1` | Replace eligible linear layers with upstream scaled FP8 modules |
| `LIVEAVATAR_OFFLOAD_MODEL` | `1` | CPU-offload components between generation stages |
| `LIVEAVATAR_OFFLOAD_KV` | `0` | Offload autoregressive KV cache to CPU |
| `ENABLE_COMPILE` | `false` | Enable upstream `torch.compile`; higher first-run latency |
| `WEIGHTS_DIR` | platform-dependent | Persistent checkpoint directory |
| `APPNZ_MODELS_BASE` | app.nz mirror | Model mirror root |

## Reproducibility and tests

The image sparse-checkouts upstream commit `76489bc1c6718edbf610009bcd5e5436b0dc8459`; it does not follow a moving branch. Lightweight tests do not import CUDA or download weights:

```sh
uv run --with pillow --with pytest --with pyyaml pytest -q
```

GPU acceptance before marking a release verified:

1. build the Cog image;
2. run a 1-clip preview on H100 with a 3–5 second WAV;
3. inspect identity, motion, lip sync, audio mux, peak VRAM, setup time, and warm prediction time;
4. run the same input twice with the same seed and compare duration/frame count;
5. run a second warm prediction to catch cache or distributed-group lifecycle bugs.

## Safety

Only animate people or characters you have the right and consent to use. Clearly disclose synthetic media where viewers could mistake it for authentic footage. This repository adds no face scraping, voice cloning, impersonation templates, or watermark removal.

## Licenses

Adapter: Apache-2.0. Upstream LiveAvatar: Apache-2.0. Wan2.2-S2V-14B weights: Apache-2.0. LiveAvatar LoRA: Apache-2.0. See `NOTICE` for pinned sources.
