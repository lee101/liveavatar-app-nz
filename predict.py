import gc
import os
import random
import socket
import subprocess
import sys
import tempfile
from pathlib import Path as FilePath

from cog import BasePredictor, Input, Path

from inputs import generation_options, media_duration, validate_image
from weights import BASE_ALLOW_PATTERNS, BASE_REPO, LORA_REPO, ensure_weights

UPSTREAM_ROOT = FilePath(os.environ.get("LIVEAVATAR_ROOT", "/opt/liveavatar"))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _mux(video: FilePath, audio: str, output: FilePath) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-i",
            audio,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output),
        ],
        check=True,
    )


class Predictor(BasePredictor):
    def setup(self) -> None:
        os.environ.setdefault("ENABLE_COMPILE", "false")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        sys.path.insert(0, str(UPSTREAM_ROOT))

        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            raise RuntimeError("LiveAvatar requires an NVIDIA GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(0)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=0,
                world_size=1,
                init_method=f"tcp://127.0.0.1:{_free_port()}",
            )

        from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
        from liveavatar.models.wan.wan_2_2.configs import WAN_CONFIGS

        checkpoint = ensure_weights(BASE_REPO, allow_patterns=BASE_ALLOW_PATTERNS)
        lora = ensure_weights(LORA_REPO) / "liveavatar.safetensors"
        if not lora.is_file():
            raise FileNotFoundError(f"missing LiveAvatar LoRA: {lora}")

        self.config = WAN_CONFIGS["s2v-14B"]
        self.pipeline = WanS2V(
            config=self.config,
            checkpoint_dir=str(checkpoint),
            device_id=0,
            rank=0,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=True,
            single_gpu=True,
            offload_kv_cache=os.environ.get("LIVEAVATAR_OFFLOAD_KV", "0") == "1",
        )
        self.pipeline.noise_model = self.pipeline.add_lora_to_model(
            self.pipeline.noise_model,
            lora_rank=128,
            lora_alpha=64.0,
            lora_target_modules="q,k,v,o,ffn.0,ffn.2",
            init_lora_weights="kaiming",
            pretrained_lora_path=str(lora),
            load_lora_weight_only=False,
        )
        if os.environ.get("LIVEAVATAR_FP8", "1") == "1":
            from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8

            replace_linear_with_scaled_fp8(
                self.pipeline.noise_model,
                ignore_keys=[
                    "text_embedding",
                    "time_embedding",
                    "time_projection",
                    "head.head",
                    "casual_audio_encoder.encoder.final_linear",
                ],
            )
        self.pipeline.set_eval()

    def predict(
        self,
        image: Path = Input(description="Reference portrait or character image"),
        audio: Path = Input(description="Speech, singing, or other driving audio (max 300 seconds)"),
        prompt: str = Input(
            default="A character speaks naturally with expressive facial movement and body gestures.",
            description="Appearance, action, camera, and scene guidance",
        ),
        quality: str = Input(
            default="standard",
            choices=["preview", "standard", "high"],
            description="Pixel-area tier; input aspect ratio is preserved",
        ),
        num_clips: int = Input(
            default=1,
            ge=1,
            le=8,
            description="Maximum autoregressive clips; generation also stops when audio ends",
        ),
        steps: int = Input(default=4, ge=4, le=8, description="DMD sampling steps; 4 is recommended"),
        seed: int = Input(default=None, description="Random seed; blank chooses one"),
        start_from_reference: bool = Input(default=True, description="Use the reference image as frame zero"),
    ) -> Path:
        import torch
        from liveavatar.models.wan.wan_2_2.utils.utils import save_video

        image_path = str(image)
        audio_path = str(audio)
        validate_image(image_path)
        media_duration(audio_path)
        actual_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        options = generation_options(
            prompt=prompt,
            image=image_path,
            audio=audio_path,
            quality=quality,
            num_clips=num_clips,
            steps=steps,
            seed=actual_seed,
            start_from_reference=start_from_reference,
            offload_model=os.environ.get("LIVEAVATAR_OFFLOAD_MODEL", "1") == "1",
        )

        work = FilePath(tempfile.mkdtemp(prefix="liveavatar-"))
        silent = work / "silent.mp4"
        output = work / "output.mp4"
        video = None
        try:
            with torch.inference_mode():
                video, _ = self.pipeline.generate(**options)
            save_video(
                tensor=video[None],
                save_file=str(silent),
                fps=self.config.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            if not silent.is_file() or silent.stat().st_size == 0:
                raise RuntimeError("LiveAvatar did not encode a video")
            _mux(silent, audio_path, output)
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("LiveAvatar did not produce an output video")
            return Path(str(output))
        finally:
            del video
            gc.collect()
            torch.cuda.empty_cache()
