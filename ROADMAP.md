# Character tooling roadmap

Snapshot: 2026-07-13. The order favors released weights, commercial-friendly licensing, differentiated capability, and a serving shape that fits either Cog or an explicit streaming service.

| Rank | Project | Product shape | Why next | Main gate |
| ---: | --- | --- | --- | --- |
| 1 | LiveAvatar v1.1 | Image + audio + prompt → long MP4; later streaming | Apache-2.0, 4-step, stylized characters, real streaming architecture, no fal endpoint found | H100 GPU acceptance and cold-start measurement |
| 2 | EchoAvatar | Audio stream + intent tools → 3D body motion | 2026 real-time interactive humanoid motion with LLM tool-call control; genuinely game/agent oriented | Confirm released code, weights, character format, and license |
| 3 | NVIDIA Audio2Face open models | Audio → blendshapes/emotion | High-value explicit 3D/game output, real-time, Maya/UE integrations | Package SDK redistribution terms and define a neutral JSON/ARKit schema |
| 4 | LongCat-Video-Avatar 1.5 | Image/audio/text → 8-step video | Strong 2026 production baseline, multi-speaker and stylized-domain support | fal already serves LongCat; compete only on v1.5 controls/cost |
| 5 | ALIVE | Image/video + audio/text → synchronized AV animation | Broader than talking heads and released in 2026 | Verify public inference weights and license before engineering |
| 6 | CartoonAlive | Cartoon image → rig/animation | Best route to editable character assets rather than pixels-only video | Code/weights maturity and export format |
| 7 | LivePortraitKJ / MuseTalk | Low-latency portrait fallback | Cheap, familiar, deployable on smaller GPUs | Commoditized; only worthwhile as a real-time budget tier |
| 8 | FaceFormer / CodeTalker | Audio + registered mesh → vertices | Deterministic game pipeline and compact output | Requires mesh registration and clearer end-user onboarding |

Older diffusion baselines (EchoMimic, Hallo3, SadTalker, Real3D-Portrait) remain useful for regression comparisons, not first-class new hosted products unless they win on cost or controllability.

## Shared acceptance harness

Every backend should expose a small common manifest even when its native input differs:

- capability and version;
- input/output media kinds;
- license provenance for code and weights;
- minimum and tested GPU/VRAM;
- cold setup time, warm latency, output seconds per GPU-second, peak VRAM;
- deterministic seed behavior;
- identity, lip-sync, temporal, motion, and artifact scores on consented fixtures;
- bounded input duration/resolution and explicit failure cases.

Do not force streaming motion, editable meshes, and offline diffusion video into one API. Share evaluation and provenance; keep serving contracts native to the workload.
