# Few-shot anomaly detection on Raspberry Pi — design spec

Terms (Part cycle, Reference set, Enrollment, Anomaly, Verdict, Leave-one-out threshold, Bad example) are defined in `GLOSSARY.md`.

## Goal
Detect an Anomaly in a fixed physical setup, with a new Reference set (5–15 good images) loaded at the start of each Part cycle. No per-part training, no operator tuning.

## Decisions
| Area | Decision |
|---|---|
| Method | One-class, PatchCore/AnomalyDINO style: backbone patch embeddings, memory bank of Reference-set patches, per-patch nearest-neighbour distance, image score = max patch distance. |
| Supervision | Good images only. Bad examples are never used. |
| Anomaly | Defect on a correct part, wrong part, or no part. One mechanism covers all. |
| Setup | Camera, light, background fixed. One part per frame. Small position shifts tolerated, no rotation, no surface variation. |
| Verdict | Pass unless the image score exceeds the threshold. Saves a heatmap. Leans toward false alarms over misses. |
| Threshold | Automatic: leave-one-out scores over the Reference set, then a fixed margin constant, with a floor so a very uniform set can't give a hair-trigger. No operator controls. |
| Small N | Minimum N = 3. Below it, use a conservative fixed threshold. |
| Enrollment | Operator-started at the start of a Part cycle. Target ≤ ~10 s for N≈5–10 (demo-grade). No training, no gradients. |
| Inference | Target ≤ ~1 s per image or better. |
| Hardware | Raspberry Pi 5 (8 GB) + AI HAT+ (Hailo), CPU fallback. |
| Language | Python (ONNX Runtime on CPU, Hailo runtime for the backbone). Deno rejected: thinner ARM ML ecosystem, and the runtime isn't the hard part. |
| Portability | Same code runs on a developer laptop and the Pi. The only backend-specific piece is the feature extractor, behind one function (`image -> patch features`): ONNX Runtime on laptop and Pi CPU, Hailo runtime on the Pi accelerator. Everything else (memory bank, kNN, threshold, heatmap) is pure numpy. ONNX is the common model format, so every candidate backbone must have a working ONNX path. Images come from a folder, webcam or Pi camera behind a similar one-function source. Dependencies via `uv`, kept to what installs on both x86 and ARM wheels. |
| Memory bank search | Plain numpy matmul on CPU. Coreset and faiss not needed at N ≤ 15 (≈3k–12k vectors, ~18 MB). |
| Backbone and Hailo work split | Backbone on Hailo if it compiles, else CPU. Memory bank and kNN on CPU always. Backbone is fixed across Part cycles, so it compiles once. |

## Open items
1. **Backbone choice** — research tickets 01 and 02 done; decision pending measurements (tickets 03, 04, 08). Shortlist: **DINOv2 ViT-S/14** (primary, start at 448 px) and **ResNet-50/WRN50** on Hailo (compile-safe fallback). DINOv3 and ConvNeXt dropped from the shortlist: no few-shot evidence for DINOv3 ViT-S or ConvNeXt, plain-kNN DINOv3 is worse than DINOv2, and the license is heavier. Findings so far:
   - Hailo model zoo has ResNet-50 (138 FPS on Hailo-8L at batch 1) and 224px ViT/DeiT. No DINOv3, ConvNeXt or WideResNet50 HEFs. ConvNeXt has reported compile failures. DINOv3 compile is undemonstrated (RoPE, LayerNorm, fixed 224px).
   - Safe path: ResNet-50/WRN50 intermediate layers on Hailo. Side experiment: DINOv2/DINOv3 ViT-S, falling back to CPU ONNX.
   - Quality: AnomalyDINO (DINOv2) reports ~96.6% one-shot AUROC on MVTec-AD. One 2026 paper found DINOv3 did not beat DINOv2 for few-shot localisation. Nothing found for small DINOv3.
   - CPU latency on Pi 5 is estimated only (~0.1–0.25 s at 224px, ~1 s at 448px for ViT-S); measure it.
2. **DINOv3 license** — no longer blocking, since DINOv3 is off the shortlist. Custom, not Apache-2.0: uncapped indemnity to Meta, Meta may amend terms at any time, and two live license texts disagree on the "Built with DINOv3" notice. Compiled HEF/ONNX files are likely derivative works. Details and counsel points: `.scratch/anomaly-detection/research/02-dino-license.md`. DINOv2 is Apache-2.0 (ship the license copy and NOTICE).
3. **Margin constant and floor** — calibrate offline on real parts plus a public set (e.g. MVTec-AD). The main thing the prototype must settle.
4. **Operator flow** — deferred by the user: how Enrollment is triggered, how inference is triggered, and where the Verdict is delivered.
5. **Surface-varying parts** — out of v1. Needs a Reference set strategy that covers the variation, probably more images.
6. **Measurements** — Pi 5 per-image latency, Enrollment time for N=5/10/15, Hailo compile result, input resolution vs. small-defect sensitivity.

## Risks
- Very small N makes leave-one-out scores noisy, so the threshold may be unreliable.
- With no operator knob and no bad examples, a wrong margin cannot be corrected in the field.
- Hailo support for non-zoo backbones is unverified; schedule risk sits there.
- Hailo runs a quantized model, so its features differ slightly from the laptop's FP32 ONNX features. A margin constant calibrated on the laptop may not transfer; calibrate per backend, or verify on the Pi.
- Position shifts beyond "small" will raise false alarms.

## Next step
Prototype with ResNet-50 (Hailo) or CPU ONNX, numpy kNN, leave-one-out threshold, then run the measurements in open item 6 and settle open items 1 and 3.
