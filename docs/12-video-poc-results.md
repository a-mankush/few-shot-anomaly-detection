# 12 Video POC results

Code: `prototype/poc.py` (enroll, run), `make_demo_video.py`, `eval_video.py`, `run_poc_eval.sh`; additions to `pipeline.py` (`auto_threshold`, `loo_scores`, `video_frames`). Demo: `prototype/demo/bottle.mp4` (input), `bottle_out.mp4` (annotated), `bottle_log.csv`, `bottle_labels.csv`; other runs in `demo/extra/`, summary in `demo/sweep.log`. Date 2026-10-13. Laptop, 8 threads, ORT CPU, DINOv2 ViT-S/14 448 px, N=8.

## What ran
- `enroll --from-video demo/bottle.mp4 --frames 0-6.4s --n 8`: 8 good frames, threshold 0.776 (mean + 3·std of LOO), **3.3 s** (bank 12.6 MB). Folder enrolment uses the same path.
- `run` over a 32 s, 320-frame, 512 px video: annotated mp4 (PASS/FAIL banner, score vs threshold, heatmap on every frame), CSV (frame, time, score, threshold, raw and smoothed verdict), summary.
- **Speed:** 0.62 s mean per scored frame (p95 0.81 s), **1.6 effective FPS** with decode and write included. Inside the ≤ ~1 s target. `--every k` raises the video rate.
- Self-check (`selfcheck.py`) covers the automatic threshold, floor and N<3 fallback.

## Demo video and how it was scored
Built by `make_demo_video.py` from MVTec-AD (CC BY-NC-SA, demo only): enrol segment (8 train images), test-good, defects, test-good, wrong part (hazelnut images), no part (background-coloured frame, plus noise), test-good. 8 frames per image. Every non-enrol frame gets a random crop-based shift of up to ±13 px (±2.5%) and sensor noise, so frames of one image differ. Enrol frames are excluded from scoring. Ground truth is in `*_labels.csv`.
Caveats: the "demo" is stills with synthetic jitter, not real motion; defect images are 8 randomly picked test images per category; one seed.

## Results (frame level, verdict at smooth=1)
| video | false alarms | misses | notes |
|---|---|---|---|
| bottle (every frame) | 14/144 = 9.7% | 7/112 = 6.3% | wrong part 32/32, no part 16/16 caught |
| transistor (every 2nd) | 1/72 = 1.4% | 24/56 = 43% | cut_lead 0/12, misplaced 0/8 caught |
| hazelnut (every 2nd) | 0/72 = 0% | 19/56 = 34% | cut 0/12, print 5/12, hole 8/8 |
| bottle, ±5% shifts | 7/72 = 9.7% | 6/56 = 11% | |
| bottle, brightness ±20% | 3/72 = 4.2% | 5/56 = 8.9% | |
| bottle, defocus blur σ=2.5 px | 5/72 = 6.9% | 0/56 = 0% | blurred good frames mostly still pass; enrolment was unblurred |

- **Wrong part and no part: always FAIL** (bottle, transistor, hazelnut: 100% of 24 and 48 frames, across all variants). Requirement 3 holds. This is the easy case, since these deviate hugely.
- **Subtle defects are missed as predicted:** 34–43% on transistor/hazelnut, matching the 40–50% forecast from ticket 10. The misses are whole defect types (cut lead, misplaced transistor, cut hazelnut), not noise, so no margin tweak fixes them.
- False alarms are 0–10% per video. On bottle they come from a couple of images repeatedly (the log shows runs of FAIL on the same image), so the rate moves by whole images with only 18 good images per video.
- **Smoothing** (computed offline from the bottle log, k consecutive scored frames): k=1 FA 14 / miss 7; k=3 FA **0** / miss 18; k=5 FA 0 / miss 26. It removes the FAs here, but costs misses at the start of each defect segment and on borderline defects. Off by default stays right.
- **Shift:** ±5% crop-based shifts add no false alarms over baseline (9.7% vs 9.7%), consistent with ticket 10. **Lighting ±20%:** no visible harm. **Blur:** no extra misses; blurred good frames mostly pass, but the test is mild and single-pass.
- The stress rows are mismatched by design (enrolment unstressed), the realistic case for drift, but single runs of 8-frame holds: ±5 points is noise.

## Limits
Uniform MVTec backgrounds only; no clutter, rotation, real camera motion, motion blur between frames, or real part change-over. Stress is synthetic. Threshold, resolution and backbone were taken from tickets 09/10 without re-tuning; DINOv2 448 only (WRN not run on video, selectable with `--backend wrn224`).

## What to fix next
1. **Real video** of a fixed scene with a good and a wrong or altered object (ticket 05/11); every number above is MVTec-with-jitter.
2. Decide the smoothing default and per-frame vs per-cycle verdict with the operator-flow ticket (06): k=3 trades FA for misses.
3. Subtle defects (cut lead, misplaced, cut) are the weak spot; ticket 07/08, not a POC issue.
4. Speed: 0.6 s/frame is fine for the demo; skip-frames or the 224 backend if live video is wanted.
