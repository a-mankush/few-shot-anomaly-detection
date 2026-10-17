# PROTOTYPE (throwaway): few-shot anomaly detection, laptop video POC

DINOv2 ViT-S/14 patch-kNN. Enrol a Reference set, run over a video, get PASS/FAIL + heatmap per frame.

    uv sync && uv run export_onnx.py                                       # 1. setup (one-time ONNX export; torch is dev-only)
    uv run poc.py enroll demo/reference_folder                             # 2. enrol: a folder of 3-15 good images ...
    uv run poc.py enroll --from-video clip.mp4 --frames 0-6.4s --n 8       #    ... or good frames of a video (or --frames 1,2.5,4)
    uv run poc.py run clip.mp4 --out out.mp4 --log out.csv                 # 3. run: annotated video + CSV log + summary

`run` options: `--every k` score every k-th frame, `--smooth k` report FAIL only after k consecutive raw FAILs (default off; the log keeps `raw_verdict`). The threshold is automatic (mean + 3·std of leave-one-out scores; fixed 0.93 below N=3); no tuning flags. `enroll --backend wrn224` selects the WRN comparison backend (needs `uv run export_resnet.py`).

Demo: `uv run make_demo_video.py --category bottle --out demo` builds a video + `*_labels.csv` from MVTec-AD (CC BY-NC-SA, demo/evaluation only; fetch with `uv run fetch_mvtec.py`). `uv run eval_video.py out.csv demo/bottle_labels.csv` scores a run against the labels. `./run_poc_eval.sh` repeats the sweep over other categories and stress variants. `uv run selfcheck.py` checks the threshold logic.

Older experiments (tickets 09/10): `cli.py`, `bench.py`, `eval_mvtec.py`, `exp10_*.py`; see `research/09-*` and `research/10-*` results.

## Not in this repo
- `models/*.onnx` (85 MB): run `uv run export_onnx.py` once to create it.
- `demo/*.mp4`: built from MVTec-AD (CC BY-NC-SA, non-commercial). Run `uv run fetch_mvtec.py` then `uv run make_demo_video.py --category bottle --out demo`. The labels and log CSVs are included.
