"""Build a demo video + ground-truth labels from MVTec-AD (CC BY-NC-SA, demonstration/evaluation only).
uv run make_demo_video.py [--category bottle] [--out demo] [--stress none|shift4|lighting|blur]
Layout: enrol segment (train good) | test good | defects | test good | wrong part | no part | test good.
Shifts are crop-based (render at R px, crop S px at a random offset), never replicated borders."""
import argparse, csv, glob, os
import cv2, numpy as np

S, R, FPS, HOLD = 512, 540, 10, 8   # frame size, render size (max shift +-14 px = 2.7%), fps, frames per image


def render(path, off=(0, 0), stress="none", rng=None):
    im = cv2.resize(cv2.imread(path), (R, R), interpolation=cv2.INTER_AREA)
    m = (R - S) // 2
    if stress == "shift4": off = (off[0] * 2, off[1] * 2)   # ~5% shifts: needs R larger
    dx, dy = off
    x0, y0 = int(np.clip(m + dx, 0, R - S)), int(np.clip(m + dy, 0, R - S))
    f = im[y0:y0 + S, x0:x0 + S].astype(np.float32)
    f += rng.normal(0, 2, f.shape)                                  # sensor noise
    if stress == "lighting": f *= rng.uniform(0.8, 1.2)             # brightness jitter +-20%
    f = np.clip(f, 0, 255).astype(np.uint8)
    if stress == "blur": f = cv2.GaussianBlur(f, (0, 0), 2.5)       # defocus / motion-blur stand-in
    return f


def pick(paths, k, rng): return [paths[i] for i in rng.choice(len(paths), min(k, len(paths)), replace=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle"); ap.add_argument("--other", default="hazelnut", help="category used as 'wrong part'")
    ap.add_argument("--out", default="demo"); ap.add_argument("--stress", default="none", choices=["none", "shift4", "lighting", "blur"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--name", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = "data/mvtec"; cat = f"{root}/{a.category}"
    train = sorted(glob.glob(f"{cat}/train/good/*.png")); good = sorted(glob.glob(f"{cat}/test/good/*.png"))
    defects = sorted(p for p in glob.glob(f"{cat}/test/*/*.png") if "/good/" not in p)
    other = sorted(glob.glob(f"{root}/{a.other}/test/good/*.png"))
    ref_imgs = pick(train, 8, rng); g = pick(good, 18, rng)
    d = pick(defects, 8, rng); o = pick(other, 4, rng)
    # (path or 'blank', label, kind)
    segs = [(p, "enroll", "train_good") for p in ref_imgs]
    segs += [(p, "good", "test_good") for p in g[:6]]
    segs += [(p, "bad", "defect:" + os.path.basename(os.path.dirname(p))) for p in d]
    segs += [(p, "good", "test_good") for p in g[6:12]]
    segs += [(p, "bad", f"wrong_part:{a.other}") for p in o]
    segs += [("blank", "bad", "no_part")]*2
    segs += [(p, "good", "test_good") for p in g[12:18]]
    os.makedirs(a.out, exist_ok=True)
    name = a.name or (f"{a.category}" + ("" if a.stress == "none" else f"_{a.stress}"))
    vpath, lpath = f"{a.out}/{name}.mp4", f"{a.out}/{name}_labels.csv"
    w = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (S, S))
    rows, n = [], 0
    for p, label, kind in segs:
        if p == "blank":
            ref = render(train[0], stress="none", rng=rng)
            bg = np.full((S, S, 3), np.median(ref[:20, :], axis=(0, 1)), np.uint8)   # background-coloured, no part
        for _ in range(HOLD):
            off = (0, 0) if label == "enroll" else tuple(rng.integers(-13, 14, 2))   # enrol segment is centred; the rest jitters
            f = bg.copy() if p == "blank" else render(p, off, a.stress if label != "enroll" else "none", rng)
            if p == "blank": f = np.clip(f.astype(np.float32) + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8)
            w.write(f); rows.append((n, round(n / FPS, 2), label, kind)); n += 1
    w.release()
    with open(lpath, "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["frame", "time", "label", "kind"]); cw.writerows(rows)
    print(f"{vpath}: {n} frames @ {FPS} fps ({n/FPS:.0f}s); enrol segment 0-{len(ref_imgs)*HOLD/FPS:.1f}s; labels -> {lpath}")


main()
