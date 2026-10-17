"""PROTOTYPE (throwaway): DINOv2 patch-kNN anomaly check. See README.md."""
import glob, os, time
import cv2, numpy as np, onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MIN_N = 3
FALLBACK_THR = 0.9   # fixed threshold below MIN_N (cosine-distance units, see results)
FLOOR = 0.3          # LOO-threshold floor
MARGIN = 1.1         # experiment flag only; no operator control in the final shape


class OnnxDino:
    """Backend seam: anything with .extract(bgr_image) -> (N, D) float32 slots in here."""
    def __init__(self, res=448, layer="last", model_dir=os.path.join(os.path.dirname(__file__), "models")):
        self.res, self.layer = res, layer
        so = ort.SessionOptions()
        self.s = ort.InferenceSession(os.path.join(model_dir, f"dinov2_s14_{res}.onnx"), so, providers=["CPUExecutionProvider"])

    def raw(self, img):  # -> {"last": (N,D), "mid": (N,D)}
        x = cv2.cvtColor(cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((x.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        last, mid = self.s.run(None, {"x": x})
        return {"last": last[0], "mid": mid[0]}

    def extract(self, img):
        f = self.raw(img)[self.layer]
        return (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float32)

    @property
    def grid(self): return self.res // 14


class OnnxWrn(OnnxDino):
    """WRN50-2 layer2+layer3 patch features (PatchCore-style, 768-d), same seam. Stride 8 -> grid res//8."""
    def __init__(self, res=448, layer="last", model_dir=os.path.join(os.path.dirname(__file__), "models")):
        self.res, self.layer = res, "last"
        self.s = ort.InferenceSession(os.path.join(model_dir, f"wrn50_{res}.onnx"), ort.SessionOptions(), providers=["CPUExecutionProvider"])

    def raw(self, img):
        x = cv2.cvtColor(cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((x.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        return {"last": self.s.run(None, {"x": x})[0][0]}

    @property
    def grid(self): return self.res // 8


def make_backend(name="dino448", layer="last"):
    if name.startswith("dino"): return OnnxDino(int(name[4:]), layer)
    if name.startswith("wrn"): return OnnxWrn(int(name[3:]), layer)
    raise ValueError(name)  # ponytail: add "hailo" here


def nn_dist(q, bank, chunk=2048):
    """Per-patch NN distance via ||q||²+||b||²-2q·b. Unit-norm features -> sqrt gives chordal distance in [0, 2]."""
    bb = (bank ** 2).sum(1)
    out = np.empty(len(q), np.float32)
    for i in range(0, len(q), chunk):
        c = q[i:i + chunk]
        d = (c ** 2).sum(1)[:, None] + bb[None] - 2 * c @ bank.T
        out[i:i + chunk] = d.min(1)
    return np.sqrt(np.maximum(out, 0))


def score(feats, bank, grid, size=None):
    d = nn_dist(feats, bank)
    hm = d.reshape(grid, grid)
    if size: hm = cv2.GaussianBlur(cv2.resize(hm, size, interpolation=cv2.INTER_LINEAR), (0, 0), 4)
    return float(d.max()), hm


def loo_threshold(feats_list, margin=MARGIN, floor=FLOOR, fallback=FALLBACK_THR):
    """Score each reference image against the bank from the others; worst * margin, floored. N<3 -> fallback."""
    n = len(feats_list)
    if n < MIN_N: return fallback, []
    scores = []
    for i in range(n):
        bank = np.concatenate([f for j, f in enumerate(feats_list) if j != i])
        scores.append(float(nn_dist(feats_list[i], bank).max()))
    return max(max(scores) * margin, floor), scores


def images(source):
    """Yield (name, bgr) from a folder or a camera device path/index."""
    if os.path.isdir(source):
        for p in sorted(glob.glob(os.path.join(source, "*"))):
            im = cv2.imread(p)
            if im is not None: yield os.path.basename(p), im
    elif os.path.isfile(source):
        yield os.path.basename(source), cv2.imread(source)
    else:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        try:
            ok, im = cap.read()   # one frame per call of the generator; caller breaks when done
            while ok:
                yield "camera", im
                ok, im = cap.read()
        finally:
            cap.release()


def enroll(folder, backend):
    t = time.perf_counter()
    feats = [backend.extract(im) for _, im in images(folder)]
    t_ext = time.perf_counter() - t
    thr, loo = loo_threshold(feats)
    return dict(bank=np.concatenate(feats), feats=feats, thr=thr, loo=loo, t_extract=t_ext, t_total=time.perf_counter() - t)


# --- video POC additions (ticket 12) ---
K_STD = 3.0                              # ticket 10: mean + 3*std of LOO scores
FALLBACKS = {"dino": 0.93, "wrn": 0.68}  # N<3 fixed thresholds per backend family (ticket 10)


def loo_scores(feats_list):
    """Max NN distance of each reference image against the bank of the others. [] below MIN_N."""
    n = len(feats_list)
    if n < MIN_N: return []
    return [float(nn_dist(feats_list[i], np.concatenate([f for j, f in enumerate(feats_list) if j != i])).max()) for i in range(n)]


def auto_threshold(loo, backend_name, k=K_STD, floor=FLOOR):
    """Automatic threshold: mean + k*std of LOO scores (floored); fixed per-backend fallback below MIN_N."""
    if len(loo) < MIN_N: return FALLBACKS["wrn" if backend_name.startswith("wrn") else "dino"]
    return max(float(np.mean(loo) + k * np.std(loo)), floor)


def video_frames(path, start=0.0, end=None, step=1):
    """Yield (index, time_s, bgr) from a video file, restricted to [start, end) seconds, every step-th frame."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise SystemExit(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        i = 0
        while True:
            ok, im = cap.read()
            if not ok: break
            t = i / fps
            if t >= (end if end is not None else 1e18): break
            if t >= start and (i % step == 0): yield i, t, im
            i += 1
    finally:
        cap.release()


def video_info(path):
    cap = cv2.VideoCapture(path)
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 30.0, n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info
