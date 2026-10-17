"""POC: enrol a Reference set, then run the detector over a video. uv run poc.py enroll|run --help"""
import argparse, csv, json, os, time
import cv2, numpy as np
import pipeline as P


def parse_window(s):
    """'0-10s' -> (0.0, 10.0); '1.5,3,7' -> explicit timestamps list."""
    if "," in s or "-" not in s:
        return None, [float(x.rstrip("s")) for x in s.split(",")]
    a, b = s.rstrip("s").split("-")
    return (float(a), float(b)), None


def grab(path, t):
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, im = cap.read(); cap.release()
    if not ok: raise SystemExit(f"no frame at {t}s in {path}")
    return im


def reference_images(a):
    if a.from_video:
        win, stamps = parse_window(a.frames)
        if stamps is None:
            stamps = list(np.linspace(win[0], win[1], a.n, endpoint=False))   # N evenly spaced inside the window
        return [(f"{a.from_video}@{t:.2f}s", grab(a.from_video, t)) for t in stamps]
    return list(P.images(a.source))


def cmd_enroll(a):
    be = P.make_backend(a.backend)
    imgs = reference_images(a)
    if not 1 <= len(imgs) <= 15: print(f"warning: N={len(imgs)}, expected 3-15 good images")
    t0 = time.perf_counter()
    feats = [be.extract(im) for _, im in imgs]
    loo = P.loo_scores(feats)
    thr = P.auto_threshold(loo, a.backend)
    dt = time.perf_counter() - t0
    bank = np.concatenate(feats)
    np.savez(a.state, bank=bank, thr=thr, loo=np.array(loo), backend=a.backend, n=len(imgs))
    how = "mean+3std of LOO" if loo else f"fixed fallback (N<{P.MIN_N})"
    print(f"enrolled N={len(imgs)} backend={a.backend} bank={bank.shape} ({bank.nbytes/1e6:.1f} MB) "
          f"threshold={thr:.3f} [{how}] loo={np.round(loo, 3).tolist()} time={dt:.2f}s -> {a.state}")


def banner(im, fail, sc, thr):
    h, w = im.shape[:2]
    col = (0, 0, 220) if fail else (0, 170, 0)
    cv2.rectangle(im, (0, 0), (w, 44), col, -1)
    cv2.putText(im, f"{'FAIL' if fail else 'PASS'}   score {sc:.3f} / thr {thr:.3f}", (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def cmd_run(a):
    s = np.load(a.state); bank, thr, name = s["bank"], float(s["thr"]), str(s["backend"])
    be = P.make_backend(name)
    info = P.video_info(a.video)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["w"], info["h"]))
    rows, last, streak = [], None, 0
    lat, t_wall = [], time.perf_counter()
    for i, t, im in P.video_frames(a.video, step=1):
        if i % a.every == 0:
            t0 = time.perf_counter()
            sc, hm = P.score(be.extract(im), bank, be.grid, (im.shape[1], im.shape[0]))
            lat.append(time.perf_counter() - t0)
            raw_fail = sc > thr
            streak = streak + 1 if raw_fail else 0
            fail = streak >= a.smooth if a.smooth > 1 else raw_fail
            hm8 = cv2.applyColorMap(np.clip(hm / max(thr * 1.5, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            last = (sc, fail, cv2.addWeighted(im, .55, hm8, .45, 0))
            rows.append(dict(frame=i, time=round(t, 3), score=round(sc, 4), threshold=round(thr, 4),
                             raw_verdict="FAIL" if raw_fail else "PASS", verdict="FAIL" if fail else "PASS"))
        sc, fail, over = last
        out = over.copy(); banner(out, fail, sc, thr); writer.write(out)
    writer.release()
    wall = time.perf_counter() - t_wall
    with open(a.log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fails = sum(r["verdict"] == "FAIL" for r in rows)
    print(f"frames read={info['n']} scored={len(rows)} (every {a.every}) fail={fails} pass={len(rows)-fails} smooth={a.smooth} "
          f"latency mean={np.mean(lat)*1000:.0f} ms p95={np.percentile(lat, 95)*1000:.0f} ms "
          f"effective scoring FPS={len(rows)/wall:.2f} wall={wall:.1f}s -> {a.out}, {a.log}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll", help="enrol a Reference set from a folder or the first good frames of a video")
    e.add_argument("source", nargs="?", help="folder of 3-15 good images (omit with --from-video)")
    e.add_argument("--from-video"); e.add_argument("--frames", default="0-10s", help="window '0-10s' or timestamps '1,2.5,4'")
    e.add_argument("--n", type=int, default=8, help="frames sampled from the window")
    e.add_argument("--backend", default="dino448"); e.add_argument("--state", default="enrolled.npz")
    r = sub.add_parser("run", help="score a video frame by frame")
    r.add_argument("video"); r.add_argument("--out", default="out.mp4"); r.add_argument("--log", default="out.csv")
    r.add_argument("--state", default="enrolled.npz"); r.add_argument("--every", type=int, default=1, help="score every k-th frame")
    r.add_argument("--smooth", type=int, default=1, help="FAIL only after k consecutive raw FAILs (1 = off)")
    a = ap.parse_args()
    if a.cmd == "enroll" and not (a.source or a.from_video): ap.error("give a folder or --from-video")
    {"enroll": cmd_enroll, "run": cmd_run}[a.cmd](a)
