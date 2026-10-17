# Pull a few MVTec-AD categories (CC BY-NC-SA, eval only) from the open Voxel51 mirror.
import json, sys, urllib.request, pathlib
from concurrent.futures import ThreadPoolExecutor
B = "https://huggingface.co/datasets/Voxel51/mvtec-ad/resolve/main/"
cats = sys.argv[1:] or ["bottle", "screw", "hazelnut"]
S = json.load(open("data/v51/samples.json"))["samples"]
jobs = []
for s in S:
    c, d, sp = s["category"]["label"], s["defect"]["label"], s["split"]
    if c not in cats: continue
    out = pathlib.Path("data/mvtec") / c / sp / d / (s["filepath"].replace("/", "_"))
    jobs.append((s["filepath"], out))
# keep 40 good train images per category, all test
n = {}
keep = []
for fp, out in jobs:
    if out.parts[-4 + 0 - 0] and "train" in out.parts:
        k = out.parts[2]; n[k] = n.get(k, 0) + 1
        if n[k] > 40: continue
    keep.append((fp, out))
def get(j):
    fp, out = j; out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.write_bytes(urllib.request.urlopen(B + fp).read())
with ThreadPoolExecutor(16) as ex: list(ex.map(get, keep))
print(len(keep), "files")
