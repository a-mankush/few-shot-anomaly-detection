"""Compare a poc.py run log with make_demo_video.py labels. uv run eval_video.py out.csv demo/x_labels.csv"""
import csv, sys, collections
log = {int(r["frame"]): r for r in csv.DictReader(open(sys.argv[1]))}
lab = list(csv.DictReader(open(sys.argv[2])))
use = "raw_verdict" if "--raw" in sys.argv else "verdict"
st = collections.defaultdict(lambda: [0, 0])      # kind -> [correct, total]
tot = collections.Counter()
for r in lab:
    f = int(r["frame"])
    if r["label"] == "enroll" or f not in log: continue
    fail = log[f][use] == "FAIL"
    ok = fail == (r["label"] == "bad")
    k = r["kind"].split(":")[0] if r["kind"].startswith(("wrong", "no_")) else r["kind"]
    st[k][0] += ok; st[k][1] += 1
    tot[(r["label"], fail)] += 1
for k, (c, n) in sorted(st.items()): print(f"{k:28s} {c:4d}/{n:<4d} {c/n:6.1%} correct")
good = tot[("good", False)] + tot[("good", True)]; bad = tot[("bad", False)] + tot[("bad", True)]
print(f"false alarms {tot[('good', True)]}/{good} = {tot[('good', True)]/max(good,1):.1%}   misses {tot[('bad', False)]}/{bad} = {tot[('bad', False)]/max(bad,1):.1%}   ({use})")
