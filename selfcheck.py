"""The one self-check: LOO threshold logic. uv run selfcheck.py"""
import numpy as np, pipeline as P
rng = np.random.default_rng(0)
def unit(x): return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
base = rng.normal(size=(49, 16))
ref = [unit(base + .05 * rng.normal(size=base.shape)) for _ in range(5)]
thr, loo = P.loo_threshold(ref, margin=1.1, floor=0.0)
assert len(loo) == 5 and abs(thr - max(loo) * 1.1) < 1e-6
assert thr > max(loo)                                            # margin applied
assert P.loo_threshold(ref, margin=1.1, floor=9.0)[0] == 9.0     # floor wins
assert P.loo_threshold(ref[:2])[0] == P.FALLBACK_THR             # N<3 fallback
bank = np.concatenate(ref)
good = unit(base + .05 * rng.normal(size=base.shape)); bad = good.copy(); bad[7] = unit(rng.normal(size=(1, 16)))
sg, sb = P.score(good, bank, 7)[0], P.score(bad, bank, 7)[0]
assert sg <= thr < sb, (sg, thr, sb)
print("ok")
# ticket 12: automatic threshold
loo = [0.4, 0.5, 0.6]
assert abs(P.auto_threshold(loo, "dino448", floor=0.0) - (np.mean(loo) + 3 * np.std(loo))) < 1e-9
assert P.auto_threshold([0.01, 0.01, 0.01], "dino448") == P.FLOOR                 # floor wins
assert P.auto_threshold([], "dino448") == 0.93 and P.auto_threshold([0.5], "wrn448") == 0.68
assert P.loo_scores(ref[:2]) == [] and len(P.loo_scores(ref)) == 5
print("ok2")
