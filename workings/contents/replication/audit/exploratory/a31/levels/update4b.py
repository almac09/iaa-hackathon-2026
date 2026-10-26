"""Scratch (A31): the halfway file with only motor new business moved, scored across the four-score family."""
import io, contextlib, sys  # noqa: E401
sys.argv = ["x"]
with contextlib.redirect_stdout(io.StringIO()):
    import update4 as u  # noqa: E402
import numpy as np  # noqa: E402
base = u.cd.cands["midpoint: as is and level reset + emerging motor NB loading"]
for lo, hi in ((0.7, 1.4), (0.6, 1.5)):
    fam = u.family(lo, hi)
    print(f"box [{lo}, {hi}]")
    for m in (0.981, 1.02, 1.05, 1.08):
        sc = dict(base) | {"motor_nb": m}
        v = np.concatenate([u.S_of(u.cd.vec(sc), cn, beta, k) for cn, beta, k in fam])
        print(f"  halfway file, motor new business x{m:.3f} of as is: {v.min():.5f} / {np.median(v):.5f} / {v.max():.5f}; beats 0.15742 in {(v > 0.15742 + 1e-9).mean():.2f}")
