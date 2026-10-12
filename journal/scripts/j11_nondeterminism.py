"""Reproduce the pre-fix non-determinism measurement.

The paper claims, in the abstract and in Section VI-G, that at a fixed seed
repeated runs produced models differing by up to |dF1| = 0.20 on the real
split, with per-sample prediction differences up to 1.0. Nothing in the
repository records that measurement: `scripts/08_determinism_check.py` verifies
the *fixed* behaviour, and `results/` holds only post-fix artifacts. The number
is therefore currently unverifiable, which is why it is being re-measured here.

Method. `train_model` wraps training in `deterministic_run`, which pins the
thread count to one and enables deterministic algorithms. We bypass that
context to restore the pre-fix condition (multi-threaded CPU, non-deterministic
`index_add_` reduction order), train the same seed repeatedly, and record the
spread. The post-fix path is run alongside as the control.

Run:  python journal/scripts/j11_nondeterminism.py
"""
import contextlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402

sys.path.insert(0, str(J.ROOT))
import crossexplain.train as T                                # noqa: E402
from crossexplain.train import train_model, predict           # noqa: E402

SEEDS = [0, 1]
REPS = 3
THREADS = 4

train = J.load("train")
val = J.load("val")
real = J.load("test_real")
y = J.y_of(real)


@contextlib.contextmanager
def _permissive(seed, threads=THREADS):
    """The pre-fix condition: multiple threads, no determinism enforcement.

    Seeds are still set, so any residual spread is attributable to reduction
    order rather than to initialization.
    """
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


J.banner("Re-measuring the non-determinism claim (abstract and Section VI-G)")
print(f"  {REPS} repeats per seed, {THREADS} threads, seeds {SEEDS}")
print(f"  torch {torch.__version__}, {torch.get_num_threads()} threads available")

OUT = {"reps": REPS, "threads": THREADS, "torch": torch.__version__}

for label, permissive in (("pre-fix (multi-threaded, non-deterministic)", True),
                          ("post-fix (deterministic_run, as shipped)", False)):
    print(f"\n  --- {label} ---")
    rows = []
    for sd in SEEDS:
        f1s, probs = [], []
        for rep in range(REPS):
            t0 = time.time()
            if permissive:
                orig = T.deterministic_run
                T.deterministic_run = _permissive
                try:
                    m = train_model(train, val, arch="gat", layers=5,
                                    epochs=40, verbose=False, seed=sd,
                                    threads=THREADS)
                finally:
                    T.deterministic_run = orig
            else:
                m = train_model(train, val, arch="gat", layers=5, epochs=40,
                                verbose=False, seed=sd)
            p = np.array(predict(m, real)[0])
            probs.append(p)
            f1s.append(J.f1(p > 0.5, y))
            print(f"    seed {sd} rep {rep}: F1={f1s[-1]:.4f}  ({time.time() - t0:.0f}s)")
        dmax = max(abs(f1s[i] - f1s[j])
                   for i in range(REPS) for j in range(i + 1, REPS))
        pmax = max(float(np.abs(probs[i] - probs[j]).max())
                   for i in range(REPS) for j in range(i + 1, REPS))
        rows.append({"seed": sd, "f1s": [float(x) for x in f1s],
                     "max_abs_delta_f1": float(dmax), "max_abs_delta_p": pmax})
        print(f"    seed {sd}: max|dF1|={dmax:.4f}  max|dp|={pmax:.4f}  "
              f"{'DETERMINISTIC' if pmax < 1e-6 else 'NON-DETERMINISTIC'}")
    key = "pre_fix" if permissive else "post_fix"
    OUT[key] = {"per_seed": rows,
                "max_abs_delta_f1": max(r["max_abs_delta_f1"] for r in rows),
                "max_abs_delta_p": max(r["max_abs_delta_p"] for r in rows)}

J.banner("Verdict on the paper's claim")
pre, post = OUT["pre_fix"], OUT["post_fix"]
print(f"  paper states: |dF1| up to 0.20, per-sample differences up to 1.0")
print(f"  re-measured : |dF1| up to {pre['max_abs_delta_f1']:.4f}, "
      f"per-sample up to {pre['max_abs_delta_p']:.4f}")
print(f"  post-fix    : |dF1| = {post['max_abs_delta_f1']:.4f}, "
      f"per-sample = {post['max_abs_delta_p']:.4f}")
supported = pre["max_abs_delta_f1"] >= 0.15
print(f"\n  claim of 0.20 is {'SUPPORTED (same order)' if supported else 'NOT REPRODUCED at this magnitude'}")
print("  note: the exact maximum depends on thread count and hardware, so the")
print("  paper should quote what this run measures, on the machine described.")
OUT["claim_supported_same_order"] = bool(supported)

J.save("j11_nondeterminism", OUT)
