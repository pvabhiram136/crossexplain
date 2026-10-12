"""Dataset construction: generate designs → synthesize (parallel) → extract
crossing samples → serialize splits.

Splits
------
train      160 designs, 2–3 clocks, 3–6 crossings, levels O0+O2
val         20 designs, same distribution, levels O0+O2
test_id     40 designs, same distribution, ALL levels (O1/O3 are
            configurations never seen in training)
test_big    16 designs, 4 clocks, 7–9 crossings (structural OOD)
test_real   hand-written UART/SPI/FIFO/handshake designs, ALL levels
"""
from __future__ import annotations

import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .rtl_generator import RTLGenerator, write_design
from .real_designs import real_designs
from .synthesize import synthesize, OPT_LEVELS
from .netlist_graph import extract_samples

TRAIN_LEVELS = ["O0", "O2"]


def _make_specs(root: Path):
    """Materialize all design specs + Verilog on disk; return job list."""
    meta = random.Random(1234)
    jobs = []  # (design_name, vfile, gt, level, split)

    def gen_block(split, seeds, n_clocks_rng, n_cross_rng, levels):
        d = root / "designs" / split
        for sd in seeds:
            g = RTLGenerator(seed=sd)
            nc = meta.choice(n_clocks_rng)
            nx = meta.choice(n_cross_rng)
            spec = g.generate(f"{split}_{sd:04d}", n_clocks=nc, n_crossings=nx)
            write_design(spec, d)
            for lvl in levels:
                jobs.append((spec.name, d / f"{spec.name}.v",
                             spec.ground_truth(), lvl, split))

    gen_block("train", range(0, 160), [2, 3], [3, 4, 5, 6], TRAIN_LEVELS)
    gen_block("val", range(200, 220), [2, 3], [3, 4, 5, 6], TRAIN_LEVELS)
    gen_block("test_id", range(300, 340), [2, 3], [3, 4, 5, 6], OPT_LEVELS)
    gen_block("test_big", range(400, 416), [4], [7, 8, 9], OPT_LEVELS)

    d = root / "designs" / "test_real"
    for spec in real_designs():
        write_design(spec, d)
        for lvl in OPT_LEVELS:
            jobs.append((spec.name, d / f"{spec.name}.v",
                         spec.ground_truth(), lvl, "test_real"))
    return jobs


def _run_job(job):
    name, vfile, gt, lvl, split = job
    jfile = vfile.parent / f"{name}.{lvl}.json"
    if not jfile.exists():
        if not synthesize(vfile, jfile, lvl):
            return split, name, lvl, None
    try:
        samples, _ = extract_samples(jfile, gt, name, lvl)
    except Exception as e:  # noqa: BLE001 — record and continue
        print(f"[extract FAIL] {name} @{lvl}: {e}")
        return split, name, lvl, None
    return split, name, lvl, [asdict(s) for s in samples]


def build_dataset(root: Path, workers: int = 6):
    root = Path(root)
    jobs = _make_specs(root)
    print(f"{len(jobs)} synthesis jobs")
    buckets = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                split, name, lvl, samples = fut.result()
            except Exception as e:  # noqa: BLE001 — one job never kills the run
                j = futs[fut]
                print(f"[job FAIL] {j[0]} @{j[3]}: {e}")
                done += 1
                continue
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}")
            if samples is None:
                continue
            buckets.setdefault(split, []).extend(samples)

    out = root / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    for split, samples in buckets.items():
        labeled = [s for s in samples if s["label_safe"] != -1]
        with open(out / f"{split}.pkl", "wb") as f:
            pickle.dump(samples, f)
        n_unsafe = sum(1 for s in labeled if s["label_safe"] == 0)
        print(f"{split}: {len(samples)} samples "
              f"({len(labeled)} labeled, {n_unsafe} unsafe)")
    return buckets


def load_split(root: Path, split: str):
    with open(Path(root) / "dataset" / f"{split}.pkl", "rb") as f:
        return pickle.load(f)
