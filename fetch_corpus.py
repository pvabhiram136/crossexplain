"""Clone the upstream RTL this corpus is built from, at the exact commits used.

The artifact does not redistribute third-party RTL. Four of these projects ship
a LICENSE and some of the smaller cores in our working tree carry none we could
find, so vendoring all of it into a release would be a licensing problem with no
benefit to anyone. Pinning commits is strictly better anyway: it makes the
corpus rebuildable AND states exactly which revision the paper's numbers came
from, which a vendored snapshot does not.

    python fetch_corpus.py [--dest journal/corpus]

Requires git. Each repository is cloned shallow at its pinned commit. Re-running
is safe: an existing checkout at the right commit is left alone, and one at the
wrong commit is reported rather than silently reset.

After this completes, rebuild the corpus with:

    python journal/scripts/j12_corpus_triage.py     # first-pass candidates
    python journal/scripts/j40_expand_corpus.py     # second pass (+ patches)
    python journal/scripts/j46_merge_corpus.py      # merge into the triage

Note that j40 applies the documented source patches in journal/corpus/patches/.
Those are ours, they are small, and each carries the argument for why it cannot
affect an extracted crossing. Read that directory before trusting the result.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# name -> (url, commit). These are the revisions the paper's numbers were
# produced from; changing one invalidates the corpus and every number that
# depends on it.
REPOS = {
    "opentitan":        ("https://github.com/lowRISC/opentitan", "7827845"),
    "axi":              ("https://github.com/pulp-platform/axi", "da8793b"),
    "common_cells":     ("https://github.com/pulp-platform/common_cells", "63b7c50"),
    "verilog-axis":     ("https://github.com/alexforencich/verilog-axis", "48ff7a7"),
    "verilog-pcie":     ("https://github.com/alexforencich/verilog-pcie", "25156a9"),
    "wb2axip":          ("https://github.com/ZipCPU/wb2axip", "2e8d3bc"),
    "basejump_stl":     ("https://github.com/bespoke-silicon-group/basejump_stl", "b48037e"),
    "verilog-ethernet": ("https://github.com/alexforencich/verilog-ethernet", "77320a9"),
}


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd,
                          capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="journal/corpus",
                    help="where to place the checkouts")
    args = ap.parse_args()

    if git("--version").returncode != 0:
        print("git is required and was not found on PATH")
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    problems = []

    for name, (url, commit) in REPOS.items():
        path = dest / name
        if (path / ".git").exists():
            have = git("rev-parse", "--short", "HEAD", cwd=path).stdout.strip()
            if have.startswith(commit) or commit.startswith(have):
                print(f"  ok       {name:18s} already at {have}")
            else:
                print(f"  MISMATCH {name:18s} at {have}, expected {commit}")
                problems.append(
                    f"{name}: checkout is at {have}, the paper used {commit}")
            continue

        print(f"  cloning  {name:18s} {url} @ {commit}")
        # Fetch just the pinned commit where the server allows it, and fall
        # back to a full clone where it does not.
        path.mkdir(parents=True, exist_ok=True)
        git("init", "-q", cwd=path)
        git("remote", "add", "origin", url, cwd=path)
        r = git("fetch", "-q", "--depth", "1", "origin", commit, cwd=path)
        if r.returncode != 0:
            git("fetch", "-q", "origin", cwd=path)
        r = git("checkout", "-q", commit, cwd=path)
        if r.returncode != 0:
            print(f"           FAILED to check out {commit}: "
                  f"{r.stderr.strip()[:120]}")
            problems.append(f"{name}: could not check out {commit}")

    print()
    if problems:
        print("  Problems, and none of them should be ignored -- the paper's")
        print("  numbers belong to the pinned revisions:")
        for p in problems:
            print(f"    - {p}")
        return 1
    print(f"  all {len(REPOS)} repositories present at the pinned commits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
