"""Yosys synthesis driver (yowasp-yosys, pure pip — no native EDA install).

Synthesizes a Verilog design to a flattened gate-level JSON netlist at a
chosen optimization level. Higher levels mangle the RTL structure more —
this is the robustness axis of the CrossExplain experiments.

Levels:
  O0  proc + flatten only (RTL structure mostly intact, mapped to gates)
  O1  + opt (constant folding, cleanup)
  O2  + full synth pipeline (fsm, memory, techmap, opt rounds)
  O3  O2 + share + opt_merge -share_all + abc-style aggressive cleanup
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# yosys script per level; {vfile} and {jfile} get substituted
_SCRIPTS = {
    "O0": "read_verilog {vfile}; hierarchy -top top; proc; flatten; "
          "techmap; opt_clean; write_json {jfile}",
    "O1": "read_verilog {vfile}; hierarchy -top top; proc; flatten; "
          "opt; techmap; opt; write_json {jfile}",
    "O2": "read_verilog {vfile}; hierarchy -top top; "
          "synth -top top -flatten -noabc; write_json {jfile}",
    "O3": "read_verilog {vfile}; hierarchy -top top; "
          "synth -top top -flatten -noabc; share -aggressive; "
          "opt -full; opt_merge -share_all; opt -full; write_json {jfile}",
}

OPT_LEVELS = list(_SCRIPTS)


def synthesize(vfile: Path, jfile: Path, level: str = "O2",
               quiet: bool = True) -> bool:
    """Run yosys on `vfile`, writing JSON netlist to `jfile`.

    Runs in a subprocess (each yowasp wasm instance is single-use and this
    also gives us parallelism for dataset generation)."""
    vfile, jfile = Path(vfile), Path(jfile)
    jfile.parent.mkdir(parents=True, exist_ok=True)
    script = _SCRIPTS[level].format(vfile=vfile.name, jfile=str(jfile.resolve()).replace("\\", "/"))
    args = ["-q", "-p", script] if quiet else ["-p", script]
    code = (
        "import sys, yowasp_yosys\n"
        f"sys.exit(yowasp_yosys.run_yosys({args!r}))\n"
    )
    for attempt, tmo in enumerate((240, 420)):
        try:
            r = subprocess.run([sys.executable, "-c", code], cwd=vfile.parent,
                               capture_output=True, text=True, timeout=tmo)
        except subprocess.TimeoutExpired:
            print(f"[synth TIMEOUT {tmo}s] {vfile.name} @{level} "
                  f"(attempt {attempt + 1})")
            continue
        if r.returncode != 0:
            print(f"[synth FAIL] {vfile.name} @{level}\n"
                  f"{r.stdout[-800:]}\n{r.stderr[-800:]}")
            return False
        return jfile.exists()
    return False


if __name__ == "__main__":
    import json
    from rtl_generator import RTLGenerator, write_design

    out = Path(__file__).resolve().parent.parent / "data" / "smoke"
    gen = RTLGenerator(seed=1)
    spec = gen.generate("smoke0", n_clocks=2, n_crossings=4)
    write_design(spec, out)
    for lvl in OPT_LEVELS:
        ok = synthesize(out / "smoke0.v", out / f"smoke0.{lvl}.json", lvl)
        if ok:
            d = json.loads((out / f"smoke0.{lvl}.json").read_text())
            cells = d["modules"]["top"]["cells"]
            print(lvl, "OK", len(cells), "cells")
        else:
            print(lvl, "FAILED")
