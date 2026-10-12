"""RTL design generator for CrossExplain.

Emits parameterized multi-clock Verilog designs containing clock-domain
crossings whose safety, hazard class, and location are known BY CONSTRUCTION.

Naming convention (the key to label survival through synthesis):
  every net belonging to crossing k is prefixed  cdc<k>_
  the capture flip-flop output nets are          cdc<k>_cap*  (stage-0 FFs)
Yosys `write_json` keeps public wire names as aliases in `netnames`, so the
labels can be recovered from the flattened, optimized netlist.

Crossing templates
------------------
SAFE:
  SYNC2      classic 2-flip-flop synchronizer
  SYNC3      3-flip-flop synchronizer (high-MTBF variant)
  MUX_HOLD   qualifier-based bus crossing: control toggles through a 2FF
             sync, the data bus is captured through a recirculation mux.
             (Structurally has comb-before-FF -> classic pattern-matcher
             false positive; behaviourally safe because data is held.)
  GRAY2      2-bit Gray-coded counter, each bit through its own 2FF sync.

UNSAFE:
  NO_SYNC          async signal enters combinational logic that fans out to
                   multiple destination FFs (divergence before sync)
  SINGLE_FF        one sync stage only, its output consumed immediately
  COMB_BEFORE_SYNC XOR of two source-domain FFs (glitchy) into a 2FF sync
  RECONV           one source bit through two parallel 2FF chains, XORed in
                   the destination domain (reconvergence: skew-sensitive)
  BUS_NOSYNC       multi-bit binary counter, each bit 2FF-synced separately
                   then recombined as a word (per-bit looks safe -> classic
                   pattern-matcher false NEGATIVE; word is incoherent)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

SAFE_TEMPLATES = ["SYNC2", "SYNC3", "MUX_HOLD", "GRAY2", "SYNC2_DEEPSRC"]
UNSAFE_TEMPLATES = ["NO_SYNC", "SINGLE_FF", "COMB_BEFORE_SYNC", "RECONV",
                    "BUS_NOSYNC", "MUX_UNSYNC_QUAL", "SYNC2_LEAK"]
ALL_TEMPLATES = SAFE_TEMPLATES + UNSAFE_TEMPLATES

# hazard classes for the type head (index 0 reserved for SAFE in reports)
HAZARD_CLASSES = ["NO_SYNC", "SINGLE_FF", "COMB_BEFORE_SYNC", "RECONV", "BUS_NOSYNC"]

# every template maps to a type-head class; near-miss templates share the
# class of the failure mode they actually produce
TEMPLATE_CLASS = {
    "NO_SYNC": "NO_SYNC", "SINGLE_FF": "SINGLE_FF",
    "COMB_BEFORE_SYNC": "COMB_BEFORE_SYNC", "RECONV": "RECONV",
    "BUS_NOSYNC": "BUS_NOSYNC",
    "MUX_UNSYNC_QUAL": "NO_SYNC",     # unsynchronized qualifier
    "SYNC2_LEAK": "SINGLE_FF",        # stage-0 output consumed early
}


@dataclass
class Crossing:
    cid: int
    template: str
    safe: bool
    src_clk: str
    dst_clk: str
    cap_nets: list = field(default_factory=list)   # stage-0 capture FF output nets
    all_prefix: str = ""                            # net prefix for this crossing


@dataclass
class DesignSpec:
    name: str
    n_clocks: int
    crossings: list = field(default_factory=list)
    verilog: str = ""

    def ground_truth(self):
        return {
            "design": self.name,
            "clocks": [f"clk{i}" for i in range(self.n_clocks)],
            "crossings": [asdict(c) for c in self.crossings],
        }


class RTLGenerator:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ fill
    def _filler(self, dom: int, width: int) -> str:
        """Per-domain filler logic: an LFSR and a counter (varies the graphs)."""
        w = width
        taps = {4: "f[3]^f[2]", 5: "f[4]^f[2]", 6: "f[5]^f[4]",
                7: "f[6]^f[5]", 8: "f[7]^f[5]^f[4]^f[3]"}[w]
        t = taps.replace("f[", f"fill{dom}_lfsr[")
        return f"""
  // ---- filler logic, domain {dom} ----
  reg [{w-1}:0] fill{dom}_lfsr = {w}'d1;
  reg [3:0]     fill{dom}_cnt  = 4'd0;
  always @(posedge clk{dom}) begin
    fill{dom}_lfsr <= {{fill{dom}_lfsr[{w-2}:0], {t} ^ din[{dom % 8}]}};
    fill{dom}_cnt  <= fill{dom}_cnt + 4'd1;
  end
"""

    def _src_bit(self, dom: int) -> str:
        """An expression producing a 1-bit source-domain signal."""
        choices = [
            f"fill{dom}_lfsr[0]",
            f"fill{dom}_lfsr[2]",
            f"fill{dom}_cnt[1]",
            f"fill{dom}_cnt[3]",
        ]
        return self.rng.choice(choices)

    # ------------------------------------------------------- crossing bodies
    def _sync2(self, k, a, b):
        return f"""
  // crossing {k}: SYNC2 (safe) clk{a} -> clk{b}
  reg cdc{k}_src = 1'b0;
  always @(posedge clk{a}) cdc{k}_src <= {self._src_bit(a)};
  reg cdc{k}_cap = 1'b0, cdc{k}_s1 = 1'b0;
  always @(posedge clk{b}) begin
    cdc{k}_cap <= cdc{k}_src;
    cdc{k}_s1  <= cdc{k}_cap;
  end
  wire cdc{k}_out = cdc{k}_s1;
""", [f"cdc{k}_cap"]

    def _sync3(self, k, a, b):
        return f"""
  // crossing {k}: SYNC3 (safe) clk{a} -> clk{b}
  reg cdc{k}_src = 1'b0;
  always @(posedge clk{a}) cdc{k}_src <= {self._src_bit(a)};
  reg cdc{k}_cap = 1'b0, cdc{k}_s1 = 1'b0, cdc{k}_s2 = 1'b0;
  always @(posedge clk{b}) begin
    cdc{k}_cap <= cdc{k}_src;
    cdc{k}_s1  <= cdc{k}_cap;
    cdc{k}_s2  <= cdc{k}_s1;
  end
  wire cdc{k}_out = cdc{k}_s2;
""", [f"cdc{k}_cap"]

    def _mux_hold(self, k, a, b):
        # randomized data width and qualifier sync depth so no fixed size
        # signature separates this SAFE template from MUX_UNSYNC_QUAL
        w = self.rng.randint(3, 8)
        depth = self.rng.randint(2, 3)   # sync stages on the qualifier
        stages = [f"cdc{k}_qs{i}" for i in range(depth + 1)]
        decl = ", ".join(f"{s} = 1'b0" for s in stages)
        chain = "\n".join(
            f"    {stages[i]} <= {'cdc%d_req' % k if i == 0 else stages[i-1]};"
            for i in range(depth + 1))
        return f"""
  // crossing {k}: MUX_HOLD (safe qualifier-based bus, w={w}, depth={depth}) clk{a} -> clk{b}
  reg [{w-1}:0] cdc{k}_data = {w}'d0;
  reg cdc{k}_req = 1'b0;
  always @(posedge clk{a}) begin
    if (fill{a}_cnt == 4'd0) begin
      cdc{k}_data <= cdc{k}_data + {w}'d3;
      cdc{k}_req  <= ~cdc{k}_req;
    end
  end
  reg {decl};
  reg [{w-1}:0] cdc{k}_capb = {w}'d0;
  wire cdc{k}_load = {stages[-2]} ^ {stages[-1]};
  always @(posedge clk{b}) begin
{chain}
    cdc{k}_capb <= cdc{k}_load ? cdc{k}_data : cdc{k}_capb;
  end
  wire cdc{k}_out = ^cdc{k}_capb;
""", [f"cdc{k}_qs0", f"cdc{k}_capb"]

    def _gray2(self, k, a, b):
        w = self.rng.randint(2, 4)
        return f"""
  // crossing {k}: GRAY2 (safe gray-coded counter, w={w}) clk{a} -> clk{b}
  reg [{w-1}:0] cdc{k}_bin = {w}'d0;
  reg [{w-1}:0] cdc{k}_gray = {w}'d0;
  always @(posedge clk{a}) begin
    cdc{k}_bin  <= cdc{k}_bin + {w}'d1;
    cdc{k}_gray <= (cdc{k}_bin >> 1) ^ cdc{k}_bin;
  end
  reg [{w-1}:0] cdc{k}_cap = {w}'d0, cdc{k}_s1 = {w}'d0;
  always @(posedge clk{b}) begin
    cdc{k}_cap <= cdc{k}_gray;
    cdc{k}_s1  <= cdc{k}_cap;
  end
  wire cdc{k}_out = ^cdc{k}_s1;
""", [f"cdc{k}_cap"]

    def _no_sync(self, k, a, b):
        return f"""
  // crossing {k}: NO_SYNC (UNSAFE: async into comb, divergent fanout) clk{a} -> clk{b}
  reg cdc{k}_src = 1'b0;
  always @(posedge clk{a}) cdc{k}_src <= {self._src_bit(a)};
  wire cdc{k}_comb = cdc{k}_src & fill{b}_cnt[0];
  reg cdc{k}_cap = 1'b0, cdc{k}_capx = 1'b0;
  always @(posedge clk{b}) begin
    cdc{k}_cap  <= cdc{k}_comb;
    cdc{k}_capx <= cdc{k}_comb ^ fill{b}_lfsr[1];
  end
  wire cdc{k}_out = cdc{k}_cap ^ cdc{k}_capx;
""", [f"cdc{k}_cap", f"cdc{k}_capx"]

    def _single_ff(self, k, a, b):
        return f"""
  // crossing {k}: SINGLE_FF (UNSAFE: one stage, used immediately) clk{a} -> clk{b}
  reg cdc{k}_src = 1'b0;
  always @(posedge clk{a}) cdc{k}_src <= {self._src_bit(a)};
  reg cdc{k}_cap = 1'b0;
  reg cdc{k}_use0 = 1'b0, cdc{k}_use1 = 1'b0;
  always @(posedge clk{b}) begin
    cdc{k}_cap  <= cdc{k}_src;
    cdc{k}_use0 <= cdc{k}_cap & fill{b}_cnt[2];
    cdc{k}_use1 <= cdc{k}_cap | fill{b}_lfsr[0];
  end
  wire cdc{k}_out = cdc{k}_use0 ^ cdc{k}_use1;
""", [f"cdc{k}_cap"]

    def _comb_before_sync(self, k, a, b):
        # randomized comb-cloud depth (1..3 gates) so the hazard does not
        # have a fixed size signature
        depth = self.rng.randint(1, 3)
        ops = [self.rng.choice(["^", "&", "|"]) for _ in range(depth)]
        cloud, prev = [], f"cdc{k}_srcA"
        for d, op in enumerate(ops):
            other = f"cdc{k}_srcB" if d == 0 else f"fill{a}_cnt[{d}]"
            cloud.append(f"  wire cdc{k}_g{d} = {prev} {op} {other};")
            prev = f"cdc{k}_g{d}"
        cloud_v = "\n".join(cloud)
        return f"""
  // crossing {k}: COMB_BEFORE_SYNC (UNSAFE: glitchy comb cloud, depth {depth}) clk{a} -> clk{b}
  reg cdc{k}_srcA = 1'b0, cdc{k}_srcB = 1'b0;
  always @(posedge clk{a}) begin
    cdc{k}_srcA <= {self._src_bit(a)};
    cdc{k}_srcB <= fill{a}_cnt[0];
  end
{cloud_v}
  reg cdc{k}_cap = 1'b0, cdc{k}_s1 = 1'b0;
  always @(posedge clk{b}) begin
    cdc{k}_cap <= {prev};
    cdc{k}_s1  <= cdc{k}_cap;
  end
  wire cdc{k}_out = cdc{k}_s1;
""", [f"cdc{k}_cap"]

    def _sync2_deepsrc(self, k, a, b):
        # SAFE hard-positive: a deep comb cloud sits in the SOURCE domain,
        # but it is registered before the crossing — flat feature counters
        # see "lots of comb near the crossing" and get confused; the graph
        # sees that the cloud is upstream of the source FF.
        depth = self.rng.randint(2, 4)
        cloud, prev = [], self._src_bit(a)
        for d in range(depth):
            op = self.rng.choice(["^", "&", "|"])
            other = f"fill{a}_lfsr[{d % 4}]" if d % 2 else f"fill{a}_cnt[{d % 4}]"
            cloud.append(f"  wire cdc{k}_p{d} = {prev} {op} {other};")
            prev = f"cdc{k}_p{d}"
        cloud_v = "\n".join(cloud)
        return f"""
  // crossing {k}: SYNC2_DEEPSRC (safe: deep src-domain cloud, registered) clk{a} -> clk{b}
{cloud_v}
  reg cdc{k}_src = 1'b0;
  always @(posedge clk{a}) cdc{k}_src <= {prev};
  reg cdc{k}_cap = 1'b0, cdc{k}_s1 = 1'b0;
  always @(posedge clk{b}) begin
    cdc{k}_cap <= cdc{k}_src;
    cdc{k}_s1  <= cdc{k}_cap;
  end
  wire cdc{k}_out = cdc{k}_s1;
""", [f"cdc{k}_cap"]

    def _mux_unsync_qual(self, k, a, b):
        # UNSAFE hard-negative: the MUX_HOLD structure except the load
        # qualifier taps an UNSETTLED point (raw toggle, or the metastable
        # first stage of a decoy chain). Node counts overlap the safe
        # MUX_HOLD range — only the qualifier PATH distinguishes them.
        w = self.rng.randint(3, 8)
        decoy = self.rng.randint(0, 2)   # decoy sync stages (never used)
        stages = [f"cdc{k}_qs{i}" for i in range(decoy + 1)]
        decl = ", ".join(f"{s} = 1'b0" for s in stages)
        chain = "\n".join(
            f"    {stages[i]} <= {'cdc%d_req' % k if i == 0 else stages[i-1]};"
            for i in range(decoy + 1))
        # always tap the raw toggle (clean NO_SYNC semantics); the decoy
        # sync chain exists purely as confusing context
        tap = f"cdc{k}_req"
        return f"""
  // crossing {k}: MUX_UNSYNC_QUAL (UNSAFE: qualifier taps {tap}, w={w}) clk{a} -> clk{b}
  reg [{w-1}:0] cdc{k}_data = {w}'d0;
  reg cdc{k}_req = 1'b0;
  always @(posedge clk{a}) begin
    if (fill{a}_cnt == 4'd0) begin
      cdc{k}_data <= cdc{k}_data + {w}'d3;
      cdc{k}_req  <= ~cdc{k}_req;
    end
  end
  reg {decl};
  reg cdc{k}_lastq = 1'b0;
  reg [{w-1}:0] cdc{k}_capb = {w}'d0;
  wire cdc{k}_load = {tap} ^ cdc{k}_lastq;
  always @(posedge clk{b}) begin
{chain}
    cdc{k}_lastq <= {tap};
    cdc{k}_capb  <= cdc{k}_load ? cdc{k}_data : cdc{k}_capb;
  end
  wire cdc{k}_out = ^cdc{k}_capb ^ {stages[-1]};
""", [f"cdc{k}_lastq", f"cdc{k}_capb", f"cdc{k}_qs0"]

    def _sync2_leak(self, k, a, b):
        # UNSAFE hard-negative: a textbook 2FF chain exists, but the
        # metastable stage-0 output ALSO feeds other destination logic.
        return f"""
  // crossing {k}: SYNC2_LEAK (UNSAFE: stage-0 output leaks into logic) clk{a} -> clk{b}
  reg cdc{k}_src = 1'b0;
  always @(posedge clk{a}) cdc{k}_src <= {self._src_bit(a)};
  reg cdc{k}_cap = 1'b0, cdc{k}_s1 = 1'b0;
  reg cdc{k}_leak0 = 1'b0, cdc{k}_leak1 = 1'b0;
  always @(posedge clk{b}) begin
    cdc{k}_cap   <= cdc{k}_src;
    cdc{k}_s1    <= cdc{k}_cap;
    cdc{k}_leak0 <= cdc{k}_cap & fill{b}_cnt[1];
    cdc{k}_leak1 <= cdc{k}_cap | fill{b}_lfsr[2];
  end
  wire cdc{k}_out = cdc{k}_s1 ^ cdc{k}_leak0 ^ cdc{k}_leak1;
""", [f"cdc{k}_cap"]

    def _reconv(self, k, a, b):
        # NOTE: the second chain syncs the INVERTED signal so the two FF
        # chains are not logically identical — otherwise opt_merge fuses
        # them and the reconvergence structure is legitimately optimized
        # away (verified with yosys).  The hazard is unchanged: one source
        # value reaches destination logic through two independently-
        # latencied synchronizers.
        return f"""
  // crossing {k}: RECONV (UNSAFE: parallel syncs reconverge) clk{a} -> clk{b}
  reg cdc{k}_src = 1'b0;
  always @(posedge clk{a}) cdc{k}_src <= {self._src_bit(a)};
  reg cdc{k}_cap = 1'b0, cdc{k}_p0s1 = 1'b0;
  reg cdc{k}_capq = 1'b1, cdc{k}_p1s1 = 1'b1;
  always @(posedge clk{b}) begin
    cdc{k}_cap  <= cdc{k}_src;    cdc{k}_p0s1 <= cdc{k}_cap;
    cdc{k}_capq <= ~cdc{k}_src;   cdc{k}_p1s1 <= cdc{k}_capq;
  end
  reg cdc{k}_recon = 1'b0;
  always @(posedge clk{b}) cdc{k}_recon <= cdc{k}_p0s1 ~^ cdc{k}_p1s1;
  wire cdc{k}_out = cdc{k}_recon;
""", [f"cdc{k}_cap", f"cdc{k}_capq"]

    def _bus_nosync(self, k, a, b):
        w = self.rng.randint(3, 6)
        return f"""
  // crossing {k}: BUS_NOSYNC (UNSAFE: binary bus w={w}, per-bit syncs) clk{a} -> clk{b}
  reg [{w-1}:0] cdc{k}_bus = {w}'d0;
  always @(posedge clk{a}) cdc{k}_bus <= cdc{k}_bus + {w}'d1;
  reg [{w-1}:0] cdc{k}_cap = {w}'d0, cdc{k}_s1 = {w}'d0;
  always @(posedge clk{b}) begin
    cdc{k}_cap <= cdc{k}_bus;
    cdc{k}_s1  <= cdc{k}_cap;
  end
  reg [{w-1}:0] cdc{k}_word = {w}'d0;
  always @(posedge clk{b}) cdc{k}_word <= cdc{k}_word + cdc{k}_s1;
  wire cdc{k}_out = ^cdc{k}_word;
""", [f"cdc{k}_cap"]

    _BODY = {
        "SYNC2": _sync2, "SYNC3": _sync3, "MUX_HOLD": _mux_hold, "GRAY2": _gray2,
        "SYNC2_DEEPSRC": _sync2_deepsrc,
        "NO_SYNC": _no_sync, "SINGLE_FF": _single_ff,
        "COMB_BEFORE_SYNC": _comb_before_sync, "RECONV": _reconv,
        "BUS_NOSYNC": _bus_nosync, "MUX_UNSYNC_QUAL": _mux_unsync_qual,
        "SYNC2_LEAK": _sync2_leak,
    }

    # ---------------------------------------------------------------- design
    def generate(self, name: str, n_clocks: int = 2, n_crossings: int = 4,
                 templates: list | None = None) -> DesignSpec:
        """Generate one design. `templates` restricts the crossing styles
        (used for family-held-out splits)."""
        rng = self.rng
        templates = templates or ALL_TEMPLATES
        spec = DesignSpec(name=name, n_clocks=n_clocks)

        body = []
        fill_width = rng.choice([4, 5, 6, 7, 8])
        for d in range(n_clocks):
            body.append(self._filler(d, fill_width))

        outs = [f"fill{d}_lfsr[0]" for d in range(n_clocks)]
        for k in range(n_crossings):
            a, b = rng.sample(range(n_clocks), 2)
            tpl = rng.choice(templates)
            code, cap_nets = self._BODY[tpl](self, k, a, b)
            body.append(code)
            outs.append(f"cdc{k}_out")
            spec.crossings.append(Crossing(
                cid=k, template=tpl, safe=tpl in SAFE_TEMPLATES,
                src_clk=f"clk{a}", dst_clk=f"clk{b}",
                cap_nets=cap_nets, all_prefix=f"cdc{k}_",
            ))

        clks = ", ".join(f"input clk{i}" for i in range(n_clocks))
        spec.verilog = (
            f"module top({clks}, input [7:0] din, output out);\n"
            + "".join(body)
            + f"\n  assign out = {' ^ '.join(outs)};\nendmodule\n"
        )
        return spec


def write_design(spec: DesignSpec, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{spec.name}.v").write_text(spec.verilog, newline="\n")
    (out_dir / f"{spec.name}.json").write_text(
        json.dumps(spec.ground_truth(), indent=1), newline="\n")


if __name__ == "__main__":
    gen = RTLGenerator(seed=42)
    spec = gen.generate("demo", n_clocks=2, n_crossings=4)
    print(spec.verilog)
    print(json.dumps(spec.ground_truth(), indent=1))
