# Source patches applied before lowering

Third-party RTL enters this corpus **unmodified** wherever possible. That is the
whole basis of the corpus's claim: the crossings are somebody else's, written to
be correct, and not touched by us.

This directory is the documented exception. Each patch here exists because a
*tool* cannot read the source, not because the design is wrong, and each one is
applied by script at build time (`j40_expand_corpus.py`) rather than committed
into the checked-out tree. The patched region is recorded below along with the
argument that it cannot affect the extracted crossing.

A patch belongs here only if all three hold:

1. it works around a limitation of our toolchain, not a property of the design;
2. the patched code is provably not reachable from the crossing we extract;
3. it is small enough to read in full in this file.

Anything that fails one of those should leave the candidate excluded, with the
reason reported in Section V, which is what we do for the candidates that are
still blocked.

---

## `axi_pkg.patch` — pulp-platform/axi, `axi/src/axi_pkg.sv`

**Tool limitation.** sv2v v0.0.13 cannot parse a `typedef` declared inside a
function body. `axi_pkg.sv` declares one in `beat_upper_byte`:

```systemverilog
  function automatic shortint unsigned
  beat_upper_byte(largest_addr_t addr, size_t size, len_t len, burst_t burst,
      shortint unsigned strobe_width, shortint unsigned i_beat);
      typedef shortint unsigned SU;          // <-- sv2v 0.0.13 stops here
```

sv2v reports `axi_pkg.sv:210:7: Parse error: missing expected 'endfunction'`
and lowers nothing, so the whole design is rejected.

**The patch.** The `typedef` is *hoisted* from the function body to package
scope. Nothing is deleted and no expression changes; `SU` resolves to the same
type in the same package, and `beat_upper_byte` is byte-for-byte identical apart
from the removed declaration line.

**Why it cannot affect the result.** `beat_upper_byte` computes the index of the
highest byte in an AXI beat (AXI spec A3-51). It is an address-arithmetic helper
used by protocol checkers and testbenches. It is not called by `axi_cdc`,
`axi_cdc_src`, `axi_cdc_dst` or `cc_cdc_fifo_gray` — the modules from which the
crossings are extracted — and it instantiates no cells, so it contributes
nothing to the synthesized netlist the graph is built from.

**Separate from the patch:** `axi_cdc.sv` also defines `axi_cdc_intf`, an
interface wrapper, so `axi/src/axi_intf.sv` must be on the source list. That is
a missing-source fix in our own build script, not a change to anybody's code.
