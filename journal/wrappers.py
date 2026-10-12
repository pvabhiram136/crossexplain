"""P1 option 3: wrap real single-clock cores in two-clock harnesses.

Why this exists
---------------
Triage (FINDINGS_P1.md) showed that the CDC-bearing open RTL population is
almost entirely CDC *primitives*: median 43 flip-flops, mostly isolated
synchronizers with little surrounding logic. P0 finding (d) established that
the sim-to-real gap is driven by the opposite situation, safe crossings
embedded in wide busy logic (43.0 nodes against 29.1 in training) which the
model misreads as hazard evidence. A corpus made only of primitives would
satisfy the design count while making the benchmark easier.

Meanwhile `uart16550` (349 FFs), `i2c_master` (200) and `simple_spi` (68)
elaborate cleanly but are single-clock, so they contribute nothing. They are
exactly the busy context the corpus lacks.

What this module does
---------------------
Each of those cores is a Wishbone peripheral. In a real SoC a bus master in one
clock domain drives a peripheral clocked in another, so placing the Wishbone
side in a second domain is CDC that genuinely occurs with these cores rather
than an artificial construct.

Honesty boundary, which the paper must state
--------------------------------------------
The *core* is third-party and unmodified. The *crossing* is ours, created by
this wrapper, so its label is by construction exactly as in
`crossexplain/real_designs.py`. These designs are semi-synthetic: real
surrounding logic, authored crossing. They are not "found" CDC and must not be
counted as such. Their value is specific: they are the only way we have to test
finding (d), because they place a crossing of known safety inside hundreds of
flip-flops of real peripheral logic.

Net naming follows the `cdc<k>_` convention so the label-recovery path in
`netlist_graph.extract_samples` works unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

NL = "\n"


@dataclass
class CoreSpec:
    name: str
    top: str
    clk: str                      # core clock port
    rst: str                      # core reset port (active high)
    adr: tuple                    # (port, width)
    dat: tuple                    # (port, width)
    we: str
    stb: str
    cyc: str
    outs: list                    # outputs to XOR-reduce, (port, width)
    pin_in: str                   # an asynchronous device pin input
    extra_tie0: list = field(default_factory=list)
    extra_sel: tuple | None = None


CORES = {
    "uart16550": CoreSpec(
        name="uart16550", top="uart_top", clk="wb_clk_i", rst="wb_rst_i",
        adr=("wb_adr_i", 5), dat=("wb_dat_i", 32), we="wb_we_i",
        stb="wb_stb_i", cyc="wb_cyc_i", extra_sel=("wb_sel_i", 4),
        pin_in="srx_pad_i",
        extra_tie0=["cts_pad_i", "dsr_pad_i", "ri_pad_i", "dcd_pad_i"],
        outs=[("wb_dat_o", 32), ("wb_ack_o", 1), ("int_o", 1),
              ("stx_pad_o", 1), ("rts_pad_o", 1), ("dtr_pad_o", 1)]),
    "i2c_master": CoreSpec(
        name="i2c_master", top="i2c_master_top", clk="wb_clk_i",
        rst="wb_rst_i", adr=("wb_adr_i", 3), dat=("wb_dat_i", 8),
        we="wb_we_i", stb="wb_stb_i", cyc="wb_cyc_i", pin_in="scl_pad_i",
        extra_tie0=["arst_i", "sda_pad_i"],
        outs=[("wb_dat_o", 8), ("wb_ack_o", 1), ("wb_inta_o", 1),
              ("scl_pad_o", 1), ("scl_padoen_o", 1), ("sda_pad_o", 1),
              ("sda_padoen_o", 1)]),
    "simple_spi": CoreSpec(
        name="simple_spi", top="simple_spi_top", clk="clk_i", rst="rst_i",
        adr=("adr_i", 2), dat=("dat_i", 8), we="we_i",
        stb="stb_i", cyc="cyc_i", pin_in="miso_i",
        outs=[("dat_o", 8), ("ack_o", 1), ("inta_o", 1), ("sck_o", 1),
              ("mosi_o", 1)]),
}


def _style_bus_nosync(c: CoreSpec, k: int):
    """UNSAFE: a multi-bit Wishbone request crosses with no synchronization."""
    aw, dw = c.adr[1], c.dat[1]
    body = f"""
  // crossing {k}: BUS_NOSYNC (UNSAFE) bus domain -> core domain
  // The whole Wishbone request is registered in clk_bus and read directly by
  // the core in clk_sys. Multi-bit, no synchronizer, incoherent on capture.
  reg [{aw - 1}:0] cdc{k}_adr = {aw}'d0;
  reg [{dw - 1}:0] cdc{k}_dat = {dw}'d0;
  reg cdc{k}_we = 1'b0, cdc{k}_stb = 1'b0, cdc{k}_cyc = 1'b0;
  always @(posedge clk_bus) begin
    cdc{k}_adr <= cdc{k}_adr + {aw}'d1;
    cdc{k}_dat <= {{cdc{k}_dat[{dw - 2}:0],
                    cdc{k}_dat[{dw - 1}] ^ cdc{k}_dat[{dw - 2}] ^ din_i[0]}};
    cdc{k}_we  <= ~cdc{k}_we;
    cdc{k}_stb <= drive_i;
    cdc{k}_cyc <= drive_i;
  end
"""
    pm = {c.adr[0]: f"cdc{k}_adr", c.dat[0]: f"cdc{k}_dat",
          c.we: f"cdc{k}_we", c.stb: f"cdc{k}_stb", c.cyc: f"cdc{k}_cyc"}
    return body, pm, [dict(cid=k, template="BUS_NOSYNC", safe=False)]


def _style_sync2_held(c: CoreSpec, k: int):
    """SAFE: control through a two-flop synchronizer, address and data held
    stable while the qualifier is in flight (qualifier-held bus)."""
    aw, dw = c.adr[1], c.dat[1]
    body = f"""
  // crossing {k}: MUX_HOLD (safe) bus domain -> core domain
  // stb/cyc cross through a two-flop synchronizer; adr/dat are updated only
  // while the request is idle, so they are stable whenever the core samples.
  reg [{aw - 1}:0] cdc{k}_adr = {aw}'d0;
  reg [{dw - 1}:0] cdc{k}_dat = {dw}'d0;
  reg cdc{k}_req = 1'b0;
  always @(posedge clk_bus) begin
    if (!cdc{k}_req) begin
      cdc{k}_adr <= cdc{k}_adr + {aw}'d1;
      cdc{k}_dat <= {{cdc{k}_dat[{dw - 2}:0],
                      cdc{k}_dat[{dw - 1}] ^ cdc{k}_dat[{dw - 2}] ^ din_i[0]}};
    end
    cdc{k}_req <= drive_i;
  end
  reg cdc{k}_cap = 1'b0, cdc{k}_s1 = 1'b0;
  always @(posedge clk_sys) begin
    cdc{k}_cap <= cdc{k}_req;
    cdc{k}_s1  <= cdc{k}_cap;
  end
  wire cdc{k}_qual = cdc{k}_s1;
"""
    pm = {c.adr[0]: f"cdc{k}_adr", c.dat[0]: f"cdc{k}_dat",
          c.we: f"cdc{k}_qual", c.stb: f"cdc{k}_qual", c.cyc: f"cdc{k}_qual"}
    return body, pm, [dict(cid=k, template="MUX_HOLD", safe=True)]


def _style_pin_sync2(c: CoreSpec, k: int):
    """SAFE: the asynchronous device pin is synchronized before the core."""
    body = f"""
  // crossing {k}: SYNC2 (safe) device pin resynchronized into the core domain
  // The source must hold real state. An earlier version registered drive_i
  // directly, and from O1 onward the optimizer removed that redundant flop and
  // fed the synchronizer from the port itself, which is not a crossing at all
  // and silently deleted the sample at three of the four levels.
  reg [3:0] cdc{k}_lfsr = 4'd1;
  reg cdc{k}_src = 1'b0;
  always @(posedge clk_bus) begin
    cdc{k}_lfsr <= {{cdc{k}_lfsr[2:0], cdc{k}_lfsr[3] ^ cdc{k}_lfsr[2] ^ din_i[2]}};
    cdc{k}_src  <= cdc{k}_lfsr[0] ^ drive_i;
  end
  reg cdc{k}_cap = 1'b0, cdc{k}_s1 = 1'b0;
  always @(posedge clk_sys) begin
    cdc{k}_cap <= cdc{k}_src;
    cdc{k}_s1  <= cdc{k}_cap;
  end
  wire cdc{k}_pin = cdc{k}_s1;
"""
    return body, {c.pin_in: f"cdc{k}_pin"}, [
        dict(cid=k, template="SYNC2", safe=True)]


def _style_pin_nosync(c: CoreSpec, k: int):
    """UNSAFE: the pin enters core logic through a glitch-prone combinational
    cloud with no synchronizer."""
    body = f"""
  // crossing {k}: NO_SYNC (UNSAFE) device pin into the core with no sync
  // Two INDEPENDENT source-domain registers converge through combinational
  // logic with no synchronizer. An earlier version used drive_i and ~drive_i,
  // whose XOR is the constant 1 and was folded away by synthesis, silently
  // deleting the crossing.
  reg [3:0] cdc{k}_cnt = 4'd0;
  reg cdc{k}_src = 1'b0, cdc{k}_srcb = 1'b0;
  always @(posedge clk_bus) begin
    cdc{k}_cnt  <= cdc{k}_cnt + 4'd1;
    cdc{k}_src  <= din_i[1];
    cdc{k}_srcb <= cdc{k}_cnt[3];
  end
  wire cdc{k}_pin = cdc{k}_src ^ cdc{k}_srcb;
"""
    return body, {c.pin_in: f"cdc{k}_pin"}, [
        dict(cid=k, template="NO_SYNC", safe=False)]


def _style_sync2_deep(c, k, width=32):
    """SAFE, SYNC2 idiom, LARGE subgraph.

    Breaks the size/idiom confound from the first wrapped run, where every
    SYNC2 sample was exactly 14 nodes and every MUX_HOLD was 72 or more, so the
    two explanations could not be separated.

    Same two-flop synchronizer as `pin_sync2`, but the source register sits at
    the output of a wide busy datapath in the source domain. The cloud is
    REGISTERED before the crossing, so the crossing is unambiguously safe (this
    is the generator's SYNC2_DEEPSRC argument), while the six-hop backward cone
    now pulls in the whole datapath and the subgraph grows.
    """
    body = f"""
  // crossing {k}: SYNC2 deep-source (safe, LARGE) width={width}
  reg [{width - 1}:0] cdc{k}_bulk = {width}'d1;
  integer cdc{k}_i;
  always @(posedge clk_bus) begin
    cdc{k}_bulk <= {{cdc{k}_bulk[{width - 2}:0],
                     cdc{k}_bulk[{width - 1}] ^ cdc{k}_bulk[{width // 2}]
                     ^ din_i[3]}};
  end
  wire cdc{k}_pre = ^cdc{k}_bulk;
  reg cdc{k}_src = 1'b0;
  always @(posedge clk_bus) cdc{k}_src <= cdc{k}_pre;   // registered: safe
  reg cdc{k}_cap = 1'b0, cdc{k}_s1 = 1'b0;
  always @(posedge clk_sys) begin
    cdc{k}_cap <= cdc{k}_src;
    cdc{k}_s1  <= cdc{k}_cap;
  end
  wire cdc{k}_pin = cdc{k}_s1;
"""
    return body, {c.pin_in: f"cdc{k}_pin"}, [
        dict(cid=k, template="SYNC2", safe=True)]


def _style_mux_hold_narrow(c, k, width=4):
    """SAFE, MUX_HOLD idiom, SMALL subgraph.

    The mirror image of `_style_sync2_deep`. The qualifier-held register lives
    in the wrapper rather than inside the core, so the crossing's capture
    flip-flops are ours and few, and the subgraph stays small while the idiom
    remains the qualifier-held bus that the model fails on at large size.
    """
    body = f"""
  // crossing {k}: MUX_HOLD narrow (safe, SMALL) width={width}
  reg [{width - 1}:0] cdc{k}_data = {width}'d0;
  reg cdc{k}_req = 1'b0;
  always @(posedge clk_bus) begin
    if (!cdc{k}_req)
      cdc{k}_data <= {{cdc{k}_data[{width - 2}:0],
                       cdc{k}_data[{width - 1}] ^ din_i[4]}};
    cdc{k}_req <= drive_i;
  end
  reg cdc{k}_cap = 1'b0, cdc{k}_s1 = 1'b0;
  always @(posedge clk_sys) begin
    cdc{k}_cap <= cdc{k}_req;
    cdc{k}_s1  <= cdc{k}_cap;
  end
  wire cdc{k}_load = cdc{k}_cap ^ cdc{k}_s1;
  reg [{width - 1}:0] cdc{k}_hold = {width}'d0;
  always @(posedge clk_sys)
    cdc{k}_hold <= cdc{k}_load ? cdc{k}_data : cdc{k}_hold;
  wire cdc{k}_pin = ^cdc{k}_hold;
"""
    return body, {c.pin_in: f"cdc{k}_pin"}, [
        dict(cid=k, template="MUX_HOLD", safe=True)]


STYLES = {
    "bus_nosync": _style_bus_nosync,
    "sync2_held": _style_sync2_held,
    "pin_sync2": _style_pin_sync2,
    "pin_nosync": _style_pin_nosync,
    "sync2_deep": _style_sync2_deep,
    "mux_hold_narrow": _style_mux_hold_narrow,
}


def make_wrapper(core: str, styles: list, module_name: str | None = None):
    """Build a two-clock wrapper around a real single-clock core.

    Returns (verilog_text, ground_truth_dict).
    """
    c = CORES[core]
    module_name = module_name or ("w_" + core + "_" + "_".join(styles))

    bodies, pm, crossings = [], {}, []
    for k, s in enumerate(styles):
        b, m, cr = STYLES[s](c, k)
        bodies.append(b)
        for port, expr in m.items():
            if port in pm:
                raise ValueError("port " + port + " driven by two styles")
            pm[port] = expr
        crossings.extend(cr)

    driven = set(pm)
    conns = [f".{c.clk}(clk_sys)", f".{c.rst}(rst_i)"]
    for port, expr in pm.items():
        conns.append(f".{port}({expr})")
    for port, w in (c.adr, c.dat):
        if port not in driven:
            conns.append(f".{port}({w}'d0)")
    for port in (c.we, c.stb, c.cyc):
        if port not in driven:
            conns.append(f".{port}(1'b0)")
    if c.pin_in not in driven:
        conns.append(f".{c.pin_in}(1'b1)")
    if c.extra_sel:
        p, w = c.extra_sel
        conns.append(f".{p}({{{w}{{1'b1}}}})")
    for p in c.extra_tie0:
        conns.append(f".{p}(1'b0)")

    outw = []
    for p, w in c.outs:
        conns.append(f".{p}({p}_w)")
        outw.append(f"  wire [{w - 1}:0] {p}_w;" if w > 1
                    else f"  wire {p}_w;")
    reduce_expr = " ^ ".join(f"(^{p}_w)" for p, _ in c.outs)
    conn_txt = ("," + NL + "    ").join(conns)

    v = f"""// Two-clock wrapper around the unmodified {core} core.
// The core is third-party; the crossings are authored by journal/wrappers.py
// and labeled by construction. Semi-synthetic: real surrounding logic,
// authored crossing. See journal/FINDINGS_P1.md.
module {module_name}(
  input clk_sys,
  input clk_bus,
  input rst_i,
  input drive_i,
  input [7:0] din_i,
  output out
);
{NL.join(outw)}
{"".join(bodies)}
  {c.top} u_core (
    {conn_txt}
  );

  assign out = {reduce_expr};
endmodule
"""
    gt = {"design": module_name, "core": core,
          "clocks": ["clk_sys", "clk_bus"],
          "semi_synthetic": True,
          "crossings": [dict(cid=cr["cid"], template=cr["template"],
                             safe=cr["safe"], src_clk="clk_bus",
                             dst_clk="clk_sys", cap_nets=[],
                             all_prefix="cdc" + str(cr["cid"]) + "_")
                        for cr in crossings]}
    return v, gt


def extract_wrapped(jfile, gt, design, level):
    """Extract crossings from a wrapped core, keyed on the SOURCE side.

    The stock recovery path in `netlist_graph.group_crossings` keys on the
    `cdc<k>_` prefix of the *capture* flip-flop's own net. That works for the
    generator, whose destination register belongs to the crossing it names, but
    not here: in a wrapped core the destination registers are inside
    third-party RTL and cannot be renamed. Only the source side carries our
    prefix.

    So a capture flip-flop is attributed to crossing k when the flip-flops in
    its combinational fan-in cone carry the `cdc<k>_` prefix, and to the
    crossing contributing the most such sources when several qualify.
    """
    from crossexplain.netlist_graph import (parse_netlist, find_capture_ffs,
                                            extract_subgraph, node_features,
                                            CrossingSample, CDC_RE)
    from crossexplain.rtl_generator import HAZARD_CLASSES, TEMPLATE_CLASS

    nl = parse_netlist(jfile)
    caps = find_capture_ffs(nl)
    gtmap = {c["cid"]: c for c in gt["crossings"]}

    def cdc_ids_of(idx):
        ids = []
        for b in nl.cell_bits_out[idx]:
            for nm in nl.bit_names.get(b, []):
                m = CDC_RE.search(nm)
                if m:
                    ids.append(int(m.group(1)))
        m = CDC_RE.search(nl.cells[idx][0])
        if m:
            ids.append(int(m.group(1)))
        return ids

    groups = {}
    for f in caps:
        src_ffs, _ = nl.d_srcs[f]
        votes = {}
        for s in src_ffs:
            if nl.domain[s] == nl.domain[f]:
                continue                      # same-domain source, not a cross
            for cid in cdc_ids_of(s):
                votes[cid] = votes.get(cid, 0) + 1
        if not votes:
            continue
        cid = max(votes, key=lambda k: (votes[k], -k))
        groups.setdefault(cid, []).append(f)

    samples = []
    for cid, ffs in sorted(groups.items()):
        if cid not in gtmap:
            continue
        seeds = set(ffs)
        src_doms = set()
        for f in ffs:
            src_doms |= caps[f][0]
        dst = nl.domain[ffs[0]]
        nodes, edges, local = extract_subgraph(nl, sorted(ffs), cid=cid)
        feats = node_features(nl, nodes, seeds, src_doms, dst, edges)
        cone = set(ffs)
        for f in ffs:
            sf, cb = nl.d_srcs[f]
            cone |= sf | cb
        mask = [1 if g in cone else 0 for g in nodes]
        g = gtmap[cid]
        cls = TEMPLATE_CLASS.get(g["template"])
        samples.append(CrossingSample(
            design=design, level=level, cid=cid, nodes=nodes,
            node_feats=feats, edges=edges,
            seeds=[local[f] for f in sorted(ffs)], loc_mask=mask,
            label_safe=1 if g["safe"] else 0,
            label_class=HAZARD_CLASSES.index(cls) if cls else -1,
            template=g["template"],
            src_dom=min(src_doms) if src_doms else -1, dst_dom=dst))
    return samples, nl
