"""Hand-written multi-clock designs for the out-of-distribution test set.

These follow classic textbook structures (Cummings-style async FIFO, UART
with a separate baud domain, SPI slave, four-phase handshake) rather than
the generator's templates. Good and deliberately-broken variants included.
The cdc<k>_ naming convention marks each crossing for label recovery.
"""
from __future__ import annotations

from .rtl_generator import DesignSpec, Crossing

_DESIGNS = []


def _add(name, n_clocks, verilog, crossings):
    spec = DesignSpec(name=name, n_clocks=n_clocks)
    spec.verilog = verilog
    spec.crossings = crossings
    _DESIGNS.append(spec)


# ------------------------------------------------------------- async FIFO
_add("r_afifo", 2, """
module top(input clk0, input clk1, input [7:0] din, output out);
  // 4-deep async FIFO: clk0 = write side, clk1 = read side
  reg [7:0] mem [0:3];
  reg [2:0] wptr_bin = 3'd0, rptr_bin = 3'd0;
  reg [2:0] cdc0_wgray = 3'd0;   // write pointer, gray (crosses to clk1)
  reg [2:0] cdc1_rgray = 3'd0;   // read pointer, gray (crosses to clk0)
  wire wput = din[0];
  wire rget = din[1];

  // ---- write domain ----
  reg [2:0] cdc1_cap = 3'd0, cdc1_s1 = 3'd0;   // rptr sync into clk0
  wire full = (cdc0_wgray == {~cdc1_s1[2:1], cdc1_s1[0]});
  always @(posedge clk0) begin
    cdc1_cap <= cdc1_rgray;
    cdc1_s1  <= cdc1_cap;
    if (wput && !full) begin
      mem[wptr_bin[1:0]] <= din;
      wptr_bin  <= wptr_bin + 3'd1;
      cdc0_wgray <= ((wptr_bin + 3'd1) >> 1) ^ (wptr_bin + 3'd1);
    end
  end

  // ---- read domain ----
  reg [2:0] cdc0_cap = 3'd0, cdc0_s1 = 3'd0;   // wptr sync into clk1
  reg [7:0] rdata = 8'd0;
  wire empty = (cdc0_s1 == ((rptr_bin >> 1) ^ rptr_bin));
  always @(posedge clk1) begin
    cdc0_cap <= cdc0_wgray;
    cdc0_s1  <= cdc0_cap;
    if (rget && !empty) begin
      rdata    <= mem[rptr_bin[1:0]];
      rptr_bin <= rptr_bin + 3'd1;
      cdc1_rgray <= ((rptr_bin + 3'd1) >> 1) ^ (rptr_bin + 3'd1);
    end
  end
  assign out = ^rdata ^ full ^ empty;
endmodule
""", [
    Crossing(cid=0, template="GRAY2", safe=True, src_clk="clk0", dst_clk="clk1",
             cap_nets=["cdc0_cap"], all_prefix="cdc0_"),
    Crossing(cid=1, template="GRAY2", safe=True, src_clk="clk1", dst_clk="clk0",
             cap_nets=["cdc1_cap"], all_prefix="cdc1_"),
])

# ---------------------------------------------------------------- UART ok
_add("r_uart_good", 2, """
module top(input clk0, input clk1, input [7:0] din, output out);
  // clk0 = system, clk1 = baud. Data handed over with a toggle qualifier.
  reg [7:0] cdc0_data = 8'd0;
  reg cdc0_req = 1'b0;
  reg [3:0] sys_cnt = 4'd0;
  always @(posedge clk0) begin
    sys_cnt <= sys_cnt + 4'd1;
    if (sys_cnt == 4'd0) begin
      cdc0_data <= din;
      cdc0_req  <= ~cdc0_req;
    end
  end
  // baud domain: sync the toggle, then load data through recirculating mux
  reg cdc0_cap = 1'b0, cdc0_s1 = 1'b0, cdc0_s2 = 1'b0;
  reg [7:0] cdc0_hold = 8'd0;
  reg [9:0] tx_shift = 10'b1111111111;
  reg [3:0] tx_cnt = 4'd0;
  wire cdc0_load = cdc0_s1 ^ cdc0_s2;
  always @(posedge clk1) begin
    cdc0_cap <= cdc0_req;
    cdc0_s1  <= cdc0_cap;
    cdc0_s2  <= cdc0_s1;
    cdc0_hold <= cdc0_load ? cdc0_data : cdc0_hold;
    if (cdc0_load) begin
      tx_shift <= {1'b1, cdc0_hold, 1'b0};
      tx_cnt   <= 4'd10;
    end else if (tx_cnt != 4'd0) begin
      tx_shift <= {1'b1, tx_shift[9:1]};
      tx_cnt   <= tx_cnt - 4'd1;
    end
  end
  assign out = tx_shift[0];
endmodule
""", [
    Crossing(cid=0, template="MUX_HOLD", safe=True, src_clk="clk0", dst_clk="clk1",
             cap_nets=["cdc0_cap", "cdc0_hold"], all_prefix="cdc0_"),
])

# --------------------------------------------------------------- UART bad
_add("r_uart_bad", 2, """
module top(input clk0, input clk1, input [7:0] din, output out);
  // BROKEN: baud domain samples the sys-domain data bus directly.
  reg [7:0] cdc0_data = 8'd0;
  reg cdc1_req = 1'b0;
  reg [3:0] sys_cnt = 4'd0;
  always @(posedge clk0) begin
    sys_cnt <= sys_cnt + 4'd1;
    if (sys_cnt == 4'd0) begin
      cdc0_data <= din;
      cdc1_req  <= ~cdc1_req;
    end
  end
  reg cdc1_cap = 1'b0;                 // single-stage sync, used at once
  reg [7:0] cdc0_cap = 8'd0;           // raw bus capture: incoherent word
  reg [9:0] tx_shift = 10'b1111111111;
  reg [3:0] tx_cnt = 4'd0;
  reg cdc1_last = 1'b0;
  wire cdc1_go = cdc1_cap ^ cdc1_last; // uses stage-0 output directly
  always @(posedge clk1) begin
    cdc1_cap  <= cdc1_req;
    cdc1_last <= cdc1_go ? ~cdc1_last : cdc1_last;
    cdc0_cap  <= cdc0_data;
    if (cdc1_go) begin
      tx_shift <= {1'b1, cdc0_cap, 1'b0};
      tx_cnt   <= 4'd10;
    end else if (tx_cnt != 4'd0) begin
      tx_shift <= {1'b1, tx_shift[9:1]};
      tx_cnt   <= tx_cnt - 4'd1;
    end
  end
  assign out = tx_shift[0];
endmodule
""", [
    Crossing(cid=0, template="BUS_NOSYNC", safe=False, src_clk="clk0", dst_clk="clk1",
             cap_nets=["cdc0_cap"], all_prefix="cdc0_"),
    Crossing(cid=1, template="SINGLE_FF", safe=False, src_clk="clk0", dst_clk="clk1",
             cap_nets=["cdc1_cap"], all_prefix="cdc1_"),
])

# ---------------------------------------------------------------- SPI ok
_add("r_spi_good", 2, """
module top(input clk0, input clk1, input [7:0] din, output out);
  // clk0 = SPI clock domain (shift), clk1 = system domain (consume)
  reg [7:0] spi_shift = 8'd0;
  reg [2:0] bit_cnt = 3'd0;
  reg [7:0] cdc0_byte = 8'd0;
  reg cdc0_flag = 1'b0;
  always @(posedge clk0) begin
    spi_shift <= {spi_shift[6:0], din[0]};
    bit_cnt   <= bit_cnt + 3'd1;
    if (bit_cnt == 3'd7) begin
      cdc0_byte <= {spi_shift[6:0], din[0]};
      cdc0_flag <= ~cdc0_flag;
    end
  end
  reg cdc0_cap = 1'b0, cdc0_s1 = 1'b0, cdc0_s2 = 1'b0;
  reg [7:0] cdc0_hold = 8'd0;
  wire cdc0_take = cdc0_s1 ^ cdc0_s2;
  reg [7:0] sys_acc = 8'd0;
  always @(posedge clk1) begin
    cdc0_cap <= cdc0_flag;
    cdc0_s1  <= cdc0_cap;
    cdc0_s2  <= cdc0_s1;
    cdc0_hold <= cdc0_take ? cdc0_byte : cdc0_hold;
    if (cdc0_take) sys_acc <= sys_acc + cdc0_hold;
  end
  assign out = ^sys_acc;
endmodule
""", [
    Crossing(cid=0, template="MUX_HOLD", safe=True, src_clk="clk0", dst_clk="clk1",
             cap_nets=["cdc0_cap", "cdc0_hold"], all_prefix="cdc0_"),
])

# --------------------------------------------------------------- SPI bad
_add("r_spi_bad", 2, """
module top(input clk0, input clk1, input [7:0] din, output out);
  // BROKEN: system domain reads the live shift register combinationally.
  reg [7:0] spi_shift = 8'd0;
  reg [2:0] bit_cnt = 3'd0;
  reg cdc1_flag = 1'b0;
  always @(posedge clk0) begin
    spi_shift <= {spi_shift[6:0], din[0]};
    bit_cnt   <= bit_cnt + 3'd1;
    if (bit_cnt == 3'd7) cdc1_flag <= ~cdc1_flag;
  end
  wire [7:0] cdc0_mix = spi_shift ^ {8{cdc1_flag}};  // comb across domains
  reg [7:0] cdc0_cap = 8'd0;
  reg [7:0] sys_acc = 8'd0;
  always @(posedge clk1) begin
    cdc0_cap <= cdc0_mix;              // glitchy comb into capture FFs
    sys_acc  <= sys_acc + cdc0_cap;
  end
  assign out = ^sys_acc;
endmodule
""", [
    Crossing(cid=0, template="COMB_BEFORE_SYNC", safe=False, src_clk="clk0",
             dst_clk="clk1", cap_nets=["cdc0_cap"], all_prefix="cdc0_"),
])

# ----------------------------------------------------------- handshake ok
_add("r_hs_good", 2, """
module top(input clk0, input clk1, input [7:0] din, output out);
  // four-phase handshake, both signals through 2FF syncs
  reg cdc0_req = 1'b0;      // clk0 -> clk1
  reg cdc1_ack = 1'b0;      // clk1 -> clk0
  reg [7:0] cdc0_data = 8'd0;
  reg cdc1_cap = 1'b0, cdc1_s1 = 1'b0;
  reg busy = 1'b0;
  always @(posedge clk0) begin
    cdc1_cap <= cdc1_ack;
    cdc1_s1  <= cdc1_cap;
    if (!busy && din[7]) begin
      cdc0_data <= din;
      cdc0_req  <= 1'b1;
      busy      <= 1'b1;
    end else if (busy && cdc1_s1) begin
      cdc0_req <= 1'b0;
      busy     <= cdc1_s1 ? 1'b0 : busy;
    end
  end
  reg cdc0_cap = 1'b0, cdc0_s1 = 1'b0;
  reg [7:0] cdc0_hold = 8'd0;
  always @(posedge clk1) begin
    cdc0_cap <= cdc0_req;
    cdc0_s1  <= cdc0_cap;
    if (cdc0_s1 && !cdc1_ack) begin
      cdc0_hold <= cdc0_data;   // data stable while req high: qualifier hold
      cdc1_ack  <= 1'b1;
    end else if (!cdc0_s1 && cdc1_ack)
      cdc1_ack <= 1'b0;
  end
  assign out = ^cdc0_hold;
endmodule
""", [
    Crossing(cid=0, template="MUX_HOLD", safe=True, src_clk="clk0", dst_clk="clk1",
             cap_nets=["cdc0_cap", "cdc0_hold"], all_prefix="cdc0_"),
    Crossing(cid=1, template="SYNC2", safe=True, src_clk="clk1", dst_clk="clk0",
             cap_nets=["cdc1_cap"], all_prefix="cdc1_"),
])

# ---------------------------------------------------------- handshake bad
_add("r_hs_bad", 2, """
module top(input clk0, input clk1, input [7:0] din, output out);
  // BROKEN handshake: ack crosses back with no synchronizer at all.
  reg cdc0_req = 1'b0;
  reg cdc1_ack = 1'b0;
  reg [7:0] cdc0_data = 8'd0;
  reg busy = 1'b0;
  wire cdc1_done = cdc1_ack & ~busy;   // comb use of async ack
  reg cdc1_cap = 1'b0, cdc1_capx = 1'b0;
  always @(posedge clk0) begin
    cdc1_cap  <= cdc1_done;
    cdc1_capx <= cdc1_ack | din[6];
    if (!busy && din[7]) begin
      cdc0_data <= din;
      cdc0_req  <= 1'b1;
      busy      <= 1'b1;
    end else if (busy && cdc1_cap) begin
      cdc0_req <= 1'b0;
      busy     <= 1'b0;
    end
  end
  reg cdc0_cap = 1'b0, cdc0_s1 = 1'b0;
  reg [7:0] cdc0_hold = 8'd0;
  always @(posedge clk1) begin
    cdc0_cap <= cdc0_req;
    cdc0_s1  <= cdc0_cap;
    if (cdc0_s1 && !cdc1_ack) begin
      cdc0_hold <= cdc0_data;
      cdc1_ack  <= 1'b1;
    end else if (!cdc0_s1 && cdc1_ack)
      cdc1_ack <= 1'b0;
  end
  assign out = ^cdc0_hold;
endmodule
""", [
    Crossing(cid=0, template="MUX_HOLD", safe=True, src_clk="clk0", dst_clk="clk1",
             cap_nets=["cdc0_cap", "cdc0_hold"], all_prefix="cdc0_"),
    Crossing(cid=1, template="NO_SYNC", safe=False, src_clk="clk1", dst_clk="clk0",
             cap_nets=["cdc1_cap", "cdc1_capx"], all_prefix="cdc1_"),
])


# ------------------------------------------------------- pulse sync (good)
_add("r_pulse_good", 2, """
module top(input clk0, input clk1, input [7:0] din, output out);
  // toggle-based pulse synchronizer: event in clk0 -> single pulse in clk1
  reg cdc0_tgl = 1'b0;
  reg [3:0] ev_cnt = 4'd0;
  always @(posedge clk0) begin
    ev_cnt <= ev_cnt + 4'd1;
    if (ev_cnt == 4'd0 && din[0]) cdc0_tgl <= ~cdc0_tgl;
  end
  reg cdc0_cap = 1'b0, cdc0_s1 = 1'b0, cdc0_s2 = 1'b0;
  reg [3:0] pulse_cnt = 4'd0;
  wire pulse = cdc0_s1 ^ cdc0_s2;
  always @(posedge clk1) begin
    cdc0_cap <= cdc0_tgl;
    cdc0_s1  <= cdc0_cap;
    cdc0_s2  <= cdc0_s1;
    if (pulse) pulse_cnt <= pulse_cnt + 4'd1;
  end
  assign out = ^pulse_cnt;
endmodule
""", [
    Crossing(cid=0, template="SYNC2", safe=True, src_clk="clk0", dst_clk="clk1",
             cap_nets=["cdc0_cap"], all_prefix="cdc0_"),
])

# -------------------------------------------------------- pulse sync (bad)
_add("r_pulse_bad", 2, """
module top(input clk0, input clk1, input [7:0] din, output out);
  // BROKEN: the raw toggle enters the edge-detect comb directly.
  reg cdc0_tgl = 1'b0;
  reg [3:0] ev_cnt = 4'd0;
  always @(posedge clk0) begin
    ev_cnt <= ev_cnt + 4'd1;
    if (ev_cnt == 4'd0 && din[0]) cdc0_tgl <= ~cdc0_tgl;
  end
  reg cdc0_cap = 1'b0;
  reg [3:0] pulse_cnt = 4'd0;
  wire pulse = cdc0_tgl ^ cdc0_cap;      // raw async in dest comb
  always @(posedge clk1) begin
    cdc0_cap <= cdc0_tgl;
    if (pulse) pulse_cnt <= pulse_cnt + 4'd1;
  end
  assign out = ^pulse_cnt;
endmodule
""", [
    Crossing(cid=0, template="NO_SYNC", safe=False, src_clk="clk0", dst_clk="clk1",
             cap_nets=["cdc0_cap"], all_prefix="cdc0_"),
])

# --------------------------------------------- async FIFO, binary pointers
_add("r_fifo_binary_bad", 2, """
module top(input clk0, input clk1, input [7:0] din, output out);
  // CLASSIC BUG: async FIFO with BINARY pointers through per-bit 2FF syncs
  reg [7:0] mem [0:3];
  reg [2:0] cdc0_wptr = 3'd0;   // binary write pointer crossing to clk1
  reg [2:0] cdc1_rptr = 3'd0;   // binary read pointer crossing to clk0
  wire wput = din[0];
  wire rget = din[1];
  reg [2:0] cdc1_cap = 3'd0, cdc1_s1 = 3'd0;
  wire full = ((cdc0_wptr - cdc1_s1) == 3'd4);
  always @(posedge clk0) begin
    cdc1_cap <= cdc1_rptr;
    cdc1_s1  <= cdc1_cap;
    if (wput && !full) begin
      mem[cdc0_wptr[1:0]] <= din;
      cdc0_wptr <= cdc0_wptr + 3'd1;
    end
  end
  reg [2:0] cdc0_cap = 3'd0, cdc0_s1 = 3'd0;
  reg [7:0] rdata = 8'd0;
  wire empty = (cdc0_s1 == cdc1_rptr);
  always @(posedge clk1) begin
    cdc0_cap <= cdc0_wptr;
    cdc0_s1  <= cdc0_cap;
    if (rget && !empty) begin
      rdata <= mem[cdc1_rptr[1:0]];
      cdc1_rptr <= cdc1_rptr + 3'd1;
    end
  end
  assign out = ^rdata ^ full ^ empty;
endmodule
""", [
    Crossing(cid=0, template="BUS_NOSYNC", safe=False, src_clk="clk0", dst_clk="clk1",
             cap_nets=["cdc0_cap"], all_prefix="cdc0_"),
    Crossing(cid=1, template="BUS_NOSYNC", safe=False, src_clk="clk1", dst_clk="clk0",
             cap_nets=["cdc1_cap"], all_prefix="cdc1_"),
])

# ------------------------------------------------------- wide gray counter
_add("r_gray_wide", 2, """
module top(input clk0, input clk1, input [7:0] din, output out);
  // 4-bit gray-coded event counter crossing, decoded in the destination
  reg [3:0] bin = 4'd0;
  reg [3:0] cdc0_gray = 4'd0;
  always @(posedge clk0) begin
    if (din[0]) begin
      bin <= bin + 4'd1;
      cdc0_gray <= ((bin + 4'd1) >> 1) ^ (bin + 4'd1);
    end
  end
  reg [3:0] cdc0_cap = 4'd0, cdc0_s1 = 4'd0;
  reg [3:0] level = 4'd0;
  always @(posedge clk1) begin
    cdc0_cap <= cdc0_gray;
    cdc0_s1  <= cdc0_cap;
    level <= cdc0_s1 ^ {1'b0, cdc0_s1[3:1]};   // gray -> binary decode
  end
  assign out = ^level;
endmodule
""", [
    Crossing(cid=0, template="GRAY2", safe=True, src_clk="clk0", dst_clk="clk1",
             cap_nets=["cdc0_cap"], all_prefix="cdc0_"),
])

# ------------------------------------------ handshake with stage-0 leak
_add("r_hs_leak_bad", 2, """
module top(input clk0, input clk1, input [7:0] din, output out);
  // SUBTLE BUG: req is 2FF-synced, but the data-hold mux is qualified by
  // the METASTABLE stage-0 output instead of the settled stage-1.
  reg cdc0_req = 1'b0;
  reg [7:0] cdc0_data = 8'd0;
  reg [3:0] tick = 4'd0;
  always @(posedge clk0) begin
    tick <= tick + 4'd1;
    if (tick == 4'd0) begin
      cdc0_data <= din;
      cdc0_req  <= ~cdc0_req;
    end
  end
  reg cdc0_cap = 1'b0, cdc0_s1 = 1'b0;
  reg cdc0_lastp = 1'b0;
  reg [7:0] cdc0_hold = 8'd0;
  wire cdc0_take = cdc0_cap ^ cdc0_lastp;   // uses stage-0! (should be s1)
  always @(posedge clk1) begin
    cdc0_cap   <= cdc0_req;
    cdc0_s1    <= cdc0_cap;
    cdc0_lastp <= cdc0_cap;
    cdc0_hold  <= cdc0_take ? cdc0_data : cdc0_hold;
  end
  assign out = ^cdc0_hold ^ cdc0_s1;
endmodule
""", [
    Crossing(cid=0, template="SINGLE_FF", safe=False, src_clk="clk0", dst_clk="clk1",
             cap_nets=["cdc0_cap", "cdc0_hold"], all_prefix="cdc0_"),
])


def real_designs() -> list:
    return list(_DESIGNS)
