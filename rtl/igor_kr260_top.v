`timescale 1ns/1ps
`default_nettype none

// Top-Level FPGA Design for AMD Kria KR260 / Project Hephaestus
// Author: Brian A. Hadian
//
// Target Board: AMD Kria KR260 Robotics Starter Kit
// Target Silicon: AMD Zynq UltraScale+ MPSoC (xck26-sfvc784-2LV-c)
//
// Dual-Interface Compute Architecture:
// 1. High-Speed AXI4-Lite Memory-Mapped Bus (PS <-> PL @ 50-100 MHz):
//    Instantaneous sub-millisecond register access via /dev/mem on Ubuntu 22.04 LTS.
// 2. High-Speed Serial UART Framing Protocol (Pmod / RPi Header @ 921,600 baud):
//    CRC-8 Maxim protected 153-byte request / 11-byte response packets.
//
// Safety Core:
// - Zero-latency RTL Safety & Security Gate with monotonic 20 ms lease watchdog
// - Physical and software fail-closed Emergency Stop interlocks
// - Non-maskable hardware status logger

module igor_kr260_top #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 12,
    parameter integer CLK_FREQ_HZ        = 50000000,
    parameter integer UART_BAUD          = 921600,
    parameter BANK0_FILE = "rtl/mem/bank0.mem",
    parameter BANK1_FILE = "rtl/mem/bank1.mem",
    parameter BANK2_FILE = "rtl/mem/bank2.mem",
    parameter BANK3_FILE = "rtl/mem/bank3.mem",
    parameter BANK4_FILE = "rtl/mem/bank4.mem",
    parameter BANK5_FILE = "rtl/mem/bank5.mem",
    parameter BANK6_FILE = "rtl/mem/bank6.mem",
    parameter BANK7_FILE = "rtl/mem/bank7.mem"
)(
    // Clock & Reset
    input  wire                                  clk,
    input  wire                                  rst_n,

    // AXI4-Lite Slave Interface (Connected to PS M_AXI_HPM0_FPD)
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]        s_axi_awaddr,
    input  wire [2:0]                           s_axi_awprot,
    input  wire                                 s_axi_awvalid,
    output wire                                 s_axi_awready,

    input  wire [C_S_AXI_DATA_WIDTH-1:0]        s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0]    s_axi_wstrb,
    input  wire                                 s_axi_wvalid,
    output wire                                 s_axi_wready,

    output wire [1:0]                           s_axi_bresp,
    output wire                                 s_axi_bvalid,
    input  wire                                 s_axi_bready,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]        s_axi_araddr,
    input  wire [2:0]                           s_axi_arprot,
    input  wire                                 s_axi_arvalid,
    output wire                                 s_axi_arready,

    output wire [C_S_AXI_DATA_WIDTH-1:0]        s_axi_rdata,
    output wire [1:0]                           s_axi_rresp,
    output wire                                 s_axi_rvalid,
    input  wire                                 s_axi_rready,

    // External Physical I/O (Pmod / RPi Header on KR260)
    input  wire                                 pmod_uart_rx,
    output wire                                 pmod_uart_tx,
    input  wire                                 estop_btn_n, // Active-low external button
    output wire [7:0]                           led_status
);
    // Heartbeat clock divider
    reg [25:0] heartbeat_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) heartbeat_cnt <= 26'd0;
        else heartbeat_cnt <= heartbeat_cnt + 1'b1;
    end
    wire heartbeat = heartbeat_cnt[24]; // ~1.5 Hz blink

    // Active-high physical E-stop inversion
    wire estop_physical = !estop_btn_n;

    // --- Neural Policy Core Signals ---
    wire        core_load_input;
    wire [6:0]  core_input_index;
    wire signed [15:0] core_input_value;
    wire        core_start;
    wire        core_abort;
    wire [1:0]  core_output_index;
    wire signed [15:0] core_output_value;
    wire        core_busy;
    wire        core_done;
    wire        core_fault;
    wire [3:0]  core_error_code;
    wire [31:0] core_latency_cycles;
    wire        core_saturated;

    // Latched actions from neural core
    reg signed [15:0] latched_a0, latched_a1, latched_a2;
    reg [1:0] capture_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            latched_a0    <= 16'sd0;
            latched_a1    <= 16'sd0;
            latched_a2    <= 16'sd0;
            capture_state <= 2'd0;
        end else if (core_done) begin
            case (capture_state)
                2'd0: begin
                    latched_a0    <= core_output_value;
                    capture_state <= 2'd1;
                end
                2'd1: begin
                    latched_a1    <= core_output_value;
                    capture_state <= 2'd2;
                end
                2'd2: begin
                    latched_a2    <= core_output_value;
                    capture_state <= 2'd0;
                end
            endcase
        end else begin
            capture_state <= 2'd0;
        end
    end

    assign core_output_index = capture_state;

    // --- Neural Policy Accelerator Instance ---
    igor_kr260_policy_core #(
        .BANK0_FILE(BANK0_FILE),
        .BANK1_FILE(BANK1_FILE),
        .BANK2_FILE(BANK2_FILE),
        .BANK3_FILE(BANK3_FILE),
        .BANK4_FILE(BANK4_FILE),
        .BANK5_FILE(BANK5_FILE),
        .BANK6_FILE(BANK6_FILE),
        .BANK7_FILE(BANK7_FILE)
    ) policy_accelerator (
        .clk(clk),
        .rst_n(rst_n && !axi_core_soft_reset),
        .load_input(core_load_input),
        .input_index(core_input_index),
        .input_value(core_input_value),
        .start(core_start),
        .abort(core_abort | estop_physical | axi_estop_soft),
        .output_index(core_output_index),
        .output_value(core_output_value),
        .busy(core_busy),
        .done(core_done),
        .fault(core_fault),
        .error_code(core_error_code),
        .latency_cycles(core_latency_cycles),
        .saturated(core_saturated)
    );

    // CRC-8 Maxim Calculation for Action Triple
    function [7:0] crc8_step;
        input [7:0] prev, byte_in;
        reg [7:0] c;
        integer j;
        begin
            c = prev ^ byte_in;
            for (j = 0; j < 8; j = j + 1)
                c = c[0] ? (c >> 1) ^ 8'h8c : (c >> 1);
            crc8_step = c;
        end
    endfunction

    function [7:0] calc_action_crc;
        input [15:0] a, b, c;
        reg [7:0] sum;
        begin
            sum = crc8_step(8'd0, 8'h02);
            sum = crc8_step(sum,  8'h06);
            sum = crc8_step(sum,  a[7:0]);
            sum = crc8_step(sum,  a[15:8]);
            sum = crc8_step(sum,  b[7:0]);
            sum = crc8_step(sum,  b[15:8]);
            sum = crc8_step(sum,  c[7:0]);
            sum = crc8_step(sum,  c[15:8]);
            calc_action_crc = sum;
        end
    endfunction

    wire [7:0] core_action_crc = calc_action_crc(latched_a0, latched_a1, latched_a2);

    // --- Safety & Security Gate ---
    wire        safety_permit;
    wire signed [15:0] safe_vx, safe_vy, safe_wz;
    wire [7:0]  active_fault_code;
    wire [31:0] lease_timer;
    wire        lease_expired;
    wire        estop_latched;
    wire [7:0]  latched_fault_code;
    wire [31:0] event_count;

    wire        axi_estop_soft;
    wire        axi_estop_clear;
    wire        axi_lease_refresh;
    wire [31:0] axi_lease_limit;
    wire signed [15:0] axi_cfg_max_v;
    wire signed [15:0] axi_cfg_max_w;
    wire        axi_core_soft_reset;

    igor_safety_security_gate #(
        .DEFAULT_LEASE_CYCLES(32'd1000000) // 20 ms @ 50 MHz
    ) safety_gate (
        .clk(clk),
        .rst_n(rst_n),
        .ready(!core_busy),
        .estop_in(estop_physical),
        .estop_soft(axi_estop_soft),
        .estop_clear(axi_estop_clear),
        .protective_stop(1'b0),
        .lease_refresh(axi_lease_refresh),
        .lease_limit(axi_lease_limit),
        .result_valid(core_done && !core_fault),
        .arithmetic_fault(core_fault),
        .raw_vx(latched_a0),
        .raw_vy(latched_a1),
        .raw_wz(latched_a2),
        .max_v(axi_cfg_max_v),
        .max_w(axi_cfg_max_w),
        .safety_permit(safety_permit),
        .safe_vx(safe_vx),
        .safe_vy(safe_vy),
        .safe_wz(safe_wz),
        .fault_code(active_fault_code),
        .lease_timer(lease_timer),
        .lease_expired(lease_expired),
        .estop_latched(estop_latched)
    );

    igor_status_logger fault_logger (
        .clk(clk),
        .rst_n(rst_n),
        .fault_code(active_fault_code),
        .event_count(event_count),
        .last_fault(latched_fault_code)
    );

    // --- AXI4-Lite Slave Bridge ---
    wire        axi_load_input;
    wire [6:0]  axi_input_index;
    wire signed [15:0] axi_input_value;
    wire        axi_start;
    wire        axi_abort;

    igor_axi_lite_bridge #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
        .CLK_FREQ_HZ(CLK_FREQ_HZ)
    ) axi_bridge (
        .S_AXI_ACLK(clk),
        .S_AXI_ARESETN(rst_n),
        .S_AXI_AWADDR(s_axi_awaddr),
        .S_AXI_AWPROT(s_axi_awprot),
        .S_AXI_AWVALID(s_axi_awvalid),
        .S_AXI_AWREADY(s_axi_awready),
        .S_AXI_WDATA(s_axi_wdata),
        .S_AXI_WSTRB(s_axi_wstrb),
        .S_AXI_WVALID(s_axi_wvalid),
        .S_AXI_WREADY(s_axi_wready),
        .S_AXI_BRESP(s_axi_bresp),
        .S_AXI_BVALID(s_axi_bvalid),
        .S_AXI_BREADY(s_axi_bready),
        .S_AXI_ARADDR(s_axi_araddr),
        .S_AXI_ARPROT(s_axi_arprot),
        .S_AXI_ARVALID(s_axi_arvalid),
        .S_AXI_ARREADY(s_axi_arready),
        .S_AXI_RDATA(s_axi_rdata),
        .S_AXI_RRESP(s_axi_rresp),
        .S_AXI_RVALID(s_axi_rvalid),
        .S_AXI_RREADY(s_axi_rready),

        // Neural Core Interconnect
        .core_load_input(axi_load_input),
        .core_input_index(axi_input_index),
        .core_input_value(axi_input_value),
        .core_start(axi_start),
        .core_abort(axi_abort),
        .core_soft_reset(axi_core_soft_reset),
        .core_busy(core_busy),
        .core_done(core_done),
        .core_fault(core_fault),
        .core_saturated(core_saturated),
        .core_error_code(core_error_code),
        .core_latency_cycles(core_latency_cycles),
        .core_action0(safe_vx),
        .core_action1(safe_vy),
        .core_action2(safe_wz),
        .core_action_crc(core_action_crc),

        // Safety Interconnect
        .estop_soft(axi_estop_soft),
        .estop_clear(axi_estop_clear),
        .lease_refresh(axi_lease_refresh),
        .lease_limit(axi_lease_limit),
        .cfg_max_v(axi_cfg_max_v),
        .cfg_max_w(axi_cfg_max_w),
        .safety_permit(safety_permit),
        .estop_active(estop_physical | axi_estop_soft | estop_latched),
        .lease_expired(lease_expired),
        .lease_timer(lease_timer),
        .active_fault_code(active_fault_code),
        .latched_fault_code(latched_fault_code),
        .event_count(event_count),
        .error_flags(16'd0),
        .debug_state({1'b0, capture_state, core_busy})
    );

    // Assign Core Control Signals to AXI
    assign core_load_input  = axi_load_input;
    assign core_input_index = axi_input_index;
    assign core_input_value = axi_input_value;
    assign core_start       = axi_start;
    assign core_abort       = axi_abort;

    // --- UART Protocol Engine Instance (Pmod) ---
    igor_kr260_policy_uart #(
        .CLK_HZ(CLK_FREQ_HZ),
        .BAUD(UART_BAUD),
        .BANK0_FILE(BANK0_FILE),
        .BANK1_FILE(BANK1_FILE),
        .BANK2_FILE(BANK2_FILE),
        .BANK3_FILE(BANK3_FILE),
        .BANK4_FILE(BANK4_FILE),
        .BANK5_FILE(BANK5_FILE),
        .BANK6_FILE(BANK6_FILE),
        .BANK7_FILE(BANK7_FILE)
    ) pmod_uart_engine (
        .clk(clk),
        .rst_n(rst_n),
        .uart_rx(pmod_uart_rx),
        .estop(estop_physical | axi_estop_soft),
        .uart_tx(pmod_uart_tx),
        .debug_state(),
        .error_code(),
        .error_flags(),
        .accepted_frames(),
        .rejected_frames(),
        .last_latency_cycles(),
        .safety_permit(),
        .response_valid(),
        .action0(),
        .action1(),
        .action2()
    );

    // --- Diagnostic Status LEDs on KR260 ---
    // LED[0]: Heartbeat (~1 Hz toggle)
    // LED[1]: Safety Permit (1 = Actuation Permitted)
    // LED[2]: Core Done
    // LED[3]: Core Busy
    // LED[7:4]: Fault Code lower nibble (0000 = OK, 0010 = E-Stop, 1000 = Watchdog Timeout)
    assign led_status = {
        active_fault_code[3:0],
        core_busy,
        core_done,
        safety_permit,
        heartbeat
    };

endmodule
`default_nettype wire
