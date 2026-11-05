`timescale 1ns/1ps
`default_nettype none

// AXI4-Lite High-Speed Slave Memory-Mapped Bridge for AMD Kria KR260 / Project Hephaestus
// Author: Brian A. Hadian
//
// Direct memory-mapped access for ARM Cortex-A53 PS running Ubuntu 22.04 LTS.
// Connects to Zynq MPSoC M_AXI_HPM0_FPD / M_AXI_HPM0_LPD at base address (e.g. 0xA000_0000).
// Enables ~140 us complete roundtrip neural policy evaluation directly via /dev/mem (mmap).

module igor_axi_lite_bridge #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 12, // 4 KB Address Space
    parameter integer CLK_FREQ_HZ        = 50000000
)(
    // AXI4-Lite Clock & Reset
    input  wire                                  S_AXI_ACLK,
    input  wire                                  S_AXI_ARESETN,

    // Write Address Channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]        S_AXI_AWADDR,
    input  wire [2:0]                           S_AXI_AWPROT,
    input  wire                                 S_AXI_AWVALID,
    output reg                                  S_AXI_AWREADY,

    // Write Data Channel
    input  wire [C_S_AXI_DATA_WIDTH-1:0]        S_AXI_WDATA,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0]    S_AXI_WSTRB,
    input  wire                                 S_AXI_WVALID,
    output reg                                  S_AXI_WREADY,

    // Write Response Channel
    output reg  [1:0]                           S_AXI_BRESP,
    output reg                                  S_AXI_BVALID,
    input  wire                                 S_AXI_BREADY,

    // Read Address Channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]        S_AXI_ARADDR,
    input  wire [2:0]                           S_AXI_ARPROT,
    input  wire                                 S_AXI_ARVALID,
    output reg                                  S_AXI_ARREADY,

    // Read Data Channel
    output reg  [C_S_AXI_DATA_WIDTH-1:0]        S_AXI_RDATA,
    output reg  [1:0]                           S_AXI_RRESP,
    output reg                                  S_AXI_RVALID,
    input  wire                                 S_AXI_RREADY,

    // Neural Core Control & Interface
    output reg                                  core_load_input,
    output reg  [6:0]                           core_input_index,
    output reg  signed [15:0]                   core_input_value,
    output reg                                  core_start,
    output reg                                  core_abort,
    output reg                                  core_soft_reset,
    input  wire                                 core_busy,
    input  wire                                 core_done,
    input  wire                                 core_fault,
    input  wire                                 core_saturated,
    input  wire [3:0]                           core_error_code,
    input  wire [31:0]                          core_latency_cycles,
    input  wire signed [15:0]                   core_action0,
    input  wire signed [15:0]                   core_action1,
    input  wire signed [15:0]                   core_action2,
    input  wire [7:0]                           core_action_crc,

    // Hardware Safety & Security Interface
    output reg                                  estop_soft,
    output reg                                  estop_clear,
    output reg                                  lease_refresh,
    output reg  [31:0]                          lease_limit,
    output reg  signed [15:0]                   cfg_max_v,
    output reg  signed [15:0]                   cfg_max_w,
    input  wire                                 safety_permit,
    input  wire                                 estop_active,
    input  wire                                 lease_expired,
    input  wire [31:0]                          lease_timer,
    input  wire [7:0]                           active_fault_code,
    input  wire [7:0]                           latched_fault_code,
    input  wire [31:0]                          event_count,
    input  wire [15:0]                          error_flags,
    input  wire [3:0]                           debug_state
);
    localparam [31:0] HW_ID_VAL = 32'h4B523236; // ASCII "KR26"

    // Internal memory for 74 observations (32-bit words, lower 16 bits hold Q8.8 value)
    reg signed [15:0] observation_regs [0:73];

    // Local statistics counters
    reg [31:0] accepted_frames;
    reg [31:0] rejected_frames;

    // AXI handshake registers
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    reg aw_en;

    wire [6:0] wr_obs_idx = axi_awaddr[8:2] - 7'd64;
    wire [6:0] rd_obs_idx = axi_araddr[8:2] - 7'd64;

    // --- AXI Write Transaction FSM ---
    always @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) begin
            S_AXI_AWREADY   <= 1'b0;
            S_AXI_WREADY    <= 1'b0;
            S_AXI_BVALID    <= 1'b0;
            S_AXI_BRESP     <= 2'b00;
            aw_en           <= 1'b1;
            axi_awaddr      <= {C_S_AXI_ADDR_WIDTH{1'b0}};
            core_load_input <= 1'b0;
            core_start      <= 1'b0;
            core_abort      <= 1'b0;
            core_soft_reset <= 1'b0;
            estop_soft      <= 1'b0;
            estop_clear     <= 1'b0;
            lease_refresh   <= 1'b0;
            lease_limit     <= 32'd1000000; // 20 ms @ 50 MHz
            cfg_max_v       <= 16'sd256;    // 1.0 m/s
            cfg_max_w       <= 16'sd384;    // 1.5 rad/s
            accepted_frames <= 32'd0;
            rejected_frames <= 32'd0;
        end else begin
            // Single-cycle strobes clear automatically
            core_load_input <= 1'b0;
            core_start      <= 1'b0;
            core_abort      <= 1'b0;
            core_soft_reset <= 1'b0;
            estop_clear     <= 1'b0;
            lease_refresh   <= 1'b0;

            // Track accepted/rejected frames
            if (core_done && safety_permit) begin
                if (accepted_frames != 32'hffffffff) accepted_frames <= accepted_frames + 1'b1;
            end else if (core_done && !safety_permit) begin
                if (rejected_frames != 32'hffffffff) rejected_frames <= rejected_frames + 1'b1;
            end

            // Address write handshake
            if (!S_AXI_AWREADY && S_AXI_AWVALID && S_AXI_WVALID && aw_en) begin
                S_AXI_AWREADY <= 1'b1;
                S_AXI_WREADY  <= 1'b1;
                aw_en         <= 1'b0;
                axi_awaddr    <= S_AXI_AWADDR;
            end else begin
                S_AXI_AWREADY <= 1'b0;
                S_AXI_WREADY  <= 1'b0;
            end

            // Write Data Execution
            if (S_AXI_WREADY && S_AXI_WVALID && S_AXI_AWREADY && S_AXI_AWVALID) begin
                if (axi_awaddr >= 12'h100 && axi_awaddr <= 12'h224) begin
                    // Observation registers (0x100 to 0x224): index = (addr - 0x100) / 4
                    if (wr_obs_idx < 7'd74) begin
                        observation_regs[wr_obs_idx] <= S_AXI_WDATA[15:0];
                        core_load_input              <= 1'b1;
                        core_input_index             <= wr_obs_idx;
                        core_input_value             <= S_AXI_WDATA[15:0];
                    end
                end else begin
                    case (axi_awaddr[7:0])
                        8'h00: begin // CR: Control Register
                            if (S_AXI_WDATA[0]) core_start      <= 1'b1;
                            if (S_AXI_WDATA[1]) core_abort      <= 1'b1;
                            if (S_AXI_WDATA[2]) estop_soft      <= 1'b1;
                            if (S_AXI_WDATA[3]) core_soft_reset <= 1'b1;
                            if (S_AXI_WDATA[4]) lease_refresh   <= 1'b1;
                            if (S_AXI_WDATA[5]) estop_clear     <= 1'b1;
                            if (S_AXI_WDATA[6]) estop_soft      <= 1'b0; // Software estop release
                        end
                        8'h20: lease_limit <= S_AXI_WDATA;
                        8'h40: cfg_max_v   <= S_AXI_WDATA[15:0];
                        8'h44: cfg_max_w   <= S_AXI_WDATA[15:0];
                        default: begin end
                    endcase
                end
                S_AXI_BVALID <= 1'b1;
                S_AXI_BRESP  <= 2'b00; // OKAY
            end

            if (S_AXI_BVALID && S_AXI_BREADY) begin
                S_AXI_BVALID <= 1'b0;
                aw_en        <= 1'b1;
            end
        end
    end

    // --- AXI Read Transaction FSM ---
    always @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) begin
            S_AXI_ARREADY <= 1'b0;
            S_AXI_RVALID  <= 1'b0;
            S_AXI_RRESP   <= 2'b00;
            S_AXI_RDATA   <= 32'd0;
            axi_araddr    <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else begin
            if (!S_AXI_ARREADY && S_AXI_ARVALID) begin
                S_AXI_ARREADY <= 1'b1;
                axi_araddr    <= S_AXI_ARADDR;
            end else begin
                S_AXI_ARREADY <= 1'b0;
            end

            if (S_AXI_ARREADY && S_AXI_ARVALID && !S_AXI_RVALID) begin
                S_AXI_RVALID <= 1'b1;
                S_AXI_RRESP  <= 2'b00; // OKAY

                // Decode Read Address
                if (axi_araddr >= 12'h100 && axi_araddr <= 12'h224) begin
                    if (rd_obs_idx < 7'd74) begin
                        S_AXI_RDATA <= {{16{observation_regs[rd_obs_idx][15]}}, observation_regs[rd_obs_idx]};
                    end else begin
                        S_AXI_RDATA <= 32'd0;
                    end
                end else begin
                    case (axi_araddr[7:0])
                        8'h00: S_AXI_RDATA <= {25'd0, estop_clear, lease_refresh, core_soft_reset, estop_soft, core_abort, core_start};
                        8'h04: S_AXI_RDATA <= {20'd0, debug_state, lease_expired, estop_active, core_saturated, safety_permit, core_fault, core_done, core_busy};
                        8'h08: S_AXI_RDATA <= {error_flags, latched_fault_code, active_fault_code};
                        8'h0C: S_AXI_RDATA <= core_latency_cycles;
                        8'h10: S_AXI_RDATA <= event_count;
                        8'h14: S_AXI_RDATA <= accepted_frames;
                        8'h18: S_AXI_RDATA <= rejected_frames;
                        8'h1C: S_AXI_RDATA <= lease_timer;
                        8'h20: S_AXI_RDATA <= lease_limit;
                        8'h24: S_AXI_RDATA <= {{16{core_action0[15]}}, core_action0};
                        8'h28: S_AXI_RDATA <= {{16{core_action1[15]}}, core_action1};
                        8'h2C: S_AXI_RDATA <= {{16{core_action2[15]}}, core_action2};
                        8'h30: S_AXI_RDATA <= {24'd0, core_action_crc};
                        8'h34: S_AXI_RDATA <= HW_ID_VAL;
                        8'h38: S_AXI_RDATA <= CLK_FREQ_HZ;
                        8'h40: S_AXI_RDATA <= {{16{cfg_max_v[15]}}, cfg_max_v};
                        8'h44: S_AXI_RDATA <= {{16{cfg_max_w[15]}}, cfg_max_w};
                        default: S_AXI_RDATA <= 32'h00000000;
                    endcase
                end
            end else if (S_AXI_RVALID && S_AXI_RREADY) begin
                S_AXI_RVALID <= 1'b0;
            end
        end
    end

endmodule
`default_nettype wire
