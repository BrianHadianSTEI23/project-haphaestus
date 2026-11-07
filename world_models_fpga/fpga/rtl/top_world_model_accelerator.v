// ============================================================================
// Module: top_world_model_accelerator.v
// Project: Project Hephaestus - World Models Accelerator on AMD Kria KR260
// Author: Brian A. Hadian
//
// Description:
// Top-Level Production Synthesizable AXI4-Lite & AXI4-Stream Accelerator Core
// for the Reconstructed Action-Conditioned World Model Architecture.
//
// Integrates:
//   - Tier 2: Latent Register Bank (16 words Q8.8)
//   - Tier 3: Macro Predictor Core (Residual Foresight S_{t+K} = S_t + T(S_t, A))
//   - Tier 4: Safety Barrier Head (Collision probability B(S, A))
//   - Tier 5: Micro Steering Policy Core (10 Hz Mecanum wheel actuation [vx, vy, wz])
//   - AXI4-Lite Slave Interface for PS register control, status, and feature I/O
//   - AXI4-Stream Slave Interface for DMA/Camera streaming feature ingress
//   - Hardware Cycle Counter for benchmarking latency at 200 MHz
// ============================================================================

`timescale 1ns / 1ps

module top_world_model_accelerator #(
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 10,
    parameter VEC_PATH           = "world_models_fpga/fpga/vectors/"
)(
    // Global Clock & Reset (200 MHz from PS FCLK_CLK0)
    input  wire                              clk,
    input  wire                              rst_n,

    // ========================================================================
    // AXI4-Lite Slave Interface
    // ========================================================================
    // Write Address Channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    input  wire [2:0]                        s_axi_awprot,
    input  wire                              s_axi_awvalid,
    output reg                               s_axi_awready,

    // Write Data Channel
    input  wire [C_S_AXI_DATA_WIDTH-1:0]     s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                              s_axi_wvalid,
    output reg                               s_axi_wready,

    // Write Response Channel
    output reg  [1:0]                        s_axi_bresp,
    output reg                               s_axi_bvalid,
    input  wire                              s_axi_bready,

    // Read Address Channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr,
    input  wire [2:0]                        s_axi_arprot,
    input  wire                              s_axi_arvalid,
    output reg                               s_axi_arready,

    // Read Data Channel
    output reg  [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    output reg  [1:0]                        s_axi_rresp,
    output reg                               s_axi_rvalid,
    input  wire                              s_axi_rready,

    // ========================================================================
    // AXI4-Stream Slave Interface (Camera / DMA Streaming Ingress)
    // ========================================================================
    input  wire [15:0]                       s_axis_tdata,
    input  wire                              s_axis_tvalid,
    output wire                              s_axis_tready,
    input  wire                              s_axis_tlast,

    // Interrupt Line to ARM Cortex-A53 GIC
    output wire                              interrupt
);

    // ========================================================================
    // Register Map & Feature Buffers
    // ========================================================================
    // 0x000: Control (bit 0: start, bit 1: soft_reset, bits [3:2]: mode)
    // 0x004: Status  (bit 0: busy, bit 1: done)
    // 0x008: Performance Cycle Counter
    // 0x010 - 0x04C: Input State S_k (16 words, Q8.8, 1 word per 32-bit register)
    // 0x050 - 0x058: Candidate Action A_k (3 words, Q8.8)
    // 0x060 - 0x09C: Target Subgoal S_subgoal (16 words, Q8.8)
    // 0x0A0 - 0x0DC: Output Next State Next_S (16 words, Q8.8)
    // 0x0E0:         Output Collision Probability (1 word, Q8.8)
    // 0x0E4 - 0x0EC: Output Primitive Action [vx, vy, wz] (3 words, Q8.8)

    reg [31:0] reg_ctrl;
    reg [31:0] reg_status;
    reg [31:0] reg_perf_cycles;

    reg signed [15:0] in_state_mem   [0:15]; // State S_k
    reg signed [15:0] in_action_mem  [0:2];  // Action A_k
    reg signed [15:0] in_subgoal_mem [0:15]; // Subgoal S_subgoal

    reg signed [15:0] out_state_mem  [0:15]; // Next_S
    reg signed [15:0] out_col_mem;           // Collision Prob
    reg signed [15:0] out_action_mem [0:2];  // Action [vx, vy, wz]

    // Cycle Counter Logic
    reg counting;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_perf_cycles <= 32'd0;
            counting <= 1'b0;
        end else begin
            if (reg_ctrl[0]) begin
                reg_perf_cycles <= 32'd0;
                counting <= 1'b1;
            end else if (counting && reg_status[1]) begin
                counting <= 1'b0;
            end else if (counting) begin
                reg_perf_cycles <= reg_perf_cycles + 1'b1;
            end
        end
    end

    // Subsystem Wire Declarations
    wire signed [15:0] macro_in [0:18];
    genvar g;
    generate
        for (g = 0; g < 16; g = g + 1) begin: GEN_MACRO_S
            assign macro_in[g] = in_state_mem[g];
        end
        for (g = 0; g < 3; g = g + 1) begin: GEN_MACRO_A
            assign macro_in[16 + g] = in_action_mem[g];
        end
    endgenerate

    wire signed [15:0] macro_out_s [0:15];
    wire signed [15:0] macro_out_col;
    wire macro_done;
    reg  macro_start;

    macro_predictor_core #(.VEC_PATH(VEC_PATH)) u_macro_pred (
        .clk(clk),
        .rst_n(rst_n),
        .start(macro_start),
        .in_vec(macro_in),
        .out_next_state(macro_out_s),
        .out_col_prob(macro_out_col),
        .done(macro_done)
    );

    wire signed [15:0] policy_in [0:31];
    generate
        for (g = 0; g < 16; g = g + 1) begin: GEN_POL_S
            assign policy_in[g] = in_state_mem[g];
            assign policy_in[16 + g] = in_subgoal_mem[g];
        end
    endgenerate

    wire signed [15:0] policy_out_act [0:2];
    wire policy_done;
    reg  policy_start;

    micro_steering_core #(.VEC_PATH(VEC_PATH)) u_micro_steer (
        .clk(clk),
        .rst_n(rst_n),
        .start(policy_start),
        .in_vec(policy_in),
        .out_action(policy_out_act),
        .done(policy_done)
    );

    // Accelerator Orchestrator FSM
    localparam FSM_IDLE   = 2'd0;
    localparam FSM_MACRO  = 2'd1;
    localparam FSM_POLICY = 2'd2;
    localparam FSM_FINISH = 2'd3;

    reg [1:0] fsm_state;
    integer idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_state <= FSM_IDLE;
            reg_status <= 32'd0;
            macro_start <= 1'b0;
            policy_start <= 1'b0;
            out_col_mem <= 16'sd0;
            for (idx = 0; idx < 16; idx = idx + 1) out_state_mem[idx] <= 16'sd0;
            for (idx = 0; idx < 3; idx = idx + 1) out_action_mem[idx] <= 16'sd0;
        end else begin
            case (fsm_state)
                FSM_IDLE: begin
                    if (reg_ctrl[0]) begin // Start pulsed
                        reg_status <= 32'b01; // Busy = 1, Done = 0
                        macro_start <= 1'b1;
                        fsm_state <= FSM_MACRO;
                    end
                end

                FSM_MACRO: begin
                    macro_start <= 1'b0;
                    if (macro_done) begin
                        // Latch macro predictions
                        for (idx = 0; idx < 16; idx = idx + 1) out_state_mem[idx] <= macro_out_s[idx];
                        out_col_mem <= macro_out_col;

                        // Chain directly into micro steering policy
                        policy_start <= 1'b1;
                        fsm_state <= FSM_POLICY;
                    end
                end

                FSM_POLICY: begin
                    policy_start <= 1'b0;
                    if (policy_done) begin
                        // Latch micro motor actions
                        for (idx = 0; idx < 3; idx = idx + 1) out_action_mem[idx] <= policy_out_act[idx];
                        fsm_state <= FSM_FINISH;
                    end
                end

                FSM_FINISH: begin
                    reg_status <= 32'b10; // Busy = 0, Done = 1
                    fsm_state <= FSM_IDLE;
                end
            endcase
        end
    end

    assign interrupt = reg_status[1];
    assign s_axis_tready = (fsm_state == FSM_IDLE); // Accept streaming packets when idle

    reg [5:0] axis_word_idx;

    // ========================================================================
    // AXI4-Lite & AXI4-Stream Bus Protocol Implementation
    // ========================================================================
    // AXI Write & Stream Ingress Handshake
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            s_axi_bresp   <= 2'b00;
            reg_ctrl      <= 32'd0;
            axis_word_idx <= 6'd0;
        end else begin
            // Write Address Ready
            if (!s_axi_awready && s_axi_awvalid && s_axi_wvalid) begin
                s_axi_awready <= 1'b1;
                s_axi_wready  <= 1'b1;
            end else begin
                s_axi_awready <= 1'b0;
                s_axi_wready  <= 1'b0;
            end

            // Write Response
            if (s_axi_awready && s_axi_wready) begin
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00; // OKAY
            end else if (s_axi_bready && s_axi_bvalid) begin
                s_axi_bvalid <= 1'b0;
            end

            // Auto-clear start bit and soft_reset bit
            if (reg_ctrl[0]) reg_ctrl[0] <= 1'b0;
            if (reg_ctrl[1]) begin
                reg_ctrl[1] <= 1'b0;
                axis_word_idx <= 6'd0;
            end

            // Memory Mapped Registers Write (AXI4-Lite) or Streaming Packet Ingress (AXI4-Stream)
            if (s_axi_awready && s_axi_wready) begin
                case (s_axi_awaddr[9:2])
                    8'h00: reg_ctrl <= s_axi_wdata;
                    // State inputs (0x010 - 0x04C -> word 4..19)
                    8'h04, 8'h05, 8'h06, 8'h07, 8'h08, 8'h09, 8'h0A, 8'h0B,
                    8'h0C, 8'h0D, 8'h0E, 8'h0F, 8'h10, 8'h11, 8'h12, 8'h13:
                        in_state_mem[s_axi_awaddr[9:2] - 8'h04] <= s_axi_wdata[15:0];
                    // Action inputs (0x050 - 0x058 -> word 20..22)
                    8'h14, 8'h15, 8'h16:
                        in_action_mem[s_axi_awaddr[9:2] - 8'h14] <= s_axi_wdata[15:0];
                    // Subgoal inputs (0x060 - 0x09C -> word 24..39)
                    8'h18, 8'h19, 8'h1A, 8'h1B, 8'h1C, 8'h1D, 8'h1E, 8'h1F,
                    8'h20, 8'h21, 8'h22, 8'h23, 8'h24, 8'h25, 8'h26, 8'h27:
                        in_subgoal_mem[s_axi_awaddr[9:2] - 8'h18] <= s_axi_wdata[15:0];
                endcase
            end else if (s_axis_tvalid && s_axis_tready) begin
                if (axis_word_idx < 6'd16) begin
                    in_state_mem[axis_word_idx] <= s_axis_tdata;
                end else if (axis_word_idx < 6'd19) begin
                    in_action_mem[axis_word_idx - 6'd16] <= s_axis_tdata;
                end else if (axis_word_idx < 6'd35) begin
                    in_subgoal_mem[axis_word_idx - 6'd19] <= s_axis_tdata;
                end

                if (s_axis_tlast || axis_word_idx == 6'd34) begin
                    axis_word_idx <= 6'd0;
                    if (s_axis_tlast && reg_ctrl[2]) begin
                        reg_ctrl[0] <= 1'b1; // Trigger start on stream TLAST when mode bit 2 is set
                    end
                end else begin
                    axis_word_idx <= axis_word_idx + 6'd1;
                end
            end
        end
    end

    // AXI Read Handshake
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rresp   <= 2'b00;
            s_axi_rdata   <= 32'd0;
        end else begin
            if (!s_axi_arready && s_axi_arvalid) begin
                s_axi_arready <= 1'b1;
            end else begin
                s_axi_arready <= 1'b0;
            end

            if (s_axi_arready && s_axi_arvalid) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rresp  <= 2'b00;
                case (s_axi_araddr[9:2])
                    8'h00: s_axi_rdata <= reg_ctrl;
                    8'h01: s_axi_rdata <= reg_status;
                    8'h02: s_axi_rdata <= reg_perf_cycles;
                    // Output next state (0x0A0 - 0x0DC -> word 40..55)
                    8'h28, 8'h29, 8'h2A, 8'h2B, 8'h2C, 8'h2D, 8'h2E, 8'h2F,
                    8'h30, 8'h31, 8'h32, 8'h33, 8'h34, 8'h35, 8'h36, 8'h37:
                        s_axi_rdata <= {{16{out_state_mem[s_axi_araddr[9:2] - 8'h28][15]}}, out_state_mem[s_axi_araddr[9:2] - 8'h28]};
                    // Collision prob (0x0E0 -> word 56)
                    8'h38: s_axi_rdata <= {{16{out_col_mem[15]}}, out_col_mem};
                    // Output Action (0x0E4 - 0x0EC -> word 57..59)
                    8'h39, 8'h3A, 8'h3B:
                        s_axi_rdata <= {{16{out_action_mem[s_axi_araddr[9:2] - 8'h39][15]}}, out_action_mem[s_axi_araddr[9:2] - 8'h39]};
                    default: s_axi_rdata <= 32'd0;
                endcase
            end else if (s_axi_rready && s_axi_rvalid) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
