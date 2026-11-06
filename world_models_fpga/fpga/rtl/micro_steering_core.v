// ============================================================================
// Module: micro_steering_core.v
// Project: Project Hephaestus - World Models Accelerator on AMD Kria KR260
// Author: Brian A. Hadian
//
// Description:
// Synthesizable Closed-Loop Micro Steering Motor Action Policy Core
// (Option B Reconstructed Architecture).
// Implements:
//   - FC1: 32 -> 32 (ReLU)
//   - FC2: 32 -> 32 (ReLU)
//   - Action Head: 32 -> 3 (Piecewise Linear Tanh)
// Precision: Signed Q8.8 fixed-point arithmetic with DSP48E2 acceleration.
// Generates holonomic 3-DOF primitive motor velocities [vx, vy, wz] at 10 Hz.
// ============================================================================

`timescale 1ns / 1ps

module micro_steering_core #(
    parameter VEC_PATH = "world_models_fpga/fpga/vectors/"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    // Input Vector: 16 words current state s_t + 16 words subgoal S_subgoal = 32 words
    input  wire signed [15:0] in_vec [0:31],

    output reg  signed [15:0] out_action [0:2], // vx, vy, wz in Q8.8
    output reg         done
);

    // On-Chip BRAM Weight and Bias ROMs
    reg signed [15:0] rom_w_fc1 [0:1023]; // 32 x 32 = 1024
    reg signed [15:0] rom_b_fc1 [0:31];   // 32
    reg signed [15:0] rom_w_fc2 [0:1023]; // 32 x 32 = 1024
    reg signed [15:0] rom_b_fc2 [0:31];   // 32
    reg signed [15:0] rom_w_act [0:95];   // 3 x 32 = 96
    reg signed [15:0] rom_b_act [0:2];    // 3

    initial begin
        $readmemh({VEC_PATH, "micro_policy_fc1_weight.hex"},         rom_w_fc1);
        $readmemh({VEC_PATH, "micro_policy_fc1_bias.hex"},           rom_b_fc1);
        $readmemh({VEC_PATH, "micro_policy_fc2_weight.hex"},         rom_w_fc2);
        $readmemh({VEC_PATH, "micro_policy_fc2_bias.hex"},           rom_b_fc2);
        $readmemh({VEC_PATH, "micro_policy_action_head_weight.hex"}, rom_w_act);
        $readmemh({VEC_PATH, "micro_policy_action_head_bias.hex"},   rom_b_act);
    end

    // Intermediate Activation Registers
    reg signed [15:0] h1 [0:31];
    reg signed [15:0] h2 [0:31];

    // FSM States
    localparam S_IDLE = 2'd0;
    localparam S_FC1  = 2'd1;
    localparam S_FC2  = 2'd2;
    localparam S_ACT  = 2'd3;

    reg [1:0] state;
    reg [5:0] out_idx;
    reg [5:0] in_idx;
    reg signed [31:0] accum;

    integer i;

    // PWL Tanh Function
    function signed [15:0] pwl_tanh;
        input signed [15:0] x;
        reg signed [15:0] abs_x;
        reg signed [15:0] seg;
        begin
            abs_x = (x >= 16'sd0) ? x : -x;
            if (abs_x < 16'sd128) begin
                pwl_tanh = x;
            end else if (abs_x < 16'sd256) begin
                seg = 16'sd64 + (abs_x >>> 1);
                pwl_tanh = (x >= 16'sd0) ? seg : -seg;
            end else begin
                pwl_tanh = (x >= 16'sd0) ? 16'sd256 : -16'sd256;
            end
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            out_idx <= 6'd0;
            in_idx <= 6'd0;
            accum <= 32'sd0;
            done <= 1'b0;
            for (i = 0; i < 3; i = i + 1) out_action[i] <= 16'sd0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        state <= S_FC1;
                        out_idx <= 6'd0;
                        in_idx <= 6'd0;
                        accum <= 32'sd0;
                    end
                end

                // FC1 Layer: 32 inputs -> 32 outputs
                S_FC1: begin
                    if (in_idx == 31) begin
                        reg signed [31:0] full_sum;
                        reg signed [31:0] scaled;
                        reg signed [31:0] biased;
                        reg signed [15:0] sat;
                        full_sum = accum + (in_vec[31] * rom_w_fc1[out_idx * 32 + 31]);
                        scaled = (full_sum + 32'sd128) >>> 8;
                        biased = scaled + {{16{rom_b_fc1[out_idx][15]}}, rom_b_fc1[out_idx]};
                        sat = (biased > 32'sd32767) ? 16'sd32767 : ((biased < -32'sd32768) ? -16'sd32768 : biased[15:0]);
                        h1[out_idx] <= (sat > 16'sd0) ? sat : 16'sd0;

                        accum <= 32'sd0;
                        in_idx <= 6'd0;
                        if (out_idx == 31) begin
                            state <= S_FC2;
                            out_idx <= 6'd0;
                        end else begin
                            out_idx <= out_idx + 1'b1;
                        end
                    end else begin
                        accum <= accum + (in_vec[in_idx] * rom_w_fc1[out_idx * 32 + in_idx]);
                        in_idx <= in_idx + 1'b1;
                    end
                end

                // FC2 Layer: 32 inputs -> 32 outputs
                S_FC2: begin
                    if (in_idx == 31) begin
                        reg signed [31:0] full_sum;
                        reg signed [31:0] scaled;
                        reg signed [31:0] biased;
                        reg signed [15:0] sat;
                        full_sum = accum + (h1[31] * rom_w_fc2[out_idx * 32 + 31]);
                        scaled = (full_sum + 32'sd128) >>> 8;
                        biased = scaled + {{16{rom_b_fc2[out_idx][15]}}, rom_b_fc2[out_idx]};
                        sat = (biased > 32'sd32767) ? 16'sd32767 : ((biased < -32'sd32768) ? -16'sd32768 : biased[15:0]);
                        h2[out_idx] <= (sat > 16'sd0) ? sat : 16'sd0;

                        accum <= 32'sd0;
                        in_idx <= 6'd0;
                        if (out_idx == 31) begin
                            state <= S_ACT;
                            out_idx <= 6'd0;
                        end else begin
                            out_idx <= out_idx + 1'b1;
                        end
                    end else begin
                        accum <= accum + (h1[in_idx] * rom_w_fc2[out_idx * 32 + in_idx]);
                        in_idx <= in_idx + 1'b1;
                    end
                end

                // Action Head: 32 inputs -> 3 outputs (vx, vy, wz with PWL Tanh)
                S_ACT: begin
                    if (in_idx == 31) begin
                        reg signed [31:0] full_sum;
                        reg signed [31:0] scaled;
                        reg signed [31:0] biased;
                        reg signed [15:0] raw;
                        full_sum = accum + (h2[31] * rom_w_act[out_idx * 32 + 31]);
                        scaled = (full_sum + 32'sd128) >>> 8;
                        biased = scaled + {{16{rom_b_act[out_idx][15]}}, rom_b_act[out_idx]};
                        raw = (biased > 32'sd32767) ? 16'sd32767 : ((biased < -32'sd32768) ? -16'sd32768 : biased[15:0]);
                        out_action[out_idx] <= pwl_tanh(raw);

                        accum <= 32'sd0;
                        in_idx <= 6'd0;
                        if (out_idx == 2) begin
                            done <= 1'b1;
                            state <= S_IDLE;
                        end else begin
                            out_idx <= out_idx + 1'b1;
                        end
                    end else begin
                        accum <= accum + (h2[in_idx] * rom_w_act[out_idx * 32 + in_idx]);
                        in_idx <= in_idx + 1'b1;
                    end
                end
            endcase
        end
    end

endmodule
