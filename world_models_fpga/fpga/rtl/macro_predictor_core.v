// ============================================================================
// Module: macro_predictor_core.v
// Project: Project Hephaestus - World Models Accelerator on AMD Kria KR260
// Author: Brian A. Hadian
//
// Description:
// Synthesizable Action-Conditioned Residual Macro Transition Predictor
// with Integrated Safety Barrier Head (Option B Reconstructed Architecture).
// Implements:
//   - FC1: 19 -> 32 (ReLU)
//   - FC2: 32 -> 32 (ReLU)
//   - Delta Head: 32 -> 16 (Linear)
//   - Residual Add: Next_S = S_k + Delta_S
//   - Collision Barrier Head: 32 -> 1 (Piecewise Linear Sigmoid)
// Precision: Signed Q8.8 fixed-point arithmetic with DSP48E2 acceleration.
// ============================================================================

`timescale 1ns / 1ps

module macro_predictor_core #(
    parameter VEC_PATH = "world_models_fpga/fpga/vectors/"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    // Input Vector: 16 words state S_k + 3 words candidate action A_k = 19 words
    input  wire signed [15:0] in_vec [0:18],

    output reg  signed [15:0] out_next_state [0:15],
    output reg  signed [15:0] out_col_prob,
    output reg         done
);

    // On-Chip BRAM Weight and Bias ROMs
    reg signed [15:0] rom_w_fc1   [0:607];  // 32 x 19 = 608
    reg signed [15:0] rom_b_fc1   [0:31];   // 32
    reg signed [15:0] rom_w_fc2   [0:1023]; // 32 x 32 = 1024
    reg signed [15:0] rom_b_fc2   [0:31];   // 32
    reg signed [15:0] rom_w_delta [0:511];  // 16 x 32 = 512
    reg signed [15:0] rom_b_delta [0:15];   // 16
    reg signed [15:0] rom_w_col   [0:31];   // 1 x 32 = 32
    reg signed [15:0] rom_b_col   [0:0];    // 1

    initial begin
        $readmemh({VEC_PATH, "macro_predictor_fc1_weight.hex"}, rom_w_fc1);
        $readmemh({VEC_PATH, "macro_predictor_fc1_bias.hex"},   rom_b_fc1);
        $readmemh({VEC_PATH, "macro_predictor_fc2_weight.hex"}, rom_w_fc2);
        $readmemh({VEC_PATH, "macro_predictor_fc2_bias.hex"},   rom_b_fc2);
        $readmemh({VEC_PATH, "macro_predictor_delta_head_weight.hex"}, rom_w_delta);
        $readmemh({VEC_PATH, "macro_predictor_delta_head_bias.hex"},   rom_b_delta);
        $readmemh({VEC_PATH, "macro_predictor_col_head_weight.hex"},   rom_w_col);
        $readmemh({VEC_PATH, "macro_predictor_col_head_bias.hex"},     rom_b_col);
    end

    // Intermediate Activation Registers
    reg signed [15:0] h1 [0:31];
    reg signed [15:0] h2 [0:31];
    reg signed [15:0] delta [0:15];
    reg signed [15:0] col_logit;

    // FSM States
    localparam S_IDLE     = 3'd0;
    localparam S_FC1      = 3'd1;
    localparam S_FC2      = 3'd2;
    localparam S_DELTA    = 3'd3;
    localparam S_COL      = 3'd4;
    localparam S_RESIDUAL = 3'd5;
    localparam S_DONE     = 3'd6;

    reg [2:0] state;
    reg [5:0] out_idx;
    reg [5:0] in_idx;
    reg signed [31:0] accum;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            out_idx <= 6'd0;
            in_idx <= 6'd0;
            accum <= 32'sd0;
            done <= 1'b0;
            out_col_prob <= 16'sd0;
            for (i = 0; i < 16; i = i + 1) out_next_state[i] <= 16'sd0;
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

                // FC1 Layer: 19 inputs -> 32 outputs
                S_FC1: begin
                    if (in_idx == 18) begin
                        reg signed [31:0] full_sum;
                        reg signed [31:0] scaled;
                        reg signed [31:0] biased;
                        reg signed [15:0] sat;
                        full_sum = accum + (in_vec[18] * rom_w_fc1[out_idx * 19 + 18]);
                        scaled = (full_sum + 32'sd128) >>> 8;
                        biased = scaled + {{16{rom_b_fc1[out_idx][15]}}, rom_b_fc1[out_idx]};
                        sat = (biased > 32'sd32767) ? 16'sd32767 : ((biased < -32'sd32768) ? -16'sd32768 : biased[15:0]);
                        // ReLU non-linearity
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
                        accum <= accum + (in_vec[in_idx] * rom_w_fc1[out_idx * 19 + in_idx]);
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
                        // ReLU non-linearity
                        h2[out_idx] <= (sat > 16'sd0) ? sat : 16'sd0;

                        accum <= 32'sd0;
                        in_idx <= 6'd0;
                        if (out_idx == 31) begin
                            state <= S_DELTA;
                            out_idx <= 6'd0;
                        end else begin
                            out_idx <= out_idx + 1'b1;
                        end
                    end else begin
                        accum <= accum + (h1[in_idx] * rom_w_fc2[out_idx * 32 + in_idx]);
                        in_idx <= in_idx + 1'b1;
                    end
                end

                // Delta Head: 32 inputs -> 16 outputs (Linear)
                S_DELTA: begin
                    if (in_idx == 31) begin
                        reg signed [31:0] full_sum;
                        reg signed [31:0] scaled;
                        reg signed [31:0] biased;
                        full_sum = accum + (h2[31] * rom_w_delta[out_idx * 32 + 31]);
                        scaled = (full_sum + 32'sd128) >>> 8;
                        biased = scaled + {{16{rom_b_delta[out_idx][15]}}, rom_b_delta[out_idx]};
                        delta[out_idx] <= (biased > 32'sd32767) ? 16'sd32767 : ((biased < -32'sd32768) ? -16'sd32768 : biased[15:0]);

                        accum <= 32'sd0;
                        in_idx <= 6'd0;
                        if (out_idx == 15) begin
                            state <= S_COL;
                            out_idx <= 6'd0;
                        end else begin
                            out_idx <= out_idx + 1'b1;
                        end
                    end else begin
                        accum <= accum + (h2[in_idx] * rom_w_delta[out_idx * 32 + in_idx]);
                        in_idx <= in_idx + 1'b1;
                    end
                end

                // Collision Head: 32 inputs -> 1 output
                S_COL: begin
                    if (in_idx == 31) begin
                        reg signed [31:0] full_sum;
                        reg signed [31:0] scaled;
                        reg signed [31:0] biased;
                        reg signed [15:0] logit;
                        reg signed [15:0] pwl_sig;
                        full_sum = accum + (h2[31] * rom_w_col[31]);
                        scaled = (full_sum + 32'sd128) >>> 8;
                        biased = scaled + {{16{rom_b_col[0][15]}}, rom_b_col[0]};
                        logit = (biased > 32'sd32767) ? 16'sd32767 : ((biased < -32'sd32768) ? -16'sd32768 : biased[15:0]);

                        // PWL Sigmoid: 128 + (logit >>> 2), saturated to [0, 256]
                        pwl_sig = 16'sd128 + (logit >>> 2);
                        if (pwl_sig > 16'sd256) pwl_sig = 16'sd256;
                        else if (pwl_sig < 16'sd0) pwl_sig = 16'sd0;
                        out_col_prob <= pwl_sig;

                        accum <= 32'sd0;
                        in_idx <= 6'd0;
                        state <= S_RESIDUAL;
                    end else begin
                        accum <= accum + (h2[in_idx] * rom_w_col[in_idx]);
                        in_idx <= in_idx + 1'b1;
                    end
                end

                // Residual Addition: Next_S = S_k + Delta_S
                S_RESIDUAL: begin
                    for (i = 0; i < 16; i = i + 1) begin
                        reg signed [31:0] sum;
                        sum = {{16{in_vec[i][15]}}, in_vec[i]} + {{16{delta[i][15]}}, delta[i]};
                        out_next_state[i] <= (sum > 32'sd32767) ? 16'sd32767 : ((sum < -32'sd32768) ? -16'sd32768 : sum[15:0]);
                    end
                    state <= S_DONE;
                end

                S_DONE: begin
                    done <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
