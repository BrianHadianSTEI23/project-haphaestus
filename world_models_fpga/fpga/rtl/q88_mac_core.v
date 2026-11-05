// ============================================================================
// Module: q88_mac_core.v
// Project: Project Hephaestus - World Models Accelerator on AMD Kria KR260
// Author: Brian A. Hadian
//
// Description:
// Signed Q8.8 Fixed-Point Vector Multiply-Accumulate (MAC) DSP Engine.
// Format: 16-bit signed [15: sign, 14:8: 7 integer bits, 7:0: 8 fractional bits]
// Performs: Y = Sat16((Sum(A_i * B_i) >>> 8) + Bias)
// ============================================================================

`timescale 1ns / 1ps

module q88_mac_core (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clr,          // Clear accumulator
    input  wire        en,           // Enable MAC step
    input  wire signed [15:0] op_a,  // Input activation (Q8.8)
    input  wire signed [15:0] op_b,  // Weight (Q8.8)
    input  wire signed [15:0] bias,  // Layer bias (Q8.8)
    input  wire        add_bias,     // Apply bias and finalize
    input  wire        apply_relu,   // Apply ReLU non-linearity
    output reg  signed [15:0] result,// Saturated Q8.8 result
    output reg         valid
);

    // 32-bit signed accumulator for Q16.16 product sum
    reg signed [31:0] accum;
    wire signed [31:0] mult_prod;

    // DSP48E2 primitive multiplication: 16-bit x 16-bit -> 32-bit signed
    assign mult_prod = op_a * op_b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum  <= 32'sd0;
            result <= 16'sd0;
            valid  <= 1'b0;
        end else begin
            valid <= 1'b0;

            if (clr) begin
                accum <= 32'sd0;
            end else if (en) begin
                // Accumulate 32-bit product sum
                accum <= accum + mult_prod;
            end

            if (add_bias) begin
                // Finalize step:
                // 1. Shift right 8 bits (Q16.16 -> Q8.8)
                // 2. Add bias
                // 3. Saturate to [-32768, 32767]
                // 4. Optional ReLU
                reg signed [31:0] scaled_sum;
                reg signed [31:0] biased_sum;
                reg signed [15:0] sat_val;

                scaled_sum = accum >>> 8;
                biased_sum = scaled_sum + {{16{bias[15]}}, bias};

                // Signed 16-bit saturation
                if (biased_sum > 32'sd32767) begin
                    sat_val = 16'sd32767;
                end else if (biased_sum < -32'sd32768) begin
                    sat_val = -16'sd32768;
                end else begin
                    sat_val = biased_sum[15:0];
                end

                // Non-linear activation
                if (apply_relu) begin
                    result <= (sat_val > 16'sd0) ? sat_val : 16'sd0;
                end else begin
                    result <= sat_val;
                end

                valid <= 1'b1;
            end
        end
    end

endmodule
