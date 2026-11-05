`timescale 1ns/1ps
`default_nettype none

// Robust UART Transmitter for AMD Kria KR260 / Project Hephaestus
module igor_uart_tx #(
    parameter [15:0] CLKS_PER_BIT = 16'd54 // Default: 50MHz / 921600 baud = 54.25
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       send,
    input  wire [7:0] data,
    output wire       tx,
    output reg        busy
);
    reg [9:0] shift;
    reg [15:0] count;
    reg [3:0] bit_index;

    assign tx = busy ? shift[0] : 1'b1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift     <= 10'h3ff;
            count     <= 16'd0;
            bit_index <= 4'd0;
            busy      <= 1'b0;
        end else if (!busy) begin
            if (send) begin
                shift     <= {1'b1, data, 1'b0}; // Stop bit (1), Data bits (8), Start bit (0)
                busy      <= 1'b1;
                count     <= CLKS_PER_BIT - 1'b1;
                bit_index <= 4'd0;
            end
        end else if (count != 16'd0) begin
            count <= count - 1'b1;
        end else begin
            count <= CLKS_PER_BIT - 1'b1;
            shift <= {1'b1, shift[9:1]};
            if (bit_index == 4'd9) begin
                busy <= 1'b0;
            end else begin
                bit_index <= bit_index + 1'b1;
            end
        end
    end
endmodule
`default_nettype wire
