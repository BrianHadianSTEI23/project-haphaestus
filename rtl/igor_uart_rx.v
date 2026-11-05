`timescale 1ns/1ps
`default_nettype none

// Robust UART Receiver for AMD Kria KR260 / Project Hephaestus
// Features: 2-stage input synchronizer, mid-bit sampling, framing error detection
module igor_uart_rx #(
    parameter [15:0] CLKS_PER_BIT = 16'd54 // Default: 50MHz / 921600 baud = 54.25
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output reg        valid,
    output reg        error,
    output reg  [7:0] data
);
    reg meta, sync;
    reg [1:0] state;
    reg [15:0] count;
    reg [2:0] bit_index;
    reg [7:0] shift;

    // Double-flop synchronizer for asynchronous RX input
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            meta <= 1'b1;
            sync <= 1'b1;
        end else begin
            meta <= rx;
            sync <= meta;
        end
    end

    localparam IDLE  = 2'd0;
    localparam START = 2'd1;
    localparam DATA  = 2'd2;
    localparam STOP  = 2'd3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            count     <= 16'd0;
            bit_index <= 3'd0;
            shift     <= 8'd0;
            data      <= 8'd0;
            valid     <= 1'b0;
            error     <= 1'b0;
        end else begin
            valid <= 1'b0;
            error <= 1'b0;

            case (state)
                IDLE: begin
                    if (!sync) begin // Falling edge: start bit detected
                        state <= START;
                        count <= (CLKS_PER_BIT / 16'd2) - 16'd1; // Wait half bit period
                    end
                end

                START: begin
                    if (count != 16'd0) begin
                        count <= count - 1'b1;
                    end else if (!sync) begin // Verify start bit is still low
                        count     <= CLKS_PER_BIT - 1'b1;
                        bit_index <= 3'd0;
                        state     <= DATA;
                    end else begin
                        state     <= IDLE; // False start glitch
                    end
                end

                DATA: begin
                    if (count != 16'd0) begin
                        count <= count - 1'b1;
                    end else begin
                        shift[bit_index] <= sync;
                        count <= CLKS_PER_BIT - 1'b1;
                        if (bit_index == 3'd7) begin
                            state <= STOP;
                        end else begin
                            bit_index <= bit_index + 1'b1;
                        end
                    end
                end

                STOP: begin
                    if (count != 16'd0) begin
                        count <= count - 1'b1;
                    end else begin
                        state <= IDLE;
                        if (sync) begin // Stop bit should be high
                            valid <= 1'b1;
                            data  <= shift;
                        end else begin
                            error <= 1'b1; // Framing error
                        end
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
