`timescale 1ns/1ps
`default_nettype none

// UART Protocol Framing Engine for AMD Kria KR260 / Project Hephaestus
// Author: Brian A. Hadian
//
// Implements the verified wire-speed binary protocol:
// Request Frame (153 bytes):
//   [0x5A, 0xA5, 0x01, 0x94] + [74 x int16 LE Q8.8] + [CRC-8 Maxim]
// Response Frame (11 bytes):
//   [0x5A, 0xA5, 0x02, 0x06] + [3 x int16 LE Q8.8] + [CRC-8 Maxim]
//
// Features:
// - Non-saturating audit statistics counters
// - Range checking: 64 LiDAR beams in [0, 256], 10 kinematic/IMU features in [-256, 256]
// - Zero-latency fail-closed E-Stop handling
// - CRC-8 Maxim validation: x^8 + x^5 + x^4 + 1 (poly 0x8C reversed)

module igor_kr260_policy_uart #(
    parameter integer CLK_HZ = 50000000,
    parameter integer BAUD   = 921600,
    parameter integer RX_TIMEOUT_CYCLES     = 1000000,
    parameter integer RESULT_TIMEOUT_CYCLES = 1000000,
    parameter integer INFER_TIMEOUT_CYCLES  = 100000,
    parameter BANK0_FILE = "rtl/mem/bank0.mem",
    parameter BANK1_FILE = "rtl/mem/bank1.mem",
    parameter BANK2_FILE = "rtl/mem/bank2.mem",
    parameter BANK3_FILE = "rtl/mem/bank3.mem",
    parameter BANK4_FILE = "rtl/mem/bank4.mem",
    parameter BANK5_FILE = "rtl/mem/bank5.mem",
    parameter BANK6_FILE = "rtl/mem/bank6.mem",
    parameter BANK7_FILE = "rtl/mem/bank7.mem"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        uart_rx,
    input  wire        estop,
    output wire        uart_tx,
    output wire [3:0]  debug_state,
    output reg  [7:0]  error_code,
    output wire [15:0] error_flags,
    output reg  [31:0] accepted_frames,
    output reg  [31:0] rejected_frames,
    output reg  [31:0] last_latency_cycles,
    output wire        safety_permit,
    output reg         response_valid,
    output wire signed [15:0] action0,
    output wire signed [15:0] action1,
    output wire signed [15:0] action2
);
    localparam integer BIT_CYCLES = (CLK_HZ + BAUD / 2) / BAUD;

    localparam RX            = 4'd0;
    localparam LOAD          = 4'd1;
    localparam START         = 4'd2;
    localparam RUN           = 4'd3;
    localparam CAP0          = 4'd4;
    localparam CAP1          = 4'd5;
    localparam CAP2          = 4'd6;
    localparam TX            = 4'd7;
    localparam TX_START_LAST = 4'd8;
    localparam TX_END_LAST   = 4'd9;

    reg estop_meta, estop_sync, estop_previous;
    wire estop_active = estop | estop_sync;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            estop_meta <= 1'b0;
            estop_sync <= 1'b0;
        end else begin
            estop_meta <= estop;
            estop_sync <= estop_meta;
        end
    end

    wire rx_valid, rx_error;
    wire [7:0] rx_data;
    reg tx_send, tx_cancel;
    reg [7:0] tx_data;
    wire tx_busy;

    igor_uart_rx #(.CLKS_PER_BIT(BIT_CYCLES[15:0])) receiver (
        .clk(clk),
        .rst_n(rst_n),
        .rx(uart_rx),
        .valid(rx_valid),
        .error(rx_error),
        .data(rx_data)
    );

    igor_uart_tx #(.CLKS_PER_BIT(BIT_CYCLES[15:0])) transmitter (
        .clk(clk),
        .rst_n(rst_n && !estop_active && !tx_cancel),
        .send(tx_send),
        .data(tx_data),
        .tx(uart_tx),
        .busy(tx_busy)
    );

    function [7:0] crc8;
        input [7:0] previous, data;
        reg [7:0] c;
        integer j;
        begin
            c = previous ^ data;
            for (j = 0; j < 8; j = j + 1)
                c = c[0] ? (c >> 1) ^ 8'h8c : (c >> 1);
            crc8 = c;
        end
    endfunction

    function [7:0] action_crc;
        input [15:0] a, b, c;
        reg [7:0] check;
        begin
            check = crc8(8'd0, 8'h02);
            check = crc8(check, 8'h06);
            check = crc8(check, a[7:0]);
            check = crc8(check, a[15:8]);
            check = crc8(check, b[7:0]);
            check = crc8(check, b[15:8]);
            check = crc8(check, c[7:0]);
            check = crc8(check, c[15:8]);
            action_crc = check;
        end
    endfunction

    reg [3:0] mode;
    reg [11:0] error_history;
    assign error_flags = {3'b000, error_history, 1'b0};

    reg [7:0] position, crc, low_byte;
    reg invalid_observation;
    reg signed [15:0] frame_observation [0:73];
    wire signed [15:0] received_word = {rx_data, low_byte};
    wire [7:0] payload_index = position - 8'd4;

    reg [6:0] load_index;
    reg core_load, core_start, core_abort;
    reg [6:0] core_input_index;
    reg signed [15:0] core_input_value;
    reg [1:0] core_output_index;
    wire signed [15:0] core_output_value;
    wire core_busy, core_done, core_fault, core_saturated;
    wire [3:0] core_error;
    wire [31:0] core_latency;

    igor_kr260_policy_core #(
        .BANK0_FILE(BANK0_FILE),
        .BANK1_FILE(BANK1_FILE),
        .BANK2_FILE(BANK2_FILE),
        .BANK3_FILE(BANK3_FILE),
        .BANK4_FILE(BANK4_FILE),
        .BANK5_FILE(BANK5_FILE),
        .BANK6_FILE(BANK6_FILE),
        .BANK7_FILE(BANK7_FILE)
    ) policy_engine (
        .clk(clk),
        .rst_n(rst_n),
        .load_input(core_load),
        .input_index(core_input_index),
        .input_value(core_input_value),
        .start(core_start),
        .abort(core_abort | estop_active),
        .output_index(core_output_index),
        .output_value(core_output_value),
        .busy(core_busy),
        .done(core_done),
        .fault(core_fault),
        .error_code(core_error),
        .latency_cycles(core_latency),
        .saturated(core_saturated)
    );

    reg signed [15:0] sample0, sample1, sample2;
    reg result_valid;
    reg [31:0] rx_timer, infer_timer, result_timer;
    reg [3:0] tx_index;
    reg [7:0] response_crc;

    assign safety_permit = result_valid && !estop_active && !core_fault;
    assign action0 = safety_permit ? sample0 : 16'sd0;
    assign action1 = safety_permit ? sample1 : 16'sd0;
    assign action2 = safety_permit ? sample2 : 16'sd0;
    assign debug_state = estop_active ? 4'hf : (core_fault ? 4'he : mode);

    task reject;
        input [7:0] code;
        begin
            error_code    <= code;
            error_history <= error_history | (12'h1 << (code[3:0] - 1'b1));
            if (rejected_frames != 32'hffffffff)
                rejected_frames <= rejected_frames + 1'b1;
            result_valid <= 1'b0;
            position     <= 8'd0;
            mode         <= RX;
            rx_timer     <= 32'd0;
            infer_timer  <= 32'd0;
            core_abort   <= 1'b1;
            core_start   <= 1'b0;
            core_load    <= 1'b0;
            tx_send      <= 1'b0;
            tx_cancel    <= 1'b1;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode                <= RX;
            position            <= 8'd0;
            crc                 <= 8'd0;
            low_byte            <= 8'd0;
            invalid_observation <= 1'b0;
            load_index          <= 7'd0;
            core_load           <= 1'b0;
            core_start          <= 1'b0;
            core_abort          <= 1'b0;
            core_input_index    <= 7'd0;
            core_input_value    <= 16'sd0;
            core_output_index   <= 2'd0;
            sample0             <= 16'sd0;
            sample1             <= 16'sd0;
            sample2             <= 16'sd0;
            result_valid        <= 1'b0;
            rx_timer            <= 32'd0;
            infer_timer         <= 32'd0;
            result_timer        <= 32'd0;
            tx_index            <= 4'd0;
            response_crc        <= 8'd0;
            tx_send             <= 1'b0;
            tx_data             <= 8'd0;
            tx_cancel           <= 1'b0;
            error_code          <= 8'd0;
            error_history       <= 12'd0;
            accepted_frames     <= 32'd0;
            rejected_frames     <= 32'd0;
            last_latency_cycles <= 32'd0;
            response_valid      <= 1'b0;
            estop_previous      <= 1'b0;
        end else begin
            core_load      <= 1'b0;
            core_start     <= 1'b0;
            core_abort     <= 1'b0;
            tx_send        <= 1'b0;
            tx_cancel      <= 1'b0;
            response_valid <= 1'b0;
            estop_previous <= estop_active;

            if (result_valid) result_timer <= result_timer + 1'b1;
            if (mode == RUN)  infer_timer  <= infer_timer + 1'b1;
            if (position != 8'd0) rx_timer <= rx_timer + 1'b1;

            if (estop_active) begin
                result_valid <= 1'b0;
                position     <= 8'd0;
                mode         <= RX;
                rx_timer     <= 32'd0;
                core_abort   <= 1'b1;
                tx_cancel    <= 1'b1;
                if (!estop_previous) begin
                    error_code    <= 8'd1;
                    error_history <= error_history | 12'h001;
                    if (rejected_frames != 32'hffffffff)
                        rejected_frames <= rejected_frames + 1'b1;
                end
            end else if (core_fault && mode != RX) begin
                reject(8'd10);
            end else if (rx_error) begin
                reject(8'd2);
            end else if (rx_valid && mode != RX) begin
                reject(8'd9);
            end else if (mode == RUN && infer_timer >= INFER_TIMEOUT_CYCLES - 1) begin
                reject(8'd12);
            end else if (position != 8'd0 && !rx_valid && rx_timer >= RX_TIMEOUT_CYCLES - 1) begin
                reject(8'd7);
            end else if (result_valid && result_timer >= RESULT_TIMEOUT_CYCLES - 1) begin
                reject(8'd11);
            end else begin
                if (rx_valid) begin
                    rx_timer <= 32'd0;
                    case (position)
                        8'd0: if (rx_data == 8'h5a) begin
                            position     <= 8'd1;
                            result_valid <= 1'b0;
                        end

                        8'd1: if (rx_data == 8'ha5) begin
                            position <= 8'd2;
                        end else if (rx_data != 8'h5a) begin
                            reject(8'd3);
                        end

                        8'd2: if (rx_data == 8'h01) begin
                            crc      <= crc8(8'd0, rx_data);
                            position <= 8'd3;
                        end else begin
                            reject(8'd4);
                        end

                        8'd3: if (rx_data == 8'h94) begin
                            crc                 <= crc8(crc, rx_data);
                            position            <= 8'd4;
                            invalid_observation <= 1'b0;
                        end else begin
                            reject(8'd5);
                        end

                        8'd152: begin
                            position <= 8'd0;
                            if (rx_data != crc) begin
                                reject(8'd6);
                            end else if (invalid_observation) begin
                                reject(8'd8);
                            end else begin
                                if (accepted_frames != 32'hffffffff)
                                    accepted_frames <= accepted_frames + 1'b1;
                                load_index <= 7'd0;
                                mode       <= LOAD;
                            end
                        end

                        default: begin
                            crc      <= crc8(crc, rx_data);
                            position <= position + 1'b1;
                            if (!payload_index[0]) begin
                                low_byte <= rx_data;
                            end else begin
                                frame_observation[payload_index[7:1]] <= received_word;
                                if (payload_index[7:1] < 64) begin
                                    // 64 Frontal LiDAR beams must be in [0, 256] (0.0 to 1.0)
                                    if (received_word < 16'sd0 || received_word > 16'sd256)
                                        invalid_observation <= 1'b1;
                                end else begin
                                    // 10 Kinematic/IMU observations must be in [-256, 256]
                                    if (received_word < -16'sd256 || received_word > 16'sd256)
                                        invalid_observation <= 1'b1;
                                end
                            end
                        end
                    endcase
                end

                case (mode)
                    RX: begin end

                    LOAD: begin
                        core_load        <= 1'b1;
                        core_input_index <= load_index;
                        core_input_value <= frame_observation[load_index];
                        if (load_index == 7'd73) begin
                            mode <= START;
                        end else begin
                            load_index <= load_index + 1'b1;
                        end
                    end

                    START: begin
                        core_start  <= 1'b1;
                        mode        <= RUN;
                        infer_timer <= 32'd0;
                    end

                    RUN: if (core_done) begin
                        last_latency_cycles <= core_latency;
                        core_output_index   <= 2'd0;
                        mode                <= CAP0;
                    end

                    CAP0: begin
                        sample0           <= core_output_value;
                        core_output_index <= 2'd1;
                        mode              <= CAP1;
                    end

                    CAP1: begin
                        sample1           <= core_output_value;
                        core_output_index <= 2'd2;
                        mode              <= CAP2;
                    end

                    CAP2: begin
                        sample2        <= core_output_value;
                        response_crc   <= action_crc(sample0, sample1, core_output_value);
                        result_valid   <= 1'b1;
                        result_timer   <= 32'd0;
                        response_valid <= 1'b1;
                        tx_index       <= 4'd0;
                        mode           <= TX;
                    end

                    TX: if (!tx_busy && !tx_send) begin
                        case (tx_index)
                            4'd0: tx_data <= 8'h5a;
                            4'd1: tx_data <= 8'ha5;
                            4'd2: tx_data <= 8'h02;
                            4'd3: tx_data <= 8'h06;
                            4'd4: tx_data <= sample0[7:0];
                            4'd5: tx_data <= sample0[15:8];
                            4'd6: tx_data <= sample1[7:0];
                            4'd7: tx_data <= sample1[15:8];
                            4'd8: tx_data <= sample2[7:0];
                            4'd9: tx_data <= sample2[15:8];
                            default: tx_data <= response_crc;
                        endcase
                        tx_send <= 1'b1;
                        if (tx_index == 4'd10) begin
                            mode <= TX_START_LAST;
                        end else begin
                            tx_index <= tx_index + 1'b1;
                        end
                    end

                    TX_START_LAST: if (tx_busy) mode <= TX_END_LAST;
                    TX_END_LAST:   if (!tx_busy) mode <= RX;

                    default: reject(8'd10);
                endcase
            end
        end
    end
endmodule
`default_nettype wire
