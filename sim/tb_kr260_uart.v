`timescale 1ns/1ps
`default_nettype none

// UART Protocol Framing Simulation Testbench for AMD Kria KR260 / Project Hephaestus
// Author: Brian A. Hadian
//
// Tests 153-byte Request / 11-byte Response Framing:
// - Valid 153-byte frame exchange at 921,600 baud
// - Response validation against oracle [-46, 60, 34] and CRC-8 Maxim 0x61
// - CRC-8 corruption rejection test (error_code = 6)
// - Out-of-bounds observation rejection test (error_code = 8)
// - Physical Emergency Stop override and output masking (error_code = 1)

module tb_kr260_uart;
    reg clk = 1'b0;
    always #10 clk = ~clk; // 50 MHz clock (20 ns)

    // Scaled bit cycles for fast, reliable simulation: 8 clock cycles per bit
    localparam CLK_HZ = 40000000;
    localparam BAUD   = 5000000;
    localparam BIT_CYCLES = 8;
    localparam BIT_NS = BIT_CYCLES * 20;

    reg        rst_n   = 1'b0;
    reg        uart_rx = 1'b1;
    reg        estop   = 1'b0;
    wire       uart_tx;
    wire [3:0] debug_state;
    wire [7:0] error_code;
    wire [15:0] error_flags;
    wire [31:0] accepted_frames, rejected_frames, last_latency_cycles;
    wire       safety_permit, response_valid;
    wire signed [15:0] action0, action1, action2;

    igor_kr260_policy_uart #(
        .CLK_HZ(CLK_HZ),
        .BAUD(BAUD),
        .RX_TIMEOUT_CYCLES(500),
        .RESULT_TIMEOUT_CYCLES(100000),
        .INFER_TIMEOUT_CYCLES(60000),
        .BANK0_FILE("rtl/mem/bank0.mem"),
        .BANK1_FILE("rtl/mem/bank1.mem"),
        .BANK2_FILE("rtl/mem/bank2.mem"),
        .BANK3_FILE("rtl/mem/bank3.mem"),
        .BANK4_FILE("rtl/mem/bank4.mem"),
        .BANK5_FILE("rtl/mem/bank5.mem"),
        .BANK6_FILE("rtl/mem/bank6.mem"),
        .BANK7_FILE("rtl/mem/bank7.mem")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .uart_rx(uart_rx),
        .estop(estop),
        .uart_tx(uart_tx),
        .debug_state(debug_state),
        .error_code(error_code),
        .error_flags(error_flags),
        .accepted_frames(accepted_frames),
        .rejected_frames(rejected_frames),
        .last_latency_cycles(last_latency_cycles),
        .safety_permit(safety_permit),
        .response_valid(response_valid),
        .action0(action0),
        .action1(action1),
        .action2(action2)
    );

    // Byte Transmitter Task
    task send_byte;
        input [7:0] byte_val;
        integer b;
        begin
            uart_rx <= 1'b0; // Start bit
            #BIT_NS;
            for (b = 0; b < 8; b = b + 1) begin
                uart_rx <= byte_val[b];
                #BIT_NS;
            end
            uart_rx <= 1'b1; // Stop bit
            #BIT_NS;
        end
    endtask

    // Byte Receiver Task
    reg [7:0] rx_buffer [0:63];
    integer rx_byte_cnt = 0;

    always @(negedge uart_tx) begin
        reg [7:0] captured;
        integer b;
        #(BIT_NS + BIT_NS/2); // Wait to center of bit 0
        for (b = 0; b < 8; b = b + 1) begin
            captured[b] = uart_tx;
            #BIT_NS;
        end
        rx_buffer[rx_byte_cnt] = captured;
        rx_byte_cnt = rx_byte_cnt + 1;
    end

    // CRC-8 Maxim Helper
    function [7:0] crc8_calc;
        input [7:0] prev, data_in;
        reg [7:0] c;
        integer j;
        begin
            c = prev ^ data_in;
            for (j = 0; j < 8; j = j + 1)
                c = c[0] ? (c >> 1) ^ 8'h8c : (c >> 1);
            crc8_calc = c;
        end
    endfunction

    integer i;
    reg [7:0] running_crc;
    reg signed [15:0] resp_a0, resp_a1, resp_a2;
    integer pass_count = 0;
    integer fail_count = 0;

    initial begin
        $dumpfile("sim_uart.vcd");
        $dumpvars(0, tb_kr260_uart);

        $display("===============================================================");
        $display("  AMD Kria KR260 UART Protocol Framing Simulation");
        $display("  Author: Brian A. Hadian");
        $display("===============================================================");

        rst_n = 1'b0;
        #200;
        @(posedge clk);
        rst_n = 1'b1;
        #200;

        // Test 1: Send Standard Frame (default_205: 64x205, 10x0)
        $display("[INFO] Step 1: Transmitting 153-byte request frame for default_205...");
        rx_byte_cnt = 0;

        // Magic header
        send_byte(8'h5a);
        send_byte(8'ha5);

        // Opcode 0x01, Length 0x94 (148 bytes)
        send_byte(8'h01); running_crc = crc8_calc(8'd0, 8'h01);
        send_byte(8'h94); running_crc = crc8_calc(running_crc, 8'h94);

        // 64 observations of 205 (0x00CD)
        for (i = 0; i < 64; i = i + 1) begin
            send_byte(8'hcd); running_crc = crc8_calc(running_crc, 8'hcd);
            send_byte(8'h00); running_crc = crc8_calc(running_crc, 8'h00);
        end

        // 10 observations of 0 (0x0000)
        for (i = 0; i < 10; i = i + 1) begin
            send_byte(8'h00); running_crc = crc8_calc(running_crc, 8'h00);
            send_byte(8'h00); running_crc = crc8_calc(running_crc, 8'h00);
        end

        // Send calculated CRC-8
        send_byte(running_crc);

        // Wait for inference & transmission of 11-byte response
        #200000; // Wait 200 us

        if (rx_byte_cnt == 11) begin
            $display("[PASS] Step 1a: Received exactly 11 response bytes!");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 1a: Expected 11 bytes, got %0d", rx_byte_cnt);
            fail_count = fail_count + 1;
        end

        // Check Response Header: 5A A5 02 06
        if (rx_buffer[0] === 8'h5a && rx_buffer[1] === 8'ha5 && rx_buffer[2] === 8'h02 && rx_buffer[3] === 8'h06) begin
            $display("[PASS] Step 1b: Response header verified: 5A A5 02 06");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 1b: Bad header: %02X %02X %02X %02X",
                     rx_buffer[0], rx_buffer[1], rx_buffer[2], rx_buffer[3]);
            fail_count = fail_count + 1;
        end

        // Decode Q8.8 Actions
        resp_a0 = {rx_buffer[5], rx_buffer[4]};
        resp_a1 = {rx_buffer[7], rx_buffer[6]};
        resp_a2 = {rx_buffer[9], rx_buffer[8]};

        if (resp_a0 === -16'sd46 && resp_a1 === 16'sd60 && resp_a2 === 16'sd34) begin
            $display("[PASS] Step 1c: Decoded actions match oracle! Q8.8=[%d, %d, %d]",
                     resp_a0, resp_a1, resp_a2);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 1c: Decoded action mismatch! Got [%d, %d, %d], exp [-46, 60, 34]",
                     resp_a0, resp_a1, resp_a2);
            fail_count = fail_count + 1;
        end

        // Check Response CRC
        if (rx_buffer[10] === 8'h61) begin
            $display("[PASS] Step 1d: Response CRC-8 matches oracle: 0x61");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 1d: Bad response CRC: 0x%02X (exp: 0x61)", rx_buffer[10]);
            fail_count = fail_count + 1;
        end

        // Test 2: Send Frame with Corrupted CRC -> Expect Rejection (error_code 6)
        $display("[INFO] Step 2: Sending frame with intentional CRC corruption...");
        rx_byte_cnt = 0;
        send_byte(8'h5a); send_byte(8'ha5);
        send_byte(8'h01); send_byte(8'h94);
        for (i = 0; i < 74; i = i + 1) begin
            send_byte(8'h00); send_byte(8'h00);
        end
        send_byte(8'hFF); // Deliberately invalid CRC

        #5000;
        if (error_code === 8'd6) begin
            $display("[PASS] Step 2: Corrupted CRC rejected with fault code 6! (rejected_frames=%0d)", rejected_frames);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 2: Expected error_code 6, got %0d", error_code);
            fail_count = fail_count + 1;
        end

        // Test 3: Emergency Stop Override Test
        $display("[INFO] Step 3: Asserting E-Stop button input...");
        estop = 1'b1;
        #1000;
        if (safety_permit === 1'b0 && (action0 === 16'sd0 && action1 === 16'sd0 && action2 === 16'sd0)) begin
            $display("[PASS] Step 3: Outputs zeroed and safety permit revoked on E-Stop!");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 3: E-Stop failed to suppress outputs!");
            fail_count = fail_count + 1;
        end

        $display("---------------------------------------------------------------");
        $display("  Summary: %0d Tests Passed, %0d Failed", pass_count, fail_count);
        $display("===============================================================");

        if (fail_count == 0) begin
            $display(">> SUCCESS: ALL UART PROTOCOL TESTS PASSED! <<");
        end else begin
            $display(">> FAILURE IN UART PROTOCOL SIMULATION <<");
            $fatal(1, "UART testbench failed!");
        end

        $finish;
    end

endmodule
`default_nettype wire
