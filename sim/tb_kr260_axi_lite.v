`timescale 1ns/1ps
`default_nettype none

// AXI4-Lite Simulation Testbench for AMD Kria KR260 / Project Hephaestus
// Author: Brian A. Hadian
//
// Tests PS <-> PL Memory-Mapped Register Access:
// - Reading HW_ID, Clock Frequency, and Status
// - Writing 74 Q8.8 observation features via AXI4-Lite slave
// - Triggering inference start and polling done status
// - Reading verified output actions (matches default_205 oracle [-46, 60, 34])
// - Testing Monotonic Watchdog lease timeout and automatic fail-closed motor cut-off
// - Testing Software E-Stop assertion and clear

module tb_kr260_axi_lite;
    reg clk = 1'b0;
    always #10 clk = ~clk; // 50 MHz clock

    reg rst_n = 1'b0;

    // AXI4-Lite Master Signals
    reg  [11:0] s_axi_awaddr = 12'd0;
    reg  [2:0]  s_axi_awprot = 3'd0;
    reg         s_axi_awvalid = 1'b0;
    wire        s_axi_awready;

    reg  [31:0] s_axi_wdata = 32'd0;
    reg  [3:0]  s_axi_wstrb = 4'hf;
    reg         s_axi_wvalid = 1'b0;
    wire        s_axi_wready;

    wire [1:0]  s_axi_bresp;
    wire        s_axi_bvalid;
    reg         s_axi_bready = 1'b0;

    reg  [11:0] s_axi_araddr = 12'd0;
    reg  [2:0]  s_axi_arprot = 3'd0;
    reg         s_axi_arvalid = 1'b0;
    wire        s_axi_arready;

    wire [31:0] s_axi_rdata;
    wire [1:0]  s_axi_rresp;
    wire        s_axi_rvalid;
    reg         s_axi_rready = 1'b0;

    // External Top-Level Ports
    reg         pmod_uart_rx = 1'b1;
    wire        pmod_uart_tx;
    reg         estop_btn_n  = 1'b1; // Active-low, normally released
    wire [7:0]  led_status;

    // DUT: Top-Level AMD Kria KR260 Wrapper
    igor_kr260_top #(
        .C_S_AXI_DATA_WIDTH(32),
        .C_S_AXI_ADDR_WIDTH(12),
        .CLK_FREQ_HZ(50000000),
        .UART_BAUD(921600),
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
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awprot(s_axi_awprot),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arprot(s_axi_arprot),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .pmod_uart_rx(pmod_uart_rx),
        .pmod_uart_tx(pmod_uart_tx),
        .estop_btn_n(estop_btn_n),
        .led_status(led_status)
    );

    // --- AXI-Lite Master Tasks ---
    task axi_write;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            s_axi_awaddr  <= addr;
            s_axi_awvalid <= 1'b1;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'hf;
            s_axi_wvalid  <= 1'b1;
            s_axi_bready  <= 1'b1;

            while (!s_axi_awready || !s_axi_wready) @(posedge clk);
            @(posedge clk);
            s_axi_awvalid <= 1'b0;
            s_axi_wvalid  <= 1'b0;

            while (!s_axi_bvalid) @(posedge clk);
            @(posedge clk);
            s_axi_bready <= 1'b0;
        end
    endtask

    task axi_read;
        input  [11:0] addr;
        output [31:0] data;
        begin
            @(posedge clk);
            s_axi_araddr  <= addr;
            s_axi_arvalid <= 1'b1;
            s_axi_rready  <= 1'b1;

            while (!s_axi_arready) @(posedge clk);
            @(posedge clk);
            s_axi_arvalid <= 1'b0;

            while (!s_axi_rvalid) @(posedge clk);
            data = s_axi_rdata;
            @(posedge clk);
            s_axi_rready <= 1'b0;
        end
    endtask

    integer i;
    reg [31:0] rd_data;
    reg signed [15:0] a0, a1, a2;
    integer pass_count = 0;
    integer fail_count = 0;

    initial begin
        $dumpfile("sim_axi_lite.vcd");
        $dumpvars(0, tb_kr260_axi_lite);

        $display("===============================================================");
        $display("  AMD Kria KR260 AXI4-Lite PS-PL Bridge Simulation");
        $display("  Author: Brian A. Hadian");
        $display("===============================================================");

        // Reset Sequence
        rst_n = 1'b0;
        #100;
        @(posedge clk);
        rst_n = 1'b1;
        #100;

        // Test 1: Read Hardware Identification
        axi_read(12'h034, rd_data);
        if (rd_data === 32'h4B523236) begin
            $display("[PASS] Step 1: Read HW_ID = 0x%08X (ASCII 'KR26')", rd_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 1: Expected HW_ID 0x4B523236, got 0x%08X", rd_data);
            fail_count = fail_count + 1;
        end

        // Test 2: Read Core Clock Frequency
        axi_read(12'h038, rd_data);
        if (rd_data === 32'd50000000) begin
            $display("[PASS] Step 2: Read CLK_FREQ = %0d Hz (50 MHz)", rd_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 2: Expected 50000000, got %0d", rd_data);
            fail_count = fail_count + 1;
        end

        // Test 3: Write 74 Observation Features (default_205: 64x205, 10x0)
        $display("[INFO] Step 3: Writing 74 observations over AXI4-Lite slave...");
        for (i = 0; i < 64; i = i + 1) begin
            axi_write(12'h100 + i*4, 32'd205);
        end
        for (i = 64; i < 74; i = i + 1) begin
            axi_write(12'h100 + i*4, 32'd0);
        end

        // Verify readback of observation 0 and observation 64
        axi_read(12'h100, rd_data);
        if (rd_data[15:0] === 16'd205) begin
            $display("[PASS] Step 3a: Observation 0 readback verified: %0d", rd_data[15:0]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 3a: Observation 0 readback mismatch: %0d", rd_data[15:0]);
            fail_count = fail_count + 1;
        end

        // Test 4: Trigger Inference Start (Control Register 0x000 bit 0 = 1)
        $display("[INFO] Step 4: Triggering neural inference via AXI CR (0x000 <= 0x01)...");
        axi_write(12'h000, 32'h00000001);

        // Poll Status Register (0x004) until DONE (bit 1) is asserted
        axi_read(12'h004, rd_data);
        while (!rd_data[1]) begin
            axi_read(12'h004, rd_data);
        end
        $display("[PASS] Step 4: Inference completed! SR = 0x%08X (BUSY=%b, DONE=%b, PERMIT=%b)",
                 rd_data, rd_data[0], rd_data[1], rd_data[3]);
        pass_count = pass_count + 1;

        // Test 5: Read Actions and Latency
        axi_read(12'h024, rd_data); a0 = rd_data[15:0];
        axi_read(12'h028, rd_data); a1 = rd_data[15:0];
        axi_read(12'h02C, rd_data); a2 = rd_data[15:0];
        axi_read(12'h00C, rd_data); // Latency
        $display("[INFO] Step 5: Inference Latency = %0d clock cycles (%0.2f us @ 50MHz)",
                 rd_data, rd_data * 0.02);

        // Expected default_205 actions: [-46, 60, 34]
        if (a0 === -16'sd46 && a1 === 16'sd60 && a2 === 16'sd34) begin
            $display("[PASS] Step 5: Output actions match oracle! Q8.8=[%d, %d, %d] (exp: [-46, 60, 34])",
                     a0, a1, a2);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 5: Action mismatch! Got [%d, %d, %d], exp [-46, 60, 34]",
                     a0, a1, a2);
            fail_count = fail_count + 1;
        end

        // Test 6: Verify CRC-8 Maxim of Output Actions
        axi_read(12'h030, rd_data);
        if (rd_data[7:0] === 8'h61) begin
            $display("[PASS] Step 6: Action CRC-8 Maxim matches oracle: 0x%02X (exp: 0x61)", rd_data[7:0]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 6: Action CRC-8 mismatch: 0x%02X (exp: 0x61)", rd_data[7:0]);
            fail_count = fail_count + 1;
        end

        // Test 7: Monotonic Watchdog Lease Expiration Test
        $display("[INFO] Step 7: Configuring watchdog lease limit to 200 cycles (0x020 <= 200)...");
        axi_write(12'h020, 32'd200);
        // Refresh lease
        axi_write(12'h000, 32'h00000010); // bit 4 = LEASE_REFRESH

        // Check permit is active
        axi_read(12'h004, rd_data);
        if (rd_data[3] === 1'b1) begin
            $display("[PASS] Step 7a: Safety permit active before timeout (SR[3]=1)");
            pass_count = pass_count + 1;
        end

        // Wait for lease timer to expire (>250 cycles = 5000 ns)
        #6000;
        axi_read(12'h004, rd_data);
        if (rd_data[3] === 1'b0 && rd_data[6] === 1'b1) begin
            $display("[PASS] Step 7b: Watchdog lease expired! (LEASE_EXPIRED=1, SAFETY_PERMIT=0)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 7b: Watchdog failed to trip! SR = 0x%08X", rd_data);
            fail_count = fail_count + 1;
        end

        // Verify that output actions are cut off (fail-closed = 0)
        axi_read(12'h024, rd_data); a0 = rd_data[15:0];
        axi_read(12'h028, rd_data); a1 = rd_data[15:0];
        axi_read(12'h02C, rd_data); a2 = rd_data[15:0];
        if (a0 === 16'sd0 && a1 === 16'sd0 && a2 === 16'sd0) begin
            $display("[PASS] Step 7c: Motor outputs zeroed on watchdog timeout (Fail-Closed verified)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 7c: Actions not zeroed during fault! Got [%d, %d, %d]", a0, a1, a2);
            fail_count = fail_count + 1;
        end

        // Verify fault code is 8 (Watchdog Timeout)
        axi_read(12'h008, rd_data);
        if (rd_data[7:0] === 8'd8) begin
            $display("[PASS] Step 7d: Fault code correctly reports 8 (Watchdog Timeout)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 7d: Fault code was %0d (exp 8)", rd_data[7:0]);
            fail_count = fail_count + 1;
        end

        // Test 8: Lease Refresh Heartbeat Recovery
        $display("[INFO] Step 8: Refreshing lease via AXI CR bit 4...");
        axi_write(12'h020, 32'd1000000); // Reset limit to 20 ms
        axi_write(12'h000, 32'h00000010); // bit 4 = LEASE_REFRESH
        #100;
        axi_read(12'h004, rd_data);
        if (rd_data[3] === 1'b1 && rd_data[6] === 1'b0) begin
            $display("[PASS] Step 8: Subsystem recovered after lease refresh! (PERMIT=1, TIMEOUT=0)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 8: Failed recovery after lease refresh! SR = 0x%08X", rd_data);
            fail_count = fail_count + 1;
        end

        // Test 9: Software E-Stop Assertion
        $display("[INFO] Step 9: Asserting software E-Stop (0x000 <= 0x04)...");
        axi_write(12'h000, 32'h00000004); // bit 2 = ESTOP_SOFT
        #100;
        axi_read(12'h004, rd_data);
        if (rd_data[3] === 1'b0 && rd_data[5] === 1'b1) begin
            $display("[PASS] Step 9: Software E-Stop asserted! (ESTOP_ACTIVE=1, PERMIT=0)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 9: Software E-Stop failed to engage! SR = 0x%08X", rd_data);
            fail_count = fail_count + 1;
        end

        $display("---------------------------------------------------------------");
        $display("  Summary: %0d Tests Passed, %0d Failed", pass_count, fail_count);
        $display("===============================================================");

        if (fail_count == 0) begin
            $display(">> SUCCESS: ALL AXI4-LITE PS-PL BRIDGE TESTS PASSED! <<");
        end else begin
            $display(">> FAILURE DETECTED IN AXI4-LITE TESTBENCH <<");
            $fatal(1, "AXI4-Lite testbench failed!");
        end

        $finish;
    end

endmodule
`default_nettype wire
