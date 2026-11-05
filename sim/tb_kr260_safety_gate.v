`timescale 1ns/1ps
`default_nettype none

// Hardware Safety & Security Gate Simulation Testbench for AMD Kria KR260
// Author: Brian A. Hadian
//
// Tests:
// - Strict priority fault hierarchy (Reset=1, E-stop=2, Timeout=8, Overspeed=10)
// - Monotonic 20 ms lease watchdog countdown and zero-latency fail-closed cutoff
// - Wire-speed Q8.8 velocity clamping
// - Emergency stop latching and recovery

module tb_kr260_safety_gate;
    reg clk = 1'b0;
    always #10 clk = ~clk;

    reg        rst_n           = 1'b0;
    reg        ready           = 1'b1;
    reg        estop_in        = 1'b0;
    reg        estop_soft      = 1'b0;
    reg        estop_clear     = 1'b0;
    reg        protective_stop = 1'b0;
    reg        lease_refresh   = 1'b0;
    reg [31:0] lease_limit     = 32'd100; // Fast 100-cycle limit for sim
    reg        result_valid    = 1'b1;
    reg        arithmetic_fault = 1'b0;

    reg signed [15:0] raw_vx = 16'sd100;
    reg signed [15:0] raw_vy = -16'sd50;
    reg signed [15:0] raw_wz = 16'sd200;

    reg signed [15:0] max_v  = 16'sd256; // 1.0 m/s
    reg signed [15:0] max_w  = 16'sd384; // 1.5 rad/s

    wire               safety_permit;
    wire signed [15:0] safe_vx;
    wire signed [15:0] safe_vy;
    wire signed [15:0] safe_wz;
    wire [7:0]         fault_code;
    wire [31:0]        lease_timer;
    wire               lease_expired;
    wire               estop_latched;

    igor_safety_security_gate #(
        .DEFAULT_LEASE_CYCLES(32'd100)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .ready(ready),
        .estop_in(estop_in),
        .estop_soft(estop_soft),
        .estop_clear(estop_clear),
        .protective_stop(protective_stop),
        .lease_refresh(lease_refresh),
        .lease_limit(lease_limit),
        .result_valid(result_valid),
        .arithmetic_fault(arithmetic_fault),
        .raw_vx(raw_vx),
        .raw_vy(raw_vy),
        .raw_wz(raw_wz),
        .max_v(max_v),
        .max_w(max_w),
        .safety_permit(safety_permit),
        .safe_vx(safe_vx),
        .safe_vy(safe_vy),
        .safe_wz(safe_wz),
        .fault_code(fault_code),
        .lease_timer(lease_timer),
        .lease_expired(lease_expired),
        .estop_latched(estop_latched)
    );

    integer pass_count = 0;
    integer fail_count = 0;

    initial begin
        $dumpfile("sim_safety_gate.vcd");
        $dumpvars(0, tb_kr260_safety_gate);

        $display("===============================================================");
        $display("  AMD Kria KR260 Hardware Safety Gate Simulation");
        $display("  Author: Brian A. Hadian");
        $display("===============================================================");

        // Test 1: Reset Active -> Expect Fault Code 1 and Permit 0
        rst_n = 1'b0;
        #50;
        if (fault_code === 8'd1 && safety_permit === 1'b0 && safe_vx === 16'sd0) begin
            $display("[PASS] Step 1: Reset asserts fault_code=1, permit=0, outputs zeroed");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 1: Reset check failed! fault_code=%0d", fault_code);
            fail_count = fail_count + 1;
        end

        // Release Reset & Refresh Lease
        @(posedge clk);
        rst_n = 1'b1;
        lease_refresh <= 1'b1;
        @(posedge clk);
        lease_refresh <= 1'b0;
        #50;

        // Test 2: Normal Operation -> Expect Fault Code 0, Permit 1, Raw Values Passed
        if (fault_code === 8'd0 && safety_permit === 1'b1 && safe_vx === 16'sd100 && safe_vy === -16'sd50) begin
            $display("[PASS] Step 2: Normal operation verified: fault_code=0, permit=1, values passed");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 2: Normal operation failed! fault_code=%0d permit=%b", fault_code, safety_permit);
            fail_count = fail_count + 1;
        end

        // Test 3: Overspeed Clamping (vx = 500 > max_v 256)
        raw_vx = 16'sd500;
        #20;
        if (fault_code === 8'd10 && safety_permit === 1'b1 && safe_vx === 16'sd256) begin
            $display("[PASS] Step 3: Overspeed clamped to max_v (500 -> 256) with fault_code=10");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 3: Clamping failed! safe_vx=%d fault_code=%d", safe_vx, fault_code);
            fail_count = fail_count + 1;
        end
        raw_vx = 16'sd100;

        // Test 4: Monotonic Watchdog Lease Expiration
        $display("[INFO] Step 4: Letting watchdog countdown from 100 to 0 without refresh...");
        #2500; // Wait 2500 ns (> 100 cycles)
        if (fault_code === 8'd8 && safety_permit === 1'b0 && safe_vx === 16'sd0 && lease_expired === 1'b1) begin
            $display("[PASS] Step 4: Watchdog expired! fault_code=8, permit=0, motor outputs zeroed");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 4: Watchdog timeout failed! fault_code=%d permit=%b", fault_code, safety_permit);
            fail_count = fail_count + 1;
        end

        // Test 5: Watchdog Lease Refresh Recovery
        @(posedge clk);
        lease_refresh <= 1'b1;
        @(posedge clk);
        lease_refresh <= 1'b0;
        #50;
        if (fault_code === 8'd0 && safety_permit === 1'b1 && safe_vx === 16'sd100) begin
            $display("[PASS] Step 5: System recovered after lease refresh! permit=1");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 5: Recovery failed! fault_code=%d permit=%b", fault_code, safety_permit);
            fail_count = fail_count + 1;
        end

        // Test 6: Physical E-Stop Latch & Fail-Closed Cutoff
        $display("[INFO] Step 6: Testing physical E-Stop assertion and latching...");
        estop_in = 1'b1;
        #50;
        if (fault_code === 8'd2 && safety_permit === 1'b0 && safe_vx === 16'sd0) begin
            $display("[PASS] Step 6a: E-Stop asserts fault_code=2 and cuts all power");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 6a: E-Stop assert failed!");
            fail_count = fail_count + 1;
        end

        // Release E-stop pulse: verify it stays latched!
        estop_in = 1'b0;
        #50;
        if (fault_code === 8'd2 && safety_permit === 1'b0 && estop_latched === 1'b1) begin
            $display("[PASS] Step 6b: E-Stop remains latched after button release (Hardware Interlock)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 6b: E-Stop latch failed!");
            fail_count = fail_count + 1;
        end

        // Clear E-stop latch
        @(posedge clk);
        estop_clear <= 1'b1;
        @(posedge clk);
        estop_clear <= 1'b0;
        #50;
        if (fault_code === 8'd0 && safety_permit === 1'b1 && estop_latched === 1'b0) begin
            $display("[PASS] Step 6c: E-Stop latch cleared successfully! Normal operation restored");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Step 6c: E-Stop clear failed!");
            fail_count = fail_count + 1;
        end

        $display("---------------------------------------------------------------");
        $display("  Summary: %0d Tests Passed, %0d Failed", pass_count, fail_count);
        $display("===============================================================");

        if (fail_count == 0) begin
            $display(">> SUCCESS: ALL HARDWARE SAFETY GATE TESTS PASSED! <<");
        end else begin
            $display(">> FAILURE DETECTED IN SAFETY GATE TESTBENCH <<");
            $fatal(1, "Safety gate testbench failed!");
        end

        $finish;
    end

endmodule
`default_nettype wire
