`timescale 1ns/1ps
`default_nettype none

// Hardware-Enforced Safety & Security Gate for AMD Kria KR260 / Project Hephaestus
// Author: Brian A. Hadian
//
// Enforces zero-latency fail-closed hardware interlocks:
// - Physical & Software Emergency Stop (latched until explicit reset)
// - Monotonic Watchdog Lease Counter (default 20 ms / 1,000,000 cycles @ 50MHz)
// - Wire-speed Velocity and Angular Rate Clamper (signed Q8.8 bounds)
// - Strict priority fault code output and hardware status logging
module igor_safety_security_gate #(
    parameter [31:0] DEFAULT_LEASE_CYCLES = 32'd1000000 // 20 ms @ 50 MHz
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        ready,             // Core / peripheral ready
    input  wire        estop_in,          // Raw E-stop input (1 = stop asserted)
    input  wire        estop_soft,        // Software E-stop register (1 = stop asserted)
    input  wire        estop_clear,       // Clear latched E-stop
    input  wire        protective_stop,   // External proximity / bumper stop
    input  wire        lease_refresh,     // Strobe from host refreshing the lease
    input  wire [31:0] lease_limit,       // Configurable lease timeout threshold
    input  wire        result_valid,      // Inference core output is valid
    input  wire        arithmetic_fault,  // Core numerical saturation / fault
    
    // Raw candidate actions from policy (signed Q8.8)
    input  wire signed [15:0] raw_vx,     // Candidate linear velocity X (Q8.8)
    input  wire signed [15:0] raw_vy,     // Candidate linear velocity Y (Q8.8)
    input  wire signed [15:0] raw_wz,     // Candidate angular velocity Z (Q8.8)
    
    // Velocity Envelope Limits (signed Q8.8)
    input  wire signed [15:0] max_v,      // Max linear speed (e.g. 1.0 m/s = 256)
    input  wire signed [15:0] max_w,      // Max angular speed (e.g. 1.5 rad/s = 384)
    
    // Gated Safe Outputs (zeroed when not safe)
    output wire               safety_permit,
    output reg  signed [15:0] safe_vx,
    output reg  signed [15:0] safe_vy,
    output reg  signed [15:0] safe_wz,
    output reg  [7:0]         fault_code,
    output reg  [31:0]        lease_timer,
    output wire               lease_expired,
    output reg                estop_latched
);
    // Double-flop synchronizer for external E-stop
    reg estop_meta, estop_sync;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            estop_meta <= 1'b0;
            estop_sync <= 1'b0;
        end else begin
            estop_meta <= estop_in;
            estop_sync <= estop_meta;
        end
    end

    wire estop_trigger = estop_in | estop_sync | estop_soft;

    // E-stop latch logic: Once triggered, remains latched until explicit reset or clear
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            estop_latched <= 1'b0;
        end else begin
            if (estop_trigger) begin
                estop_latched <= 1'b1;
            end else if (estop_clear) begin
                estop_latched <= 1'b0;
            end
        end
    end

    wire estop_active = estop_trigger | estop_latched;

    // Monotonic Watchdog Lease Counter
    wire [31:0] active_limit = (lease_limit != 32'd0) ? lease_limit : DEFAULT_LEASE_CYCLES;
    reg timeout_flag;
    assign lease_expired = timeout_flag;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lease_timer  <= DEFAULT_LEASE_CYCLES;
            timeout_flag <= 1'b0;
        end else begin
            if (lease_refresh) begin
                lease_timer  <= active_limit;
                timeout_flag <= 1'b0;
            end else if (lease_timer != 32'd0) begin
                lease_timer  <= lease_timer - 1'b1;
                timeout_flag <= 1'b0;
            end else begin
                timeout_flag <= 1'b1; // Lease expired: fail-closed!
            end
        end
    end

    // Velocity Clamper
    wire signed [15:0] abs_vx = (raw_vx < 16'sd0) ? -raw_vx : raw_vx;
    wire signed [15:0] abs_vy = (raw_vy < 16'sd0) ? -raw_vy : raw_vy;
    wire signed [15:0] abs_wz = (raw_wz < 16'sd0) ? -raw_wz : raw_wz;

    wire overspeed = (abs_vx > max_v) || (abs_vy > max_v) || (abs_wz > max_w);

    // Combinational Safety Evaluation with Strict Fault Priority Hierarchy
    // Priority:
    // 1: Reset active
    // 2: E-stop active (physical or soft or latched)
    // 3: Protective stop
    // 4: Peripheral / system not ready
    // 7: Arithmetic fault (numerical overflow/saturation)
    // 8: Watchdog lease timeout (control heartbeat lost)
    // 9: Inference result invalid / uncalculated
    // 10: Speed limit exceeded (rate clamped)
    // 0: All clear / safe
    always @(*) begin
        if (!rst_n) begin
            fault_code = 8'd1;
        end else if (estop_active) begin
            fault_code = 8'd2;
        end else if (protective_stop) begin
            fault_code = 8'd3;
        end else if (!ready) begin
            fault_code = 8'd4;
        end else if (arithmetic_fault) begin
            fault_code = 8'd7;
        end else if (timeout_flag) begin
            fault_code = 8'd8;
        end else if (!result_valid) begin
            fault_code = 8'd9;
        end else if (overspeed) begin
            fault_code = 8'd10;
        end else begin
            fault_code = 8'd0;
        end
    end

    assign safety_permit = (fault_code == 8'd0 || fault_code == 8'd10);

    // Actuation output masking & clamping
    always @(*) begin
        if (safety_permit) begin
            // Clamped output values
            safe_vx = (raw_vx > max_v)  ? max_v :
                      (raw_vx < -max_v) ? -max_v : raw_vx;
            safe_vy = (raw_vy > max_v)  ? max_v :
                      (raw_vy < -max_v) ? -max_v : raw_vy;
            safe_wz = (raw_wz > max_w)  ? max_w :
                      (raw_wz < -max_w) ? -max_w : raw_wz;
        end else begin
            // Fail-closed: Zero all outputs immediately
            safe_vx = 16'sd0;
            safe_vy = 16'sd0;
            safe_wz = 16'sd0;
        end
    end

endmodule
`default_nettype wire
