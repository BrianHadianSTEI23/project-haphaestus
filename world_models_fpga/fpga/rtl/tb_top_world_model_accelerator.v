// ============================================================================
// Module: tb_top_world_model_accelerator.v
// Project: Project Hephaestus - World Models Accelerator on AMD Kria KR260
// Author: Brian A. Hadian
//
// Description:
// Self-Checking Verilog RTL Testbench for the Reconstructed Action-Conditioned
// World Model Accelerator Core on AMD Kria KR260.
// Verifies bit-exact numerical equivalence against the Q8.8 software reference:
//   || y_RTL - y_Q8.8 ||_2 == 0.0
// Simulates with Icarus Verilog (iverilog / vvp) and ModelSim / Vivado XSim.
// ============================================================================

`timescale 1ns / 1ps

module tb_top_world_model_accelerator;

    parameter VEC_PATH = "world_models_fpga/fpga/vectors/";

    // Clock and Reset
    reg clk;
    reg rst_n;

    // AXI4-Lite Signals
    reg  [9:0]  s_axi_awaddr;
    reg  [2:0]  s_axi_awprot;
    reg         s_axi_awvalid;
    wire        s_axi_awready;

    reg  [31:0] s_axi_wdata;
    reg  [3:0]  s_axi_wstrb;
    reg         s_axi_wvalid;
    wire        s_axi_wready;

    wire [1:0]  s_axi_bresp;
    wire        s_axi_bvalid;
    reg         s_axi_bready;

    reg  [9:0]  s_axi_araddr;
    reg  [2:0]  s_axi_arprot;
    reg         s_axi_arvalid;
    wire        s_axi_arready;

    wire [31:0] s_axi_rdata;
    wire [1:0]  s_axi_rresp;
    wire        s_axi_rvalid;
    reg         s_axi_rready;

    // AXI-Stream Signals
    reg  [15:0] s_axis_tdata;
    reg         s_axis_tvalid;
    wire        s_axis_tready;
    reg         s_axis_tlast;

    wire        interrupt;

    // Golden Vectors Storage
    reg signed [15:0] vec_macro_in    [0:18];
    reg signed [15:0] vec_policy_in   [0:31];
    reg signed [15:0] golden_next_s   [0:15];
    reg signed [15:0] golden_col_prob [0:0];
    reg signed [15:0] golden_action   [0:2];

    // DUT Instantiation
    top_world_model_accelerator #(
        .C_S_AXI_DATA_WIDTH(32),
        .C_S_AXI_ADDR_WIDTH(10),
        .VEC_PATH(VEC_PATH)
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
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .interrupt(interrupt)
    );

    // 200 MHz Clock Generator (5.0 ns period)
    initial begin
        clk = 1'b0;
        forever #2.5 clk = ~clk;
    end

    // AXI-Lite Master Write Task
    task axi_write;
        input [9:0]  addr;
        input [31:0] data;
        begin
            @(posedge clk);
            s_axi_awaddr  <= addr;
            s_axi_awvalid <= 1'b1;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'b1111;
            s_axi_wvalid  <= 1'b1;
            s_axi_bready  <= 1'b1;

            @(posedge clk);
            while (!s_axi_awready || !s_axi_wready) @(posedge clk);

            s_axi_awvalid <= 1'b0;
            s_axi_wvalid  <= 1'b0;

            while (!s_axi_bvalid) @(posedge clk);
            @(posedge clk);
            s_axi_bready  <= 1'b0;
        end
    endtask

    // AXI-Lite Master Read Task
    task axi_read;
        input  [9:0]  addr;
        output [31:0] data;
        begin
            @(posedge clk);
            s_axi_araddr  <= addr;
            s_axi_arvalid <= 1'b1;
            s_axi_rready  <= 1'b1;

            @(posedge clk);
            while (!s_axi_arready) @(posedge clk);
            s_axi_arvalid <= 1'b0;

            while (!s_axi_rvalid) @(posedge clk);
            data = s_axi_rdata;
            @(posedge clk);
            s_axi_rready  <= 1'b0;
        end
    endtask

    integer i;
    integer mismatch_count;
    reg [31:0] read_val;
    reg [31:0] measured_cycles;

    initial begin
        $display("\n=======================================================");
        $display("  STARTING HARDWARE RTL TESTBENCH SIMULATION          ");
        $display("  Target: AMD Kria KR260 Accelerator (200 MHz Clock)   ");
        $display("=======================================================\n");

        // Load Golden Files
        $readmemh({VEC_PATH, "tb_input_macro_19.hex"},    vec_macro_in);
        $readmemh({VEC_PATH, "tb_input_policy_32.hex"},   vec_policy_in);
        $readmemh({VEC_PATH, "tb_golden_next_S_16.hex"},  golden_next_s);
        $readmemh({VEC_PATH, "tb_golden_col_prob_1.hex"}, golden_col_prob);
        $readmemh({VEC_PATH, "tb_golden_action_3.hex"},   golden_action);

        // Initialize signals
        rst_n         = 1'b0;
        s_axi_awaddr  = 10'd0;
        s_axi_awprot  = 3'd0;
        s_axi_awvalid = 1'b0;
        s_axi_wdata   = 32'd0;
        s_axi_wstrb   = 4'd0;
        s_axi_wvalid  = 1'b0;
        s_axi_bready  = 1'b0;
        s_axi_araddr  = 10'd0;
        s_axi_arprot  = 3'd0;
        s_axi_arvalid = 1'b0;
        s_axi_rready  = 1'b0;
        s_axis_tdata  = 16'd0;
        s_axis_tvalid = 1'b0;
        s_axis_tlast  = 1'b0;
        mismatch_count = 0;

        // Apply Reset
        #20;
        rst_n = 1'b1;
        #20;
        $display("[Time %0t ns] Reset released. Configuring AXI registers...", $time);

        // Step 1: Write State S_k (16 words: offsets 0x010 to 0x04C)
        for (i = 0; i < 16; i = i + 1) begin
            axi_write(10'h010 + (i * 4), {{16{vec_macro_in[i][15]}}, vec_macro_in[i]});
        end
        $display("[Time %0t ns] Wrote 16 words of state S_k via AXI4-Lite.", $time);

        // Step 2: Write Candidate Action A_k (3 words: offsets 0x050 to 0x058)
        for (i = 0; i < 3; i = i + 1) begin
            axi_write(10'h050 + (i * 4), {{16{vec_macro_in[16 + i][15]}}, vec_macro_in[16 + i]});
        end
        $display("[Time %0t ns] Wrote 3 words of candidate action A_k via AXI4-Lite.", $time);

        // Step 3: Write Subgoal S_subgoal (16 words: offsets 0x060 to 0x09C)
        for (i = 0; i < 16; i = i + 1) begin
            axi_write(10'h060 + (i * 4), {{16{vec_policy_in[16 + i][15]}}, vec_policy_in[16 + i]});
        end
        $display("[Time %0t ns] Wrote 16 words of target subgoal S_subgoal via AXI4-Lite.", $time);

        // Step 4: Assert Start (Write 0x001 to Control Register 0x000)
        $display("[Time %0t ns] Asserting Start trigger in Control Register...", $time);
        axi_write(10'h000, 32'h00000001);

        // Step 5: Wait for Interrupt or Polling Done
        @(posedge clk);
        while (!interrupt) @(posedge clk);
        $display("[Time %0t ns] *** INTERRUPT DETECTED! *** Execution completed.", $time);

        // Step 6: Read Performance Cycle Counter
        axi_read(10'h008, measured_cycles);
        $display("[Time %0t ns] Measured Hardware Latency: %0d clock cycles (%0.2f us at 200 MHz)",
                 $time, measured_cycles, (measured_cycles * 5.0) / 1000.0);

        // Step 7: Verify Next State Outputs (0x0A0 to 0x0DC)
        $display("\n--- VERIFYING PREDICTED NEXT STATE S_{t+K} ---");
        for (i = 0; i < 16; i = i + 1) begin
            axi_read(10'h0A0 + (i * 4), read_val);
            if (read_val[15:0] !== golden_next_s[i]) begin
                $display("MISMATCH at Next_S[%0d]! RTL: 0x%04X (%0d), Golden: 0x%04X (%0d)",
                         i, read_val[15:0], $signed(read_val[15:0]), golden_next_s[i], golden_next_s[i]);
                mismatch_count = mismatch_count + 1;
            end else begin
                $display("  MATCH Next_S[%2d]: 0x%04X (Float: %0.4f)",
                         i, read_val[15:0], $itor($signed(read_val[15:0])) / 256.0);
            end
        end

        // Step 8: Verify Collision Barrier Probability (0x0E0)
        $display("\n--- VERIFYING SAFETY BARRIER PROBABILITY B(S, A) ---");
        axi_read(10'h0E0, read_val);
        if (read_val[15:0] !== golden_col_prob[0]) begin
            $display("MISMATCH at Collision Prob! RTL: 0x%04X, Golden: 0x%04X", read_val[15:0], golden_col_prob[0]);
            mismatch_count = mismatch_count + 1;
        end else begin
            $display("  MATCH Col Prob: 0x%04X (Float: %0.4f)", read_val[15:0], $itor($signed(read_val[15:0])) / 256.0);
        end

        // Step 9: Verify Micro Steering Action Outputs (0x0E4 to 0x0EC)
        $display("\n--- VERIFYING MICRO MOTOR ACTIONS [vx, vy, wz] ---");
        for (i = 0; i < 3; i = i + 1) begin
            axi_read(10'h0E4 + (i * 4), read_val);
            if (read_val[15:0] !== golden_action[i]) begin
                $display("MISMATCH at Action[%0d]! RTL: 0x%04X, Golden: 0x%04X", i, read_val[15:0], golden_action[i]);
                mismatch_count = mismatch_count + 1;
            end else begin
                $display("  MATCH Action[%0d]: 0x%04X (Float: %0.4f)", i, read_val[15:0], $itor($signed(read_val[15:0])) / 256.0);
            end
        end

        // Final Summary
        $display("\n=======================================================");
        if (mismatch_count == 0) begin
            $display("  [PASSED] 100%% BIT-ACCURATE HARDWARE EQUIVALENCE!   ");
            $display("  Total Verified Outputs: 20 / 20 words               ");
            $display("  Numerical Relative Error E_rel = 0.000000           ");
            $display("=======================================================\n");
        end else begin
            $display("  [FAILED] Found %0d mismatches out of 20 outputs!     ", mismatch_count);
            $display("=======================================================\n");
        end

        #50;
        $finish;
    end

endmodule
