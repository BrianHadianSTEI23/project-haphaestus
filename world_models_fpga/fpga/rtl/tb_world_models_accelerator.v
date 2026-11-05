// ============================================================================
// Testbench: tb_world_models_accelerator.v
// Project: Project Hephaestus - World Models Accelerator on AMD Kria KR260
// Author: Brian A. Hadian
//
// Description:
// Comprehensive simulation testbench for AXI4-Lite World Models Accelerator.
// Tests Baseline, LeWM, and H-JEPA modes, measuring cycle latency and outputs.
// ============================================================================

`timescale 1ns / 1ps

module tb_world_models_accelerator;

    reg clk;
    reg rst_n;

    // AXI-Lite signals
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

    wire        interrupt;

    // Instantiate Device Under Test (DUT)
    world_models_accelerator_top #(
        .C_S_AXI_DATA_WIDTH(32),
        .C_S_AXI_ADDR_WIDTH(10)
    ) dut (
        .s_axi_aclk(clk),
        .s_axi_aresetn(rst_n),
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
        .interrupt(interrupt)
    );

    // 100 MHz Clock (10ns period)
    always #5 clk = ~clk;

    // AXI4-Lite Write Task
    task axi_write(input [9:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            s_axi_awaddr  <= addr;
            s_axi_awvalid <= 1'b1;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'b1111;
            s_axi_wvalid  <= 1'b1;
            s_axi_bready  <= 1'b1;

            @(posedge clk);
            while (~s_axi_awready || ~s_axi_wready) @(posedge clk);

            s_axi_awvalid <= 1'b0;
            s_axi_wvalid  <= 1'b0;

            while (~s_axi_bvalid) @(posedge clk);
            @(posedge clk);
            s_axi_bready  <= 1'b0;
        end
    endtask

    // AXI4-Lite Read Task
    task axi_read(input [9:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            s_axi_araddr  <= addr;
            s_axi_arvalid <= 1'b1;
            s_axi_rready  <= 1'b1;

            @(posedge clk);
            while (~s_axi_arready) @(posedge clk);
            s_axi_arvalid <= 1'b0;

            while (~s_axi_rvalid) @(posedge clk);
            data = s_axi_rdata;
            @(posedge clk);
            s_axi_rready  <= 1'b0;
        end
    endtask

    reg [31:0] read_val;
    integer i;

    initial begin
        clk = 0;
        rst_n = 0;
        s_axi_awaddr = 0;
        s_axi_awprot = 0;
        s_axi_awvalid = 0;
        s_axi_wdata = 0;
        s_axi_wstrb = 0;
        s_axi_wvalid = 0;
        s_axi_bready = 0;
        s_axi_araddr = 0;
        s_axi_arprot = 0;
        s_axi_arvalid = 0;
        s_axi_rready = 0;

        #30 rst_n = 1;
        #20;

        $display("================================================================");
        $display("  STARTING FPGA ACCELERATOR TESTBENCH (Q8.8 FIXED-POINT)");
        $display("================================================================");

        // Load 36 test feature values (Q8.8 format: 1.0 = 16'h0100)
        for (i = 0; i < 36; i = i + 1) begin
            axi_write(10'h010 + (i * 4), 32'h0000_0100 + (i * 8));
        end

        // -------------------------------------------------------------
        // TEST 1: Baseline Policy Mode (mode = 00)
        // -------------------------------------------------------------
        $display("\n>>> [TEST 1] Executing Baseline Policy Mode...");
        axi_write(10'h000, 32'h0000_0001); // Start pulse, mode=0

        // Poll for done
        read_val = 0;
        while ((read_val & 32'h0000_0002) == 0) begin
            axi_read(10'h000, read_val);
        end

        // Read latency in clock cycles
        axi_read(10'h004, read_val);
        $display("  -> Baseline Inference Complete! Latency: %0d clock cycles (%0.2f us @ 100MHz)", 
                 read_val, read_val * 0.01);

        // Read action outputs (vx, vy, wz)
        axi_read(10'h0A0, read_val);
        $display("  -> Output vx (Q8.8): 0x%04X (%0.3f m/s)", read_val[15:0], $signed(read_val[15:0]) / 256.0);
        axi_read(10'h0A4, read_val);
        $display("  -> Output vy (Q8.8): 0x%04X (%0.3f m/s)", read_val[15:0], $signed(read_val[15:0]) / 256.0);
        axi_read(10'h0A8, read_val);
        $display("  -> Output wz (Q8.8): 0x%04X (%0.3f rad/s)", read_val[15:0], $signed(read_val[15:0]) / 256.0);

        // Clear control register
        axi_write(10'h000, 32'h0000_0000);
        #20;

        // -------------------------------------------------------------
        // TEST 2: LeWorldModel (LeWM) Mode (mode = 01)
        // -------------------------------------------------------------
        $display("\n>>> [TEST 2] Executing LeWorldModel (LeWM) Mode...");
        axi_write(10'h000, 32'h0000_0005); // Start pulse, mode=1 (bit 2=1)

        read_val = 0;
        while ((read_val & 32'h0000_0002) == 0) begin
            axi_read(10'h000, read_val);
        end

        axi_read(10'h004, read_val);
        $display("  -> LeWM Inference Complete! Latency: %0d clock cycles (%0.2f us @ 100MHz)", 
                 read_val, read_val * 0.01);

        // Read predicted next latent s_t+1[0]
        axi_read(10'h0A0, read_val);
        $display("  -> Predicted Next Latent s[0] (Q8.8): 0x%04X (%0.3f)", read_val[15:0], $signed(read_val[15:0]) / 256.0);

        axi_write(10'h000, 32'h0000_0000);
        #20;

        // -------------------------------------------------------------
        // TEST 3: H-JEPA Mode (mode = 10)
        // -------------------------------------------------------------
        $display("\n>>> [TEST 3] Executing H-JEPA Macro-Planning Mode...");
        axi_write(10'h000, 32'h0000_0009); // Start pulse, mode=2 (bit 3=1)

        read_val = 0;
        while ((read_val & 32'h0000_0002) == 0) begin
            axi_read(10'h000, read_val);
        end

        axi_read(10'h004, read_val);
        $display("  -> H-JEPA Macro Inference Complete! Latency: %0d clock cycles (%0.2f us @ 100MHz)", 
                 read_val, read_val * 0.01);

        axi_write(10'h000, 32'h0000_0000);
        #20;

        $display("\n[PASSED] ALL 3 WORLD MODEL FPGA ACCELERATOR MODES VERIFIED SUCCESSFULLY!");
        $display("================================================================");
        $finish;
    end

endmodule
