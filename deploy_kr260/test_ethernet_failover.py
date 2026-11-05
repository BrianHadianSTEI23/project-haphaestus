#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Ethernet Link & Failover Regression Test
Author: Brian A. Hadian
Project: Hephaestus - KR260 Autonomous Robotics Hardware Accelerator
"""

import sys
import os
import time
import unittest

try:
    from kr260_ethernet_server import KR260EthernetServer
    from jetson_link_client import JetsonLinkClient
except ImportError:
    from deploy_kr260.kr260_ethernet_server import KR260EthernetServer
    from deploy_kr260.jetson_link_client import JetsonLinkClient


class TestEthernetFailover(unittest.TestCase):
    """Verifies Gigabit Ethernet link between Jetson Orin Nano & KR260 with UART failover."""

    def test_ethernet_and_failover_lifecycle(self):
        print("\n=======================================================")
        print("  TESTING GIGABIT ETHERNET & UART FAILOVER LIFECYCLE")
        print("  Author: Brian A. Hadian")
        print("=======================================================")

        # Step 1: Start Ethernet Server
        server = KR260EthernetServer(host="127.0.0.1", port=5006, mock=True)
        server.start()
        time.sleep(0.1)

        # Step 2: Initialize Jetson Link Client
        client = JetsonLinkClient(
            eth_host="127.0.0.1",
            eth_port=5006,
            uart_port="/dev/ttyTHS1",
            eth_timeout=0.015,
            mock=True
        )

        test_obs = [205] * 64 + [0] * 10

        # Step 3: Verify Primary Ethernet Link
        print("\n[Step 1] Testing Primary Gigabit Ethernet Link...")
        resp1 = client.exchange(test_obs)
        print(f"  Transport:     {resp1['transport']}")
        print(f"  Actions:       {resp1['actions_q88']}")
        print(f"  Roundtrip:     {resp1.get('roundtrip_ms', 0):.2f} ms")
        self.assertEqual(resp1["actions_q88"], [-46, 60, 34])
        self.assertFalse(resp1["failover_active"])

        # Step 4: Stop Server to simulate Ethernet physical disconnection
        print("\n[Step 2] Simulating Ethernet cable disconnect / network failure...")
        server.stop()
        time.sleep(0.05)

        # Step 5: Verify Automatic Seamless Failover to UART
        print("\n[Step 3] Verifying Automatic Failover to UART Link...")
        resp2 = client.exchange(test_obs)
        print(f"  Transport:     {resp2['transport']}")
        print(f"  Actions:       {resp2['actions_q88']}")
        print(f"  Failover:      {resp2['failover_active']}")
        self.assertEqual(resp2["actions_q88"], [-46, 60, 34])
        self.assertTrue(resp2["failover_active"])
        self.assertEqual(client.active_link, "uart")

        # Step 6: Restart Ethernet Server to simulate network recovery
        print("\n[Step 4] Simulating Ethernet cable reconnection (recovering server)...")
        server_reborn = KR260EthernetServer(host="127.0.0.1", port=5006, mock=True)
        server_reborn.start()
        time.sleep(0.1)

        # Step 7: Fast-forward probe cycle to trigger failback
        client._consecutive_uart_cycles = 55
        resp3 = client.exchange(test_obs)
        print(f"  Transport:     {resp3['transport']}")
        print(f"  Actions:       {resp3['actions_q88']}")
        print(f"  Failover:      {resp3['failover_active']}")
        self.assertEqual(resp3["actions_q88"], [-46, 60, 34])
        self.assertFalse(resp3["failover_active"])
        self.assertEqual(client.active_link, "ethernet")

        # Cleanup
        server_reborn.stop()
        client.close()
        print("\n>> ALL ETHERNET & UART FAILOVER TESTS PASSED 100%! <<\n")


if __name__ == "__main__":
    unittest.main()
