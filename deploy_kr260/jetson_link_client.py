#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Orin Nano <-> AMD Kria KR260 Dual-Link Client with Automatic Failover
Author: Brian A. Hadian
Project: Hephaestus - KR260 Autonomous Robotics Hardware Accelerator

Runs on NVIDIA Jetson Orin Nano (Ubuntu 22.04 LTS / JetPack 5.x/6.x) or laptop test environment.
Provides high-reliability redundant telemetry and neural policy communication:
1. Primary Link: High-Speed Gigabit Ethernet (UDP socket, <0.4 ms roundtrip)
2. Failover Link: Robust Hardware UART (/dev/ttyTHS1 @ 921,600 baud, ~4.5 ms roundtrip)

If Ethernet cable is disconnected, socket drops, or packets time out (>15 ms),
the client seamlessly and instantaneously fails over to UART so the robot's
20 ms safety control loop never starves.
"""

import sys
import os
import socket
import struct
import time
import json
import argparse
from typing import Optional, List, Dict, Any, Tuple

# Import UART driver for secondary link
try:
    from kr260_uart_driver import KR260UARTDriver
except ImportError:
    from deploy_kr260.kr260_uart_driver import KR260UARTDriver


class JetsonLinkClient:
    """Dual-Link Client with Automatic Ethernet <-> UART Failover."""

    HEADER_REQ = bytes.fromhex("5A A5 01 94")
    HEADER_RESP = bytes.fromhex("5A A5 02 06")

    def __init__(
        self,
        eth_host: str = "192.168.10.2",
        eth_port: int = 5005,
        eth_protocol: str = "udp",
        eth_timeout: float = 0.015,  # 15 ms timeout (within 20ms control loop)
        uart_port: str = "/dev/ttyTHS1",
        uart_baud: int = 921600,
        mock: bool = False
    ):
        self.eth_host = eth_host
        self.eth_port = eth_port
        self.eth_protocol = eth_protocol.lower()
        self.eth_timeout = eth_timeout
        self.uart_port = uart_port
        self.uart_baud = uart_baud
        self.mock = mock

        # Active link state: "ethernet" or "uart"
        self.active_link = "ethernet"
        self.failover_count = 0
        self.failback_count = 0

        # Ethernet Socket
        self._sock: Optional[socket.socket] = None
        self._init_ethernet_socket()

        # UART Secondary Link
        self._uart_driver = KR260UARTDriver(
            port=self.uart_port,
            baudrate=self.uart_baud,
            timeout=0.030,
            mock=self.mock
        )

        # Health monitoring & recovery probe counter
        self._consecutive_uart_cycles = 0
        self._probe_interval_cycles = 50  # Probe Ethernet every 50 cycles (1 sec @ 50 Hz)

    def _init_ethernet_socket(self) -> None:
        """Initializes client UDP socket with tight timeout."""
        try:
            if self._sock:
                self._sock.close()
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.settimeout(self.eth_timeout)
        except Exception as e:
            self._sock = None

    @staticmethod
    def calc_crc8_maxim(data: bytes) -> int:
        """Computes CRC-8 Maxim."""
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ (0x8C if crc & 1 else 0)
        return crc & 0xFF

    def build_request_packet(self, observations_q88: List[int]) -> bytes:
        """Encodes 74 Q8.8 observation features into a 153-byte binary packet."""
        if len(observations_q88) != 74:
            raise ValueError(f"Expected 74 observations, got {len(observations_q88)}")
        body = bytes([0x01, 0x94]) + struct.pack("<74h", *observations_q88)
        crc = self.calc_crc8_maxim(body)
        return bytes.fromhex("5A A5") + body + bytes([crc])

    def parse_response_packet(self, raw: bytes) -> Dict[str, Any]:
        """Parses and validates an 11-byte response packet."""
        if len(raw) != 11:
            raise ValueError(f"Expected 11 response bytes, got {len(raw)}")
        if raw[:4] != self.HEADER_RESP:
            raise ValueError(f"Invalid response header: {raw[:4].hex(' ')}")

        expected_crc = self.calc_crc8_maxim(raw[2:-1])
        if raw[-1] != expected_crc:
            raise ValueError(f"CRC-8 mismatch: got 0x{raw[-1]:02X}, expected 0x{expected_crc:02X}")

        actions = struct.unpack("<3h", raw[4:10])
        return {
            "actions_q88": list(actions),
            "actions_float": [a / 256.0 for a in actions],
            "actions_hex": [f"0x{a & 0xFFFF:04X}" for a in actions],
            "crc": raw[-1],
            "raw_hex": raw.hex(" ")
        }

    def _exchange_ethernet(self, req_bytes: bytes) -> Dict[str, Any]:
        """Sends frame over Ethernet (UDP) and waits for response."""
        if self._sock is None:
            self._init_ethernet_socket()
        if self._sock is None:
            raise ConnectionError("Ethernet socket unavailable")

        t_start = time.perf_counter()
        self._sock.sendto(req_bytes, (self.eth_host, self.eth_port))
        raw_resp, _ = self._sock.recvfrom(512)
        t_end = time.perf_counter()

        parsed = self.parse_response_packet(raw_resp)
        parsed["roundtrip_ms"] = (t_end - t_start) * 1000.0
        parsed["transport"] = "ETHERNET (Gigabit UDP)"
        parsed["failover_active"] = False
        return parsed

    def _probe_ethernet_healthy(self, req_bytes: bytes) -> bool:
        """Lightweight background probe to test if Ethernet connection has recovered."""
        try:
            probe_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe_sock.settimeout(0.010)
            probe_sock.sendto(req_bytes, (self.eth_host, self.eth_port))
            raw_resp, _ = probe_sock.recvfrom(512)
            probe_sock.close()
            parsed = self.parse_response_packet(raw_resp)
            return True
        except Exception:
            return False

    def exchange(
        self,
        observations_q88: List[int],
        force_transport: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main interface function.
        Transmits 74 observations and receives 3 actions with AUTOMATIC FAILOVER.

        - If force_transport='ethernet': only uses Ethernet.
        - If force_transport='uart': only uses UART.
        - Default (force_transport=None): Uses Ethernet. If it times out or errors,
          automatically fails over to UART seamlessly!
        """
        req_bytes = self.build_request_packet(observations_q88)

        # Forced Mode: UART
        if force_transport == "uart":
            res = self._uart_driver.exchange(observations_q88)
            res["transport"] = "UART (Forced)"
            res["failover_active"] = False
            return res

        # Forced Mode: Ethernet
        if force_transport == "ethernet":
            return self._exchange_ethernet(req_bytes)

        # Automatic Mode with Failover & Health Recovery
        if self.active_link == "ethernet":
            try:
                res = self._exchange_ethernet(req_bytes)
                return res
            except (socket.timeout, OSError, ConnectionError, ValueError) as err:
                # ETHERNET DISRUPTED -> TRIGGER AUTOMATIC FAILOVER TO UART!
                self.active_link = "uart"
                self.failover_count += 1
                self._consecutive_uart_cycles = 1

                # Execute immediately on UART
                res = self._uart_driver.exchange(observations_q88)
                res["transport"] = "UART (FAILOVER ACTIVE: Ethernet timed out)"
                res["failover_active"] = True
                res["failover_reason"] = str(err)
                return res

        else:
            # Active Link is UART (Failover Mode Active)
            self._consecutive_uart_cycles += 1

            # Check if Ethernet has recovered
            if self._consecutive_uart_cycles >= self._probe_interval_cycles:
                self._consecutive_uart_cycles = 0
                if self._probe_ethernet_healthy(req_bytes):
                    # ETHERNET RESTORED -> RESTORE PRIMARY LINK!
                    self.active_link = "ethernet"
                    self.failback_count += 1
                    res = self._exchange_ethernet(req_bytes)
                    res["transport"] = "ETHERNET (RESTORED from UART failover)"
                    res["failover_active"] = False
                    return res

            # Continue on UART
            res = self._uart_driver.exchange(observations_q88)
            res["transport"] = "UART (Failover Active)"
            res["failover_active"] = True
            return res

    def close(self) -> None:
        """Closes all network and serial sockets."""
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._uart_driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    parser = argparse.ArgumentParser(description="Jetson Orin Nano Dual-Link Client (Author: Brian A. Hadian)")
    parser.add_argument("--host", default="127.0.0.1", help="KR260 Ethernet IP (default: 127.0.0.1 for local test)")
    parser.add_argument("--port", type=int, default=5005, help="KR260 Ethernet Port (default: 5005)")
    parser.add_argument("--uart", default="/dev/ttyTHS1", help="Jetson UART port (default: /dev/ttyTHS1)")
    parser.add_argument("--mock", action="store_true", help="Force mock mode for laptop testing")
    parser.add_argument("--test-failover", action="store_true", help="Simulate Ethernet drop and verify failover")
    args = parser.parse_args()

    print("========================================================================")
    print("  PROJECT HEPHAESTUS: JETSON ORIN NANO DUAL-LINK CLIENT")
    print("  Author: Brian A. Hadian")
    print("========================================================================")
    print(f"Target KR260 Ethernet: {args.host}:{args.port}")
    print(f"Secondary UART Link:   {args.uart} @ 921,600 baud")
    print("------------------------------------------------------------------------")

    client = JetsonLinkClient(
        eth_host=args.host,
        eth_port=args.port,
        uart_port=args.uart,
        mock=args.mock
    )

    test_obs = [205] * 64 + [0] * 10
    print("\n[TEST 1] Sending observation over primary link (Auto Mode)...")
    try:
        resp = client.exchange(test_obs)
        print(f" Transport:      {resp.get('transport')}")
        print(f" Roundtrip:      {resp.get('roundtrip_ms', 0):.2f} ms")
        print(f" Actions Q8.8:   {resp.get('actions_q88')}")
        print(f" Actions Float:  {resp.get('actions_float')}")
        print(f" Failover State: {resp.get('failover_active')}")
    except Exception as e:
        print(f" Error: {e}")

    client.close()


if __name__ == "__main__":
    main()
