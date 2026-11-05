#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMD Kria KR260 High-Speed Ethernet Bridge Server
Author: Brian A. Hadian
Project: Hephaestus - KR260 Autonomous Robotics Hardware Accelerator

Provides high-speed Gigabit Ethernet (UDP/TCP) communication between the
NVIDIA Jetson Orin Nano perception host and the AMD Kria KR260 PL Neural Accelerator.

Architecture:
- Jetson Orin Nano transmits 74-element Q8.8 observation frames over Gigabit Ethernet.
- KR260 PS receives the UDP datagrams and writes directly to the PL via AXI4-Lite (0xA000_0000).
- PL neural core completes inference in 138.32 us.
- KR260 PS returns 3-element Q8.8 motor action commands & safety telemetry to Jetson in < 0.4 ms.
- If Ethernet link experiences disruption, Jetson automatically fails over to the secondary UART link.
"""

import sys
import os
import socket
import select
import struct
import time
import json
import threading
import argparse
from typing import Optional, Tuple, Dict, Any, List

# Local imports
try:
    from kr260_pl_driver import KR260PLDriver
    from kr260_uart_driver import KR260UARTDriver
except ImportError:
    from deploy_kr260.kr260_pl_driver import KR260PLDriver
    from deploy_kr260.kr260_uart_driver import KR260UARTDriver


class KR260EthernetServer:
    """High-speed UDP/TCP server bridging Jetson Orin Nano to AMD Kria KR260 PL."""

    HEADER_REQ = bytes.fromhex("5A A5 01 94")
    HEADER_RESP = bytes.fromhex("5A A5 02 06")

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5005,
        protocol: str = "udp",
        pl_driver: Optional[KR260PLDriver] = None,
        mock: bool = False
    ):
        self.host = host
        self.port = port
        self.protocol = protocol.lower()
        self.mock = mock

        # Instantiate PL Driver if not provided
        if pl_driver is not None:
            self.pl = pl_driver
        else:
            self.pl = KR260PLDriver(mock=self.mock)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

        # Telemetry & Statistics
        self.stats = {
            "packets_received": 0,
            "packets_responded": 0,
            "crc_errors": 0,
            "header_errors": 0,
            "last_latency_us": 0.0,
            "mean_latency_us": 0.0,
            "uptime_seconds": 0.0,
            "client_endpoint": None
        }

        self._total_latency = 0.0
        self._start_time = 0.0

    @staticmethod
    def calc_crc8_maxim(data: bytes) -> int:
        """Computes CRC-8 Maxim (polynomial 0x8C reversed, init 0x00)."""
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ (0x8C if crc & 1 else 0)
        return crc & 0xFF

    def build_response_packet(self, actions_q88: List[int]) -> bytes:
        """Encodes 3 Q8.8 action outputs into an 11-byte binary packet."""
        body = bytes([0x02, 0x06]) + struct.pack("<3h", *actions_q88)
        crc = self.calc_crc8_maxim(body)
        return bytes.fromhex("5A A5") + body + bytes([crc])

    def parse_request_packet(self, raw: bytes) -> List[int]:
        """Validates and extracts 74 Q8.8 observations from a 153-byte packet."""
        if len(raw) != 153:
            raise ValueError(f"Packet length {len(raw)} != 153 bytes")
        if raw[:4] != self.HEADER_REQ:
            raise ValueError(f"Invalid request header: {raw[:4].hex(' ')}")

        expected_crc = self.calc_crc8_maxim(raw[2:-1])
        if raw[-1] != expected_crc:
            raise ValueError(f"CRC-8 mismatch: got 0x{raw[-1]:02X}, expected 0x{expected_crc:02X}")

        return list(struct.unpack("<74h", raw[4:152]))

    def handle_datagram(self, data: bytes, client_addr: Tuple[str, int]) -> bytes:
        """Processes an incoming observation frame from Jetson Orin Nano."""
        t_start = time.perf_counter()
        self.stats["packets_received"] += 1
        self.stats["client_endpoint"] = f"{client_addr[0]}:{client_addr[1]}"

        try:
            obs_q88 = self.parse_request_packet(data)
        except ValueError as e:
            if "CRC-8" in str(e):
                self.stats["crc_errors"] += 1
            else:
                self.stats["header_errors"] += 1
            raise

        # Execute PL Inference via AXI4-Lite
        result = self.pl.infer(obs_q88)
        actions = result["actions_q88"]

        # Refresh Watchdog Lease on Valid Host Inferences
        self.pl.refresh_lease()

        # Build Response
        resp = self.build_response_packet(actions)
        self.stats["packets_responded"] += 1

        # Track Latency
        lat_us = (time.perf_counter() - t_start) * 1e6
        self.stats["last_latency_us"] = lat_us
        self._total_latency += lat_us
        self.stats["mean_latency_us"] = self._total_latency / self.stats["packets_responded"]

        return resp

    def start(self) -> None:
        """Starts the server in a background worker thread."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops the server gracefully."""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

    def _serve_loop(self) -> None:
        """Main non-blocking server event loop."""
        if self.protocol == "udp":
            self._serve_udp()
        else:
            self._serve_tcp()

    def _serve_udp(self) -> None:
        """UDP Server implementation for lowest latency (<0.4 ms roundtrip)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(0.2)
        self._sock = sock

        while self._running:
            try:
                data, addr = sock.recvfrom(2048)
                if not data:
                    continue
                try:
                    resp = self.handle_datagram(data, addr)
                    sock.sendto(resp, addr)
                except Exception as ex:
                    # Send fault or drop malformed packet
                    pass
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception:
                pass

    def _serve_tcp(self) -> None:
        """TCP Server implementation with persistent client connection."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(0.5)
        self._sock = server

        while self._running:
            try:
                conn, addr = server.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(0.5)
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                while self._running:
                    try:
                        raw = conn.recv(153)
                        if not raw or len(raw) != 153:
                            break
                        resp = self.handle_datagram(raw, addr)
                        conn.sendall(resp)
                    except socket.timeout:
                        continue
                    except Exception:
                        break

    def get_status_dict(self) -> Dict[str, Any]:
        """Returns live statistics."""
        st = dict(self.stats)
        if self._start_time > 0:
            st["uptime_seconds"] = time.time() - self._start_time
        return st


def main():
    parser = argparse.ArgumentParser(description="AMD Kria KR260 Ethernet Server (Author: Brian A. Hadian)")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5005, help="Port to listen on (default: 5005)")
    parser.add_argument("--protocol", choices=["udp", "tcp"], default="udp", help="Transport protocol")
    parser.add_argument("--mock", action="store_true", help="Force mock PL mode for laptop testing")
    args = parser.parse_args()

    print("========================================================================")
    print("  PROJECT HEPHAESTUS: AMD KRIA KR260 ETHERNET SERVER")
    print("  Author: Brian A. Hadian")
    print("========================================================================")
    print(f"Target: Jetson Orin Nano <-> AMD Kria KR260 High-Speed Link")
    print(f"Listening on: {args.host}:{args.port} ({args.protocol.upper()})")
    print(f"PL Interface: AXI4-Lite Direct Memory-Mapped (0xA000_0000)")
    print("------------------------------------------------------------------------")

    server = KR260EthernetServer(
        host=args.host,
        port=args.port,
        protocol=args.protocol,
        mock=args.mock
    )
    server.start()

    try:
        while True:
            time.sleep(1.0)
            st = server.get_status_dict()
            sys.stdout.write(
                f"\r[STATUS] Rx: {st['packets_received']} | "
                f"Tx: {st['packets_responded']} | "
                f"CRC Err: {st['crc_errors']} | "
                f"Latency: {st['last_latency_us']:.1f} us | "
                f"Client: {st['client_endpoint'] or 'None'}  "
            )
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nStopping Ethernet Server...")
    finally:
        server.stop()
        print("Server shutdown complete.")


if __name__ == "__main__":
    main()
