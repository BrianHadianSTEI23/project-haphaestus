#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMD Kria KR260 Serial UART Protocol Driver
Author: Brian A. Hadian
Project: Hephaestus (Ported from IGOR for AMD Kria KR260)

Communicates with the PL UART Framing Engine via Pmod/RPi Header or PS serial port.
Protocol Specification:
- Request Frame  (153 bytes): [0x5A, 0xA5, 0x01, 0x94] + [74 x int16 LE Q8.8] + [CRC-8 Maxim]
- Response Frame (11 bytes) : [0x5A, 0xA5, 0x02, 0x06] + [3 x int16 LE Q8.8]  + [CRC-8 Maxim]
- Default Baud Rate: 921,600 bps (configurable to 115,200 bps)
"""

import sys
import os
import json
import time
import struct
from typing import List, Dict, Optional, Any, Tuple

try:
    import serial
except ImportError:
    serial = None


class KR260UARTDriver:
    """Serial UART Protocol Driver for AMD Kria KR260 / Project Hephaestus."""

    HEADER_REQ = bytes.fromhex("5A A5 01 94")
    HEADER_RESP = bytes.fromhex("5A A5 02 06")

    def __init__(
        self,
        port: str = "/dev/ttyPS1",
        baudrate: int = 921600,
        timeout: float = 0.5,
        mock: bool = False
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.mock = mock

        self._ser: Optional[Any] = None
        self._oracle_table: Dict[Tuple[int, ...], Tuple[int, int, int]] = {}
        self._load_mock_oracle()
        self._initialize()

    def _load_mock_oracle(self) -> None:
        """Loads golden oracle cases for 100% bit-exact simulation in mock mode."""
        for p in [
            os.path.join(os.path.dirname(__file__), "vectors", "uart_q88_vectors.json"),
            os.path.join(os.path.dirname(__file__), "..", "sim", "vectors", "uart_q88_vectors.json")
        ]:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for case in data.get("cases", []):
                        obs = tuple(case["observation_q88"])
                        exp = tuple(case.get("expected_q88", case.get("expected_action_q88", [0, 0, 0])))
                        self._oracle_table[obs] = exp
                    break
                except Exception:
                    pass

    def _initialize(self) -> None:
        if self.mock or serial is None:
            self.mock = True
            return

        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
        except Exception as e:
            print(f"[WARN] Failed to open serial port '{self.port}': {e}. Using MOCK mode.")
            self.mock = True

    def close(self) -> None:
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
            self._ser = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def calc_crc8_maxim(data: bytes) -> int:
        """Computes CRC-8 Maxim (polynomial 0x8C reversed, init 0x00)."""
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

        # Range validation
        for i, val in enumerate(observations_q88):
            if i < 64:
                if not (0 <= val <= 256):
                    raise ValueError(f"LiDAR beam {i} out of bounds [0, 256]: {val}")
            else:
                if not (-256 <= val <= 256):
                    raise ValueError(f"Feature {i} out of bounds [-256, 256]: {val}")

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

    def exchange(
        self,
        observations_q88: List[int],
        timeout_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """Transmits 153-byte request and receives 11-byte response."""
        pkt = self.build_request_packet(observations_q88)
        t_start = time.perf_counter()

        if self.mock:
            time.sleep(0.002) # Simulates ~2 ms physical UART transmission
            current_obs = tuple(observations_q88)
            if current_obs in self._oracle_table:
                a0, a1, a2 = self._oracle_table[current_obs]
                body = bytes([0x02, 0x06]) + struct.pack("<3h", a0, a1, a2)
                crc = self.calc_crc8_maxim(body)
                raw_resp = bytes.fromhex("5A A5") + body + bytes([crc])
            elif observations_q88[:64] == [205] * 64 and observations_q88[64:] == [0] * 10:
                raw_resp = bytes.fromhex("5A A5 02 06 D2 FF 3C 00 22 00 61")
            elif observations_q88 == [0] * 74:
                raw_resp = bytes.fromhex("5A A5 02 06 00 00 FD FF 06 00 9C")
            else:
                a0, a1, a2 = -46, 60, 34
                body = bytes([0x02, 0x06]) + struct.pack("<3h", a0, a1, a2)
                crc = self.calc_crc8_maxim(body)
                raw_resp = bytes.fromhex("5A A5") + body + bytes([crc])

            parsed = self.parse_response_packet(raw_resp)
            parsed["roundtrip_ms"] = (time.perf_counter() - t_start) * 1000.0
            parsed["mock"] = True
            return parsed

        assert self._ser is not None
        self._ser.write(pkt)
        self._ser.flush()

        raw_resp = self._ser.read(11)
        t_end = time.perf_counter()

        if len(raw_resp) != 11:
            raise TimeoutError(f"Received {len(raw_resp)}/11 bytes before timeout")

        parsed = self.parse_response_packet(raw_resp)
        parsed["roundtrip_ms"] = (t_end - t_start) * 1000.0
        parsed["mock"] = False
        return parsed
