#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMD Kria KR260 FPGA Manager & Bitstream Programming Manager
Author: Brian A. Hadian
Project: Hephaestus (Ported from IGOR for AMD Kria KR260)

Manages PL bitstream programming and Device Tree Overlay configuration
directly from the ARM Cortex-A53 PS on Ubuntu 22.04 LTS.
Supports:
1. AMD xmutil (official Kria app manager)
2. Linux sysfs FPGA Manager (/sys/class/fpga_manager/fpga0/firmware)
3. Offline/mock simulation mode for development on laptop
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

FPGA_SYSFS_PATH = Path("/sys/class/fpga_manager/fpga0")
FIRMWARE_DEST_DIR = Path("/lib/firmware/xilinx")
DEFAULT_APP_NAME = "igor_kr260"


class KR260FPGAManager:
    """Controls FPGA bitstream programming from PS Linux on KR260."""

    def __init__(self, app_name: str = DEFAULT_APP_NAME, mock: bool = False):
        self.app_name = app_name
        self.mock = mock or not FPGA_SYSFS_PATH.exists()
        self.app_dir = Path(__file__).resolve().parent / "firmware"

    def get_state(self) -> str:
        """Reads current FPGA Manager state."""
        if self.mock:
            return "operating (MOCK)"
        state_file = FPGA_SYSFS_PATH / "state"
        if state_file.exists():
            try:
                return state_file.read_text().strip()
            except Exception as e:
                return f"error: {e}"
        return "sysfs_not_found"

    def list_apps(self) -> Dict[str, Any]:
        """Lists available and active Kria apps via xmutil."""
        if self.mock:
            return {
                "active": self.app_name,
                "available": ["k26-starter-kits", self.app_name],
                "source": "mock"
            }

        try:
            res = subprocess.run(["xmutil", "listapps"], capture_output=True, text=True, check=True)
            active_app = None
            for line in res.stdout.splitlines():
                if "Active" in line or "1" in line:
                    parts = line.split()
                    if parts:
                        active_app = parts[0]
            return {
                "raw": res.stdout,
                "active": active_app,
                "source": "xmutil"
            }
        except Exception:
            # Fallback to sysfs check
            return {
                "active": self.get_state(),
                "source": "sysfs"
            }

    def install_firmware_files(self) -> bool:
        """Copies bit.bin, dtbo, and shell.json into /lib/firmware/xilinx/<app_name>/."""
        if self.mock:
            return True

        target_dir = FIRMWARE_DEST_DIR / self.app_name
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            for ext in [".bit.bin", ".dtbo", ".json"]:
                for src in self.app_dir.glob(f"*{ext}"):
                    shutil.copy2(src, target_dir / src.name)
            return True
        except PermissionError:
            print("[ERROR] Writing to /lib/firmware/xilinx requires root/sudo privileges!")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to install firmware files: {e}")
            return False

    def load_app(self) -> bool:
        """Loads and programs the FPGA bitstream using xmutil or direct sysfs."""
        if self.mock:
            print(f"[MOCK] Simulated bitstream load for app '{self.app_name}' successful.")
            return True

        # Ensure files are in place
        if not (FIRMWARE_DEST_DIR / self.app_name).exists():
            if not self.install_firmware_files():
                print("[ERROR] Failed installing firmware files prior to loading.")
                return False

        # Try xmutil unload and load
        try:
            print(f">>> Unloading any active app via xmutil...")
            subprocess.run(["xmutil", "unloadapp"], capture_output=True, text=True)
            print(f">>> Loading app '{self.app_name}' via xmutil...")
            res = subprocess.run(["xmutil", "loadapp", self.app_name], capture_output=True, text=True, check=True)
            print(res.stdout)
            state = self.get_state()
            print(f">>> FPGA Manager State: {state}")
            return "operating" in state
        except Exception as exc:
            print(f"[WARN] xmutil loadapp failed: {exc}. Trying direct FPGA Manager sysfs interface...")

        # Fallback to direct sysfs loading
        try:
            flags_file = FPGA_SYSFS_PATH / "flags"
            fw_file = FPGA_SYSFS_PATH / "firmware"
            bin_name = f"{self.app_name}/{self.app_name}.bit.bin"

            # Reset flags to 0 (full bitstream)
            flags_file.write_text("0\n")
            # Write firmware name relative to /lib/firmware
            fw_file.write_text(f"{bin_name}\n")

            state = self.get_state()
            print(f">>> Direct sysfs programming result: {state}")
            return "operating" in state
        except Exception as e:
            print(f"[ERROR] Direct sysfs programming failed: {e}")
            return False
