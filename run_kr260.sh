#!/usr/bin/env bash
# ==============================================================================
# AMD Kria KR260 Interactive Test & Program Runner (Root Launcher)
# Author: Brian A. Hadian
# Project: Hephaestus
# ==============================================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT_DIR}/deploy_kr260/run_kr260.sh" "$@"
