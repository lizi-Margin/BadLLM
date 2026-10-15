#!/usr/bin/env python3
"""llamafactory-cli wrapper for the MTT MUSA stack.

Applies the musa_compat shims BEFORE LLaMA-Factory / transformers are
imported, then hands off to the normal LF CLI. On NVIDIA CUDA boxes use
``llamafactory-cli`` directly (no shims needed).

Usage: python tools/lf_musa.py train <config.json> [other lf args]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import musa_compat  # noqa: F401  E402  must precede transformers/LF imports

from llamafactory.cli import main

main()
