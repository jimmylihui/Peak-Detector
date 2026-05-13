#!/usr/bin/env python3
"""Copy clean BCG Arrhythmia split arrays into the baseline data location."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from baseline_array_utils import copy_split_arrays


SOURCE_DIR = Path("/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/BCG_Arrhythmia")
OUT_DIR = Path("/path/to/workspace/project-BCG-LLM/combined_data/baseline_arrays/BCG_Arrhythmia")


def main() -> None:
    copy_split_arrays(SOURCE_DIR, OUT_DIR)


if __name__ == "__main__":
    main()
