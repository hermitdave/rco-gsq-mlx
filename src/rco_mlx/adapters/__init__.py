#!/usr/bin/env python3
"""
Model adapters package for the RCO quantization pipeline.

Each supported model gets a small adapter class that describes:
  - how to enumerate its layers and tensors
  - which tensors to quantize vs keep BF16
  - how to group layers for RCO allocation
  - how to load calibration data

The pipeline (pipeline_rco_mlx.py) calls the adapter; it does not need to
know which model it is running on.
"""

from .base import ModelAdapter, get, register
from .default import DefaultAdapter
from .flint import FlintAdapter
from .lfm2 import LFM2Adapter, LFM2MoEAdapter
from .olmoe import OLMoEAdapter
from .qwen3_5_moe import Qwen3NextMoEAdapter

__all__ = [
    "ModelAdapter",
    "DefaultAdapter",
    "FlintAdapter",
    "LFM2Adapter",
    "LFM2MoEAdapter",
    "OLMoEAdapter",
    "Qwen3NextMoEAdapter",
    "register",
    "get",
]
