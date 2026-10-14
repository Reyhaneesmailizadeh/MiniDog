"""Vendored Meta Perception Encoder (PE) model code.

Copied verbatim from the diffusion-bench reference implementation
(src/encoders/models/) so minidog stays self-contained. Weights are fetched
from HuggingFace at first use: facebook/PE-Spatial-B16-512.
"""
from .pe import VisionTransformer
from .pe_config import PE_VISION_CONFIG

__all__ = ["VisionTransformer", "PE_VISION_CONFIG"]
