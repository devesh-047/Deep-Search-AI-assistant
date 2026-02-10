"""
OpenVINO subpackage -- hardware-aware acceleration helpers.

Modules:
    device_manager   -- detect and select inference devices
    model_converter  -- convert ONNX / PyTorch models to OpenVINO IR

Learning hooks:
    openvino_encoder and openvino_llm live under their respective domain
    packages (src/embeddings/ and src/llm/) so that swap-in requires only
    an import change.  This package provides shared infrastructure.
"""

from src.openvino.device_manager import DeviceManager
