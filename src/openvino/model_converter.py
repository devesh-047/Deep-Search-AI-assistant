"""
OpenVINO Model Converter
=========================
Automates conversion of ONNX models to OpenVINO Intermediate Representation
(IR) for optimised inference.

This module wraps the OpenVINO ``openvino.tools.ovc`` API (preferred) or
the legacy ``mo`` command-line tool so that conversion can be triggered
from Python rather than the command line.

Conversion targets for this project:
    1. Embedding model:  all-MiniLM-L6-v2  (ONNX -> IR)
    2. OCR models:       PaddlePaddle detection + recognition (ONNX -> IR)
    3. LLM:              Mistral 7B via optimum-intel (future, complex)

Learning TODO (all implemented):
    1. ✅ Export all-MiniLM-L6-v2 to ONNX using optimum-cli or the script below.
    2. ✅ Run ``convert_onnx_to_ir()`` and verify IR output.
    3. ✅ Load the IR in ``OVEmbeddingEncoder`` and compare to PyTorch output.
    4. ✅ Explore FP16 compression (``compress_to_fp16=True``) and measure
       accuracy vs speed trade-offs.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def convert_onnx_to_ir(
    onnx_path: str,
    output_dir: str,
    model_name: Optional[str] = None,
    compress_to_fp16: bool = False,
) -> Path:
    """
    Convert an ONNX model to OpenVINO IR (xml + bin).

    Uses ``openvino.tools.ovc`` if available (OpenVINO 2023.1+), otherwise
    falls back to the legacy ``mo`` command-line tool.

    The conversion process:
        1. Parse the ONNX graph (operators, shapes, data types)
        2. Map each ONNX operator to OpenVINO's internal operators
        3. Apply graph optimizations (operator fusion, constant folding)
        4. Serialize to .xml (graph structure) + .bin (weights)

    If compress_to_fp16=True:
        Weights are stored as FP16 instead of FP32. This:
        - Halves the model file size (from ~90MB to ~45MB)
        - May slightly reduce accuracy (typically <0.1% difference)
        - Does NOT change computation precision on CPU (still FP32)
        - DOES use FP16 compute on GPU/NPU if supported

    Args:
        onnx_path       : path to the .onnx model file
        output_dir      : directory for the output .xml and .bin files
        model_name      : optional name for the output files
        compress_to_fp16: if True, compress weights to FP16

    Returns:
        Path to the generated .xml file.

    Raises:
        FileNotFoundError : if the ONNX file does not exist
        RuntimeError      : if conversion fails
    """
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_file}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_name = (model_name or onnx_file.stem) + ".xml"
    xml_path = out_dir / xml_name

    # Try the modern Python API first.
    try:
        import openvino as ov

        logger.info("Converting %s -> OpenVINO IR (ovc API)", onnx_path)

        # Try modern ovc.convert_model first (OpenVINO 2023.1+)
        try:
            from openvino.tools import ovc
            ov_model = ovc.convert_model(str(onnx_file))
        except (ImportError, AttributeError):
            # Fallback: use openvino.convert_model (OpenVINO 2024+)
            ov_model = ov.convert_model(str(onnx_file))

        # Save the model (with optional FP16 compression)
        ov.save_model(
            ov_model,
            str(xml_path),
            compress_to_fp16=compress_to_fp16,
        )
        logger.info(
            "IR saved to %s (FP16=%s)", xml_path, compress_to_fp16
        )
        return xml_path

    except ImportError:
        logger.info("OpenVINO Python API not available, trying mo CLI")

    # Fallback: legacy Model Optimizer CLI.
    cmd = [
        "mo",
        "--input_model", str(onnx_file),
        "--output_dir", str(out_dir),
    ]
    if model_name:
        cmd += ["--model_name", model_name]
    if compress_to_fp16:
        cmd += ["--compress_to_fp16"]

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("mo failed:\n%s", result.stderr)
        raise RuntimeError(f"Model Optimizer failed: {result.stderr[:500]}")

    logger.info("IR saved to %s", xml_path)
    return xml_path


def export_sentence_transformer_to_onnx(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "models/onnx/all-MiniLM-L6-v2",
) -> Path:
    """
    Export a sentence-transformers model to ONNX format.

    This is a convenience wrapper around optimum-cli.

    Prerequisites:
        pip install optimum[exporters] optimum-intel

    Returns:
        Path to the output directory containing the ONNX model.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", model_name,
        str(out),
    ]
    logger.info("Exporting to ONNX: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("ONNX export failed:\n%s", result.stderr)
        raise RuntimeError(f"ONNX export failed: {result.stderr[:500]}")

    logger.info("ONNX model exported to %s", out)
    return out


def convert_embedding_model_pipeline(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    onnx_dir: str = "models/onnx/all-MiniLM-L6-v2",
    ir_dir: str = "models/ov/all-MiniLM-L6-v2",
    compress_to_fp16: bool = False,
) -> Path:
    """
    Full pipeline: export to ONNX then convert to OpenVINO IR.

    This is the convenience function that combines both steps:
        1. Export HuggingFace model -> ONNX
        2. Convert ONNX -> OpenVINO IR (.xml + .bin)

    Usage:
        from src.openvino.model_converter import convert_embedding_model_pipeline
        xml_path = convert_embedding_model_pipeline()
        print(f"IR model at: {xml_path}")

    Args:
        model_name     : HuggingFace model ID
        onnx_dir       : directory for ONNX output
        ir_dir         : directory for IR output
        compress_to_fp16: compress weights to FP16

    Returns:
        Path to the generated .xml file.
    """
    onnx_out = Path(onnx_dir)
    onnx_model = onnx_out / "model.onnx"

    # Step 1: Export to ONNX (if not already done)
    if not onnx_model.exists():
        logger.info("Step 1: Exporting %s to ONNX...", model_name)
        export_sentence_transformer_to_onnx(model_name, onnx_dir)
    else:
        logger.info("Step 1: ONNX model already exists at %s, skipping export", onnx_model)

    if not onnx_model.exists():
        raise FileNotFoundError(
            f"ONNX export completed but model.onnx not found at {onnx_model}. "
            f"Check the export output directory."
        )

    # Step 2: Convert ONNX to IR
    logger.info("Step 2: Converting ONNX to OpenVINO IR...")
    xml_path = convert_onnx_to_ir(
        onnx_path=str(onnx_model),
        output_dir=ir_dir,
        compress_to_fp16=compress_to_fp16,
    )

    logger.info("Pipeline complete! IR model at: %s", xml_path)
    return xml_path
