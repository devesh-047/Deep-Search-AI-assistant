"""
Setup Verification Script
==========================
Quick smoke-test that checks whether every required dependency is
importable and correctly versioned.

Run after setting up the virtual environment:
    python scripts/verify_setup.py

Exit codes:
    0  -- all checks passed
    1  -- one or more checks failed
"""

import importlib
import logging
import shutil
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# (module_name, display_name, required)
REQUIRED_PACKAGES = [
    ("numpy", "NumPy", True),
    ("PIL", "Pillow", True),
    ("cv2", "OpenCV", True),
    ("pytesseract", "pytesseract", True),
    ("faiss", "FAISS (CPU)", True),
    ("sentence_transformers", "sentence-transformers", True),
    ("datasets", "HuggingFace datasets", True),
    ("yaml", "PyYAML", True),
    ("tqdm", "tqdm", True),
    ("openvino", "OpenVINO", False),  # optional at this stage
]


def check_python_version() -> bool:
    """Verify Python >= 3.9."""
    v = sys.version_info
    ok = v >= (3, 9)
    logger.info(
        "Python %d.%d.%d %s",
        v.major, v.minor, v.micro,
        "(OK)" if ok else "(FAIL: need >= 3.9)",
    )
    return ok


def check_package(module: str, display: str, required: bool) -> bool:
    """Try to import a package and report its version if available."""
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "unknown")
        logger.info("  %-30s  %s", display, version)
        return True
    except ImportError:
        tag = "MISSING (required)" if required else "MISSING (optional)"
        logger.warning("  %-30s  %s", display, tag)
        return not required  # optional packages don't cause failure


def check_tesseract_binary() -> bool:
    """Verify the tesseract binary is on PATH."""
    path = shutil.which("tesseract")
    if path:
        try:
            result = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            version = result.stdout.split("\n")[0]
            logger.info("  %-30s  %s", "Tesseract binary", version)
            return True
        except Exception:
            pass
    logger.warning(
        "  %-30s  NOT FOUND (install: sudo apt install tesseract-ocr)",
        "Tesseract binary",
    )
    return False


def check_ollama() -> bool:
    """Check if Ollama server is reachable."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:11434/api/tags", method="GET"
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            logger.info("  %-30s  reachable", "Ollama server")
            return True
    except Exception:
        logger.warning(
            "  %-30s  NOT REACHABLE (start with: ollama serve)",
            "Ollama server",
        )
        return False


def check_raw_data() -> bool:
    """Verify the raw data directory exists and has expected datasets."""
    from pathlib import Path
    raw = Path("/mnt/d/Openvino-project/data/raw")
    # Fallback for native Windows execution.
    if not raw.exists():
        raw = Path(r"D:\Openvino-project\data\raw")
    if not raw.exists():
        logger.warning("  %-30s  NOT FOUND", "Raw data directory")
        return False

    datasets = sorted(d.name for d in raw.iterdir() if d.is_dir())
    logger.info("  %-30s  %s", "Raw data directory", str(raw))
    logger.info("  %-30s  %s", "Datasets found", ", ".join(datasets))
    return True


def main() -> None:
    logger.info("=" * 60)
    logger.info("Deep Search AI Assistant -- Setup Verification")
    logger.info("=" * 60)

    all_ok = True

    logger.info("\n[1/5] Python version")
    all_ok &= check_python_version()

    logger.info("\n[2/5] Python packages")
    for module, display, required in REQUIRED_PACKAGES:
        all_ok &= check_package(module, display, required)

    logger.info("\n[3/5] Tesseract OCR binary")
    all_ok &= check_tesseract_binary()

    logger.info("\n[4/5] Ollama LLM server")
    check_ollama()  # advisory only -- don't fail on this

    logger.info("\n[5/5] Raw data")
    all_ok &= check_raw_data()

    logger.info("\n" + "=" * 60)
    if all_ok:
        logger.info("All required checks PASSED.")
    else:
        logger.error("Some checks FAILED.  Fix the issues above and re-run.")
    logger.info("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
