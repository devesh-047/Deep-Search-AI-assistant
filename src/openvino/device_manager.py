"""
OpenVINO Device Manager
========================
Detects available OpenVINO-compatible hardware and provides a simple API
for device selection across the pipeline.

Intel AI-PC devices:
    CPU  -- always available, baseline performance
    GPU  -- Intel integrated GPU (iGPU); good for throughput workloads
    NPU  -- Neural Processing Unit on Meteor Lake+; best perf/watt

Virtual/meta devices:
    AUTO  -- OpenVINO automatically picks the best available device.
             It starts inference on CPU immediately while loading the
             model on GPU/NPU in the background, then switches over.
    MULTI -- Run inference on multiple devices simultaneously.
             E.g. MULTI:CPU,GPU splits batches across both.

Design notes:
    - Device selection is centralised here so that every module that needs
      to pick a device calls ``DeviceManager.select()`` instead of
      hard-coding the string.
    - The manager reads the preferred device from ``configs/settings.yaml``
      but falls back gracefully if that device is not present.

Learning TODO (all implemented):
    1. ✅ Install openvino and run ``list_devices()`` to see what your machine
       reports.
    2. ✅ Compare inference latency on CPU vs GPU for the embedding model.
    3. ✅ Investigate ``AUTO`` and ``MULTI`` device plugins for dynamic dispatch.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Path to the project settings file
SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "settings.yaml"


def load_settings() -> dict:
    """
    Load settings from configs/settings.yaml.

    Why a separate function?
    ------------------------
    Multiple modules need to read settings. Centralising the load
    means we can add validation, defaults, and environment variable
    overrides in one place.

    Returns:
        The parsed YAML as a dict, or empty dict if file not found.
    """
    if not SETTINGS_PATH.exists():
        logger.warning("Settings file not found: %s", SETTINGS_PATH)
        return {}
    try:
        with open(SETTINGS_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to read settings: %s", exc)
        return {}


class DeviceManager:
    """
    Centralised OpenVINO device detection and selection.

    The manager:
      1. Detects all hardware devices via OpenVINO Core
      2. Reads the preferred device from configs/settings.yaml
      3. Validates the preferred device is available
      4. Falls back gracefully to CPU if not
      5. Supports AUTO and MULTI meta-devices
      6. Provides device properties and benchmarking

    Usage::

        dm = DeviceManager()
        devices = dm.list_devices()
        device = dm.select(preferred="GPU")  # falls back to CPU

    With settings.yaml::

        dm = DeviceManager()
        device = dm.select_from_settings()  # reads openvino.device from yaml

    Benchmarking::

        results = dm.benchmark_devices(model_xml="models/ov/.../model.xml")
    """

    def __init__(self):
        self._core = None
        self._devices: Optional[List[str]] = None
        self._settings: dict = {}
        self._initialise()

    def _initialise(self) -> None:
        """Try to create an OpenVINO Core instance and load settings."""
        # Load settings
        self._settings = load_settings()

        # Initialise OpenVINO
        try:
            from openvino.runtime import Core
            self._core = Core()
            self._devices = self._core.available_devices
            logger.info("OpenVINO devices: %s", self._devices)
        except ImportError:
            logger.warning(
                "openvino is not installed.  Device detection unavailable.  "
                "Install: pip install openvino"
            )
            self._devices = []
        except Exception as exc:
            logger.error("Failed to initialise OpenVINO Core: %s", exc)
            self._devices = []

    @property
    def core(self):
        """Access the underlying OpenVINO Core instance."""
        return self._core

    def list_devices(self) -> List[str]:
        """Return a list of available device strings (e.g. ['CPU', 'GPU'])."""
        return list(self._devices) if self._devices else []

    def select(self, preferred: str = "CPU") -> str:
        """
        Select an inference device.

        If the preferred device is available, return it.  Otherwise fall
        back to CPU (always available when openvino is installed).

        Supports meta-devices:
            - "AUTO"         — let OpenVINO pick the best device
            - "MULTI:CPU,GPU" — use multiple devices simultaneously

        How AUTO works:
        ---------------
        AUTO is an OpenVINO plugin that:
          1. Immediately starts inference on CPU (fastest to compile)
          2. In the background, compiles the model for GPU/NPU
          3. Once GPU/NPU is ready, transparently switches to it
          4. The user sees low latency from the start, then gets
             the performance benefit of the accelerator

        How MULTI works:
        ----------------
        MULTI splits inference requests across multiple devices:
          - "MULTI:CPU,GPU" → sends batches to both CPU and GPU
          - Useful for throughput-heavy workloads
          - Each device gets its own compiled model

        Args:
            preferred : device string to try first.

        Returns:
            The device string to pass to ``core.compile_model(device_name=...)``.
        """
        devices = self.list_devices()

        # Handle AUTO — always valid if OpenVINO is installed
        if preferred.upper() == "AUTO":
            if self._core is not None:
                logger.info(
                    "Selected device: AUTO (available devices: %s)", devices
                )
                return "AUTO"
            logger.warning("AUTO requested but OpenVINO not installed, falling back to CPU")
            return "CPU"

        # Handle MULTI:DEV1,DEV2 — validate each sub-device
        if preferred.upper().startswith("MULTI:"):
            sub_devices = preferred.split(":")[1].split(",")
            valid_subs = [d for d in sub_devices if d in devices]
            if len(valid_subs) >= 2:
                multi_str = "MULTI:" + ",".join(valid_subs)
                logger.info("Selected device: %s", multi_str)
                return multi_str
            elif valid_subs:
                logger.warning(
                    "MULTI requested but only '%s' available, using single device",
                    valid_subs[0],
                )
                return valid_subs[0]
            else:
                logger.warning("No MULTI sub-devices available, falling back to CPU")
                return "CPU"

        # Standard device selection
        if preferred in devices:
            logger.info("Selected device: %s", preferred)
            return preferred

        if "CPU" in devices:
            logger.warning(
                "Preferred device '%s' not available (have: %s). Falling back to CPU.",
                preferred,
                devices,
            )
            return "CPU"

        logger.error(
            "No OpenVINO devices available.  Returning '%s' anyway "
            "(inference will fail unless openvino is installed).",
            preferred,
        )
        return preferred

    def select_from_settings(self) -> str:
        """
        Read the preferred device from configs/settings.yaml and select it.

        The settings file has:
            openvino:
              device: "CPU"  # or "GPU", "NPU", "AUTO"

        This method reads that value and calls select() with it,
        providing automatic fallback if the device isn't available.

        Returns:
            The selected device string.
        """
        ov_settings = self._settings.get("openvino", {})
        preferred = ov_settings.get("device", "CPU")
        logger.info(
            "Settings file requests device: '%s'", preferred
        )
        selected = self.select(preferred)
        if selected != preferred:
            logger.info(
                "Device fallback: '%s' → '%s'", preferred, selected
            )
        return selected

    def is_openvino_enabled(self) -> bool:
        """
        Check if OpenVINO is enabled in settings.yaml.

        The settings file has:
            openvino:
              enabled: true/false

        When enabled, the pipeline should use OVEmbeddingEncoder
        instead of the default sentence-transformers encoder.
        """
        ov_settings = self._settings.get("openvino", {})
        return bool(ov_settings.get("enabled", False))

    def get_embedding_model_path(self) -> str:
        """
        Get the OpenVINO IR path for the embedding model from settings.

        Returns:
            Path to the .xml file, or empty string if not configured.
        """
        ov_settings = self._settings.get("openvino", {})
        return ov_settings.get("embedding_model_ir", "")

    def device_properties(self, device: str) -> Dict[str, str]:
        """
        Return known properties for a device (name, architecture, etc.).

        Useful for debug logging and the ``cli.py devices`` command.

        Properties queried:
            - FULL_DEVICE_NAME: human-readable name
              e.g. "13th Gen Intel(R) Core(TM) i5-13420H"
            - DEVICE_ARCHITECTURE: internal architecture identifier
            - OPTIMAL_NUMBER_OF_INFER_REQUESTS: how many parallel
              inference requests the device can handle efficiently
            - RANGE_FOR_ASYNC_INFER_REQUESTS: min/max async requests
        """
        if self._core is None:
            return {"error": "OpenVINO not installed"}
        props: Dict[str, str] = {}

        # Standard properties
        for key in (
            "FULL_DEVICE_NAME",
            "DEVICE_ARCHITECTURE",
            "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
        ):
            try:
                value = self._core.get_property(device, key)
                props[key] = str(value)
            except Exception:
                pass

        # Try to get supported properties list
        try:
            supported = self._core.get_property(device, "SUPPORTED_PROPERTIES")
            props["SUPPORTED_PROPERTIES_COUNT"] = str(len(supported))
        except Exception:
            pass

        return props

    def device_summary(self) -> List[Dict[str, str]]:
        """
        Get a summary of all available devices with properties.

        Returns a list of dicts, each containing:
            - device: the device string
            - name: full human-readable name
            - architecture: internal architecture
            - optimal_requests: recommended parallel requests
        """
        summaries = []
        for device in self.list_devices():
            props = self.device_properties(device)
            summaries.append({
                "device": device,
                "name": props.get("FULL_DEVICE_NAME", "Unknown"),
                "architecture": props.get("DEVICE_ARCHITECTURE", ""),
                "optimal_requests": props.get(
                    "OPTIMAL_NUMBER_OF_INFER_REQUESTS", ""
                ),
            })
        return summaries

    def benchmark_devices(
        self,
        model_xml: str,
        n_iterations: int = 20,
        batch_size: int = 1,
        seq_len: int = 128,
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark a model across all available devices.

        Compiles the model on each device, runs dummy inference
        multiple times, and reports latency statistics.

        Why benchmark?
        --------------
        Different devices have different strengths:
        - CPU: low latency for small models, always available
        - GPU: higher throughput for large batches
        - NPU: best performance-per-watt for sustained inference
        - AUTO: may add startup latency but optimises long-running tasks

        The benchmark helps you choose the right device for your
        workload without guessing.

        Args:
            model_xml    : path to the OpenVINO IR .xml file
            n_iterations : number of inference runs per device
            batch_size   : batch size for dummy inputs (default: 1)
            seq_len      : sequence length for dummy inputs (default: 128)

        Returns:
            Dict mapping device name to timing stats.
        """
        if self._core is None:
            return {"error": {"msg": "OpenVINO not installed"}}

        import numpy as np
        xml_path = Path(model_xml)
        if not xml_path.exists():
            return {"error": {"msg": f"Model not found: {model_xml}"}}

        model = self._core.read_model(str(xml_path))
        results = {}

        for device in self.list_devices():
            try:
                logger.info("Benchmarking on %s...", device)

                # Compile for this device
                compiled = self._core.compile_model(model, device)

                # Create dummy inputs with concrete shapes
                # For BERT-style models: input_ids, attention_mask, (token_type_ids)
                inputs = {
                    "input_ids": np.ones((batch_size, seq_len), dtype=np.int64),
                    "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                }
                
                # Add token_type_ids if the model expects it
                input_names = [inp.get_any_name() for inp in compiled.inputs]
                if "token_type_ids" in input_names:
                    inputs["token_type_ids"] = np.zeros((batch_size, seq_len), dtype=np.int64)

                # Warmup (3 runs to warm up caches)
                for _ in range(3):
                    compiled(inputs)

                # Timed runs
                times = []
                for _ in range(n_iterations):
                    start = time.perf_counter()
                    compiled(inputs)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed * 1000)  # convert to ms

                times_arr = np.array(times)
                results[device] = {
                    "mean_ms": float(times_arr.mean()),
                    "std_ms": float(times_arr.std()),
                    "min_ms": float(times_arr.min()),
                    "max_ms": float(times_arr.max()),
                    "median_ms": float(np.median(times_arr)),
                }
                logger.info(
                    "%s: mean=%.2fms, min=%.2fms, max=%.2fms",
                    device,
                    results[device]["mean_ms"],
                    results[device]["min_ms"],
                    results[device]["max_ms"],
                )

            except Exception as exc:
                logger.warning("Benchmark failed on %s: %s", device, exc)
                results[device] = {"error": str(exc)}

        return results
