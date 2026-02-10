"""
OpenVINO Device Manager
========================
Detects available OpenVINO-compatible hardware and provides a simple API
for device selection across the pipeline.

Intel AI-PC devices:
    CPU  -- always available, baseline performance
    GPU  -- Intel integrated GPU (iGPU); good for throughput workloads
    NPU  -- Neural Processing Unit on Meteor Lake+; best perf/watt

Design notes:
    - Device selection is centralised here so that every module that needs
      to pick a device calls ``DeviceManager.select()`` instead of
      hard-coding the string.
    - The manager reads the preferred device from ``configs/settings.yaml``
      but falls back gracefully if that device is not present.

Learning TODO:
    1. Install openvino and run ``list_devices()`` to see what your machine
       reports.
    2. Compare inference latency on CPU vs GPU for the embedding model.
    3. Investigate ``AUTO`` and ``MULTI`` device plugins for dynamic dispatch.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Centralised OpenVINO device detection and selection.

    Usage::

        dm = DeviceManager()
        devices = dm.list_devices()
        device = dm.select(preferred="GPU")  # falls back to CPU
    """

    def __init__(self):
        self._core = None
        self._devices: Optional[List[str]] = None
        self._initialise()

    def _initialise(self) -> None:
        """Try to create an OpenVINO Core instance."""
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

    def list_devices(self) -> List[str]:
        """Return a list of available device strings (e.g. ['CPU', 'GPU'])."""
        return list(self._devices) if self._devices else []

    def select(self, preferred: str = "CPU") -> str:
        """
        Select an inference device.

        If the preferred device is available, return it.  Otherwise fall
        back to CPU (always available when openvino is installed).

        Args:
            preferred : device string to try first ("CPU", "GPU", "NPU").

        Returns:
            The device string to pass to ``core.compile_model(device_name=...)``.
        """
        devices = self.list_devices()
        if preferred in devices:
            logger.info("Selected device: %s", preferred)
            return preferred
        if "CPU" in devices:
            logger.warning(
                "Preferred device '%s' not available.  Falling back to CPU.",
                preferred,
            )
            return "CPU"
        logger.error(
            "No OpenVINO devices available.  Returning '%s' anyway "
            "(inference will fail unless openvino is installed).",
            preferred,
        )
        return preferred

    def device_properties(self, device: str) -> Dict[str, str]:
        """
        Return known properties for a device (name, architecture, etc.).

        Useful for debug logging and the ``cli.py devices`` command.
        """
        if self._core is None:
            return {"error": "OpenVINO not installed"}
        props: Dict[str, str] = {}
        try:
            for key in ("FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE"):
                try:
                    props[key] = self._core.get_property(device, key)
                except Exception:
                    pass
        except Exception as exc:
            props["error"] = str(exc)
        return props
