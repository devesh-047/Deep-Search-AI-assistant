"""
Frame Captioner
================
Generates natural language descriptions for video frames using a Vision
Language Model (VLM).  Frames that contain no OCR-readable text can still
be semantically understood through these captions, dramatically improving
video RAG quality.

Default model: ``Salesforce/blip-image-captioning-base``

Optional OpenVINO acceleration is supported when the ``optimum-intel``
package is installed.

Dependencies::

    pip install transformers Pillow

Optional::

    pip install optimum[openvino]
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FrameCaption:
    """Caption generated for a single video frame.

    Attributes
    ----------
    frame_path : str
        Path to the frame image.
    timestamp : float
        Video timestamp of this frame (seconds).
    caption : str
        Generated natural-language description.
    """
    frame_path: str
    timestamp: float
    caption: str


class FrameCaptioner:
    """Generate captions for sampled video frames using BLIP.

    Usage::

        captioner = FrameCaptioner(model_name="Salesforce/blip-image-captioning-base")
        captions = captioner.caption_batch(sampled_frames, interval=5)
        for c in captions:
            print(f"[{c.timestamp:.1f}s] {c.caption}")

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the captioning model.
    use_openvino : bool
        If True, attempt to load the model with OpenVINO acceleration.
    device : str
        PyTorch device string (``"cpu"`` or ``"cuda"``).  Ignored when
        using OpenVINO.
    max_new_tokens : int
        Maximum number of tokens in generated captions.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        use_openvino: bool = False,
        device: str = "cpu",
        max_new_tokens: int = 50,
    ):
        self.model_name = model_name
        self.use_openvino = use_openvino
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def caption_batch(
        self,
        frames,
        interval: int = 1,
    ) -> List[FrameCaption]:
        """Generate captions for a batch of ``SampledFrame`` objects.

        Parameters
        ----------
        frames : list
            List of ``SampledFrame`` objects (from ``FrameSampler.sample()``).
        interval : int
            Caption every Nth frame (default: 1 = every frame).  Setting
            this to 5 means only frames 0, 5, 10, 15, … are captioned.

        Returns
        -------
        List[FrameCaption]
            Captions for the selected frames.
        """
        if not frames:
            return []

        model, processor = self._get_model()
        if model is None or processor is None:
            return []

        interval = max(1, interval)
        selected = [f for i, f in enumerate(frames) if i % interval == 0]

        captions: List[FrameCaption] = []

        pbar = tqdm(
            selected,
            desc="Captioning frames (BLIP)",
            unit="frame",
            leave=False,
        )

        for frame in pbar:
            caption = self._generate_caption(frame.frame_path, model, processor)
            if caption:
                captions.append(FrameCaption(
                    frame_path=frame.frame_path,
                    timestamp=frame.timestamp,
                    caption=caption,
                ))

        logger.info(
            "Captioned %d/%d frames (interval=%d)",
            len(captions), len(frames), interval,
        )
        return captions

    def generate_caption(self, image_path: str) -> str:
        """Generate a caption for a single image.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        str
            Generated caption, or empty string on failure.
        """
        model, processor = self._get_model()
        if model is None or processor is None:
            return ""
        return self._generate_caption(image_path, model, processor)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_caption(self, image_path: str, model, processor) -> str:
        """Run BLIP inference on a single image."""
        try:
            from PIL import Image
        except ImportError:
            logger.error("Pillow is not installed.  pip install Pillow")
            return ""

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            logger.debug("Could not open image %s: %s", image_path, exc)
            return ""

        try:
            inputs = processor(images=image, return_tensors="pt")

            # Move inputs to correct device for PyTorch (not needed for OV).
            if not self.use_openvino:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            caption = processor.decode(outputs[0], skip_special_tokens=True).strip()
            return caption
        except Exception as exc:
            logger.debug("Caption generation failed for %s: %s", image_path, exc)
            return ""

    def _get_model(self):
        """Lazy-load the BLIP model and processor."""
        if self._model is not None:
            return self._model, self._processor

        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
        except ImportError:
            logger.error(
                "transformers is not installed.  "
                "Install with: pip install transformers"
            )
            return None, None

        logger.info(
            "Loading BLIP captioning model: %s (this may take a moment "
            "on first run)", self.model_name,
        )

        # Try OpenVINO path first if requested.
        if self.use_openvino:
            try:
                from optimum.intel import OVModelForVision2Seq
                self._processor = BlipProcessor.from_pretrained(self.model_name)
                self._model = OVModelForVision2Seq.from_pretrained(
                    self.model_name, export=True,
                )
                logger.info("BLIP loaded with OpenVINO acceleration")
                return self._model, self._processor
            except ImportError:
                logger.warning(
                    "optimum-intel not available — falling back to PyTorch"
                )
            except Exception as exc:
                logger.warning(
                    "OpenVINO BLIP load failed (%s) — falling back to PyTorch",
                    exc,
                )

        # PyTorch fallback.
        try:
            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
            ).to(self.device)
            self._model.eval()
            logger.info("BLIP loaded on %s (PyTorch)", self.device)
        except Exception as exc:
            logger.error("Failed to load BLIP model '%s': %s",
                         self.model_name, exc)
            return None, None

        return self._model, self._processor
