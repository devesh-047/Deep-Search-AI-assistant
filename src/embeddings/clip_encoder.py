"""
CLIP Encoder
=============
Multimodal encoder using OpenAI's CLIP (Contrastive Language-Image
Pre-Training) for image and text embedding.

This enables **visual** retrieval — queries like "find slides with charts"
or "show images containing invoices" work even when OCR text is minimal,
because CLIP embeds images and text into a shared vector space.

Model:
    openai/clip-vit-base-patch32
        - 512-dimensional embeddings for both images and text
        - Lightweight ViT-B/32 vision encoder
        - Good balance of speed and quality

OpenVINO acceleration:
    When ``use_openvino=True`` and the ``openvino`` package is installed,
    the CLIP vision model is converted to OpenVINO IR for accelerated
    inference on Intel hardware.  Text encoding uses the HuggingFace
    pipeline by default (it is already fast on CPU).

Integration:
    - During ingestion, documents with ``image_path`` get CLIP embeddings
      stored in a separate FAISS index (``clip_index.faiss``).
    - During retrieval, text queries are encoded with both MiniLM (for text
      similarity) and CLIP (for image similarity).  Results are fused.
    - Metadata for CLIP-embedded chunks includes ``modality="image"``
      and ``embedding_type="clip"``.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# CLIP embedding dimension for ViT-B/32
CLIP_DIMENSION = 512
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"

# ---------------------------------------------------------------------------
# Lazy Imports: Defer loading transformers and PIL until CLIPEncoder is instantiated
# ---------------------------------------------------------------------------


class CLIPEncoder:
    """
    CLIP-based multimodal encoder for images and text.

    Produces 512-dimensional embeddings in a shared vector space,
    enabling cross-modal similarity search (text query ↔ image).

    Usage::

        encoder = CLIPEncoder()
        img_emb = encoder.encode_image("photo.jpg")     # (512,)
        txt_emb = encoder.encode_text("a chart")         # (512,)
        similarity = np.dot(img_emb, txt_emb)            # cosine sim

    With OpenVINO acceleration::

        encoder = CLIPEncoder(use_openvino=True, device="CPU")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        use_openvino: bool = False,
        device: str = "CPU",
    ):
        """
        Args:
            model_name    : HuggingFace model identifier for CLIP.
            use_openvino  : If True, try to compile the vision model
                            with OpenVINO for accelerated inference.
            device        : OpenVINO device string ("CPU", "GPU", "NPU").
        """
        self.model_name = model_name
        self.use_openvino = use_openvino
        self.device = device
        self._dim = CLIP_DIMENSION
        self._model = None
        self._processor = None
        self._ov_vision_model = None

        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            logger.warning(
                "transformers CLIP classes not available. "
                "Install: pip install transformers"
            )
            return

        try:
            from PIL import Image
        except ImportError:
            logger.warning(
                "Pillow is required for image processing. "
                "Install: pip install Pillow"
            )
            return

        try:
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._model = CLIPModel.from_pretrained(model_name)
            self._model.eval()
            logger.info("CLIP model loaded: %s", model_name)
        except Exception as exc:
            logger.error("Failed to load CLIP model '%s': %s", model_name, exc)
            self._model = None
            self._processor = None
            return

        # Attempt OpenVINO acceleration for the vision encoder
        if use_openvino:
            self._try_openvino_vision(device)

    def _try_openvino_vision(self, device: str) -> None:
        """
        Try to convert and compile the CLIP vision model with OpenVINO.

        This uses ``torch.onnx.export`` → ``openvino.convert_model`` for
        on-the-fly conversion.  The converted model is cached in memory
        for the lifetime of this encoder instance.
        """
        try:
            import torch
            import openvino as ov

            # Validate device via DeviceManager if available
            try:
                from src.openvino.device_manager import DeviceManager
                dm = DeviceManager()
                device = dm.select(device)
            except Exception:
                pass

            self.device = device

            # Create a dummy input for tracing
            dummy_image = Image.new("RGB", (224, 224))
            inputs = self._processor(images=dummy_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"]

            # Export vision model to ONNX in memory via torch tracing
            import io
            buffer = io.BytesIO()

            vision_model = self._model.vision_model

            class VisionWrapper(torch.nn.Module):
                def __init__(self, vision_model, visual_projection):
                    super().__init__()
                    self.vision_model = vision_model
                    self.visual_projection = visual_projection

                def forward(self, pixel_values):
                    vision_outputs = self.vision_model(pixel_values=pixel_values)
                    pooled = vision_outputs.pooler_output
                    return self.visual_projection(pooled)

            wrapper = VisionWrapper(self._model.vision_model,
                                    self._model.visual_projection)
            wrapper.eval()

            with torch.no_grad():
                # Directly convert the PyTorch model to OpenVINO.
                # This explicitly avoids torch.onnx and its problematic
                # new dynamo export behavior, saving ~30s of log spam.
                ov_model = ov.convert_model(
                    wrapper,
                    example_input=pixel_values
                )

            core = ov.Core()
            self._ov_vision_model = core.compile_model(ov_model, device)
            logger.info(
                "CLIP vision model compiled with OpenVINO on %s", device
            )
        except Exception as exc:
            logger.warning(
                "OpenVINO acceleration for CLIP vision failed (%s). "
                "Falling back to PyTorch.",
                exc,
            )
            self._ov_vision_model = None

    @property
    def dimension(self) -> int:
        """Return the CLIP embedding dimensionality (512)."""
        return self._dim

    @property
    def is_available(self) -> bool:
        """Check if the encoder is ready for inference."""
        return self._model is not None and self._processor is not None

    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Encode a single image into a CLIP embedding vector.

        Args:
            image_path : Path to an image file (PNG, JPG, etc.)

        Returns:
            np.ndarray of shape (512,), L2-normalised, or None on failure.
        """
        if not self.is_available:
            logger.warning("CLIP encoder not available")
            return None

        path = Path(image_path)
        if not path.exists():
            logger.error("Image not found: %s", image_path)
            return None

        try:
            image = Image.open(path).convert("RGB")

            if self._ov_vision_model is not None:
                # OpenVINO path
                inputs = self._processor(images=image, return_tensors="np")
                pixel_values = inputs["pixel_values"].astype(np.float32)
                result = self._ov_vision_model({"pixel_values": pixel_values})
                embedding = result[0].squeeze()
            else:
                # PyTorch path
                import torch
                inputs = self._processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_features = self._model.get_image_features(
                        **inputs
                    )
                embedding = image_features.squeeze().numpy()

            # L2-normalise for cosine similarity via FAISS inner product
            norm = np.linalg.norm(embedding)
            if norm > 1e-9:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as exc:
            logger.error("CLIP image encoding failed for %s: %s",
                         image_path, exc)
            return None

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """
        Encode multiple images into CLIP embedding vectors.

        Args:
            image_paths : List of paths to image files.

        Returns:
            np.ndarray of shape (N, 512), L2-normalised.
            Images that fail to encode get zero vectors.
        """
        embeddings = np.zeros((len(image_paths), self._dim), dtype=np.float32)
        for i, path in enumerate(image_paths):
            emb = self.encode_image(path)
            if emb is not None:
                embeddings[i] = emb
        return embeddings

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """
        Encode a text query into a CLIP embedding vector.

        This uses the CLIP text encoder so that the resulting vector
        is in the same space as image embeddings, enabling cross-modal
        similarity search.

        Args:
            text : A text query string.

        Returns:
            np.ndarray of shape (512,), L2-normalised, or None on failure.
        """
        if not self.is_available:
            logger.warning("CLIP encoder not available")
            return None

        try:
            import torch
            inputs = self._processor(text=[text], return_tensors="pt",
                                     padding=True, truncation=True)
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)

            embedding = text_features.squeeze().numpy()

            # L2-normalise
            norm = np.linalg.norm(embedding)
            if norm > 1e-9:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as exc:
            logger.error("CLIP text encoding failed: %s", exc)
            return None

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple text queries into CLIP embedding vectors.

        Args:
            texts : List of text query strings.

        Returns:
            np.ndarray of shape (N, 512), L2-normalised.
        """
        embeddings = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            emb = self.encode_text(text)
            if emb is not None:
                embeddings[i] = emb
        return embeddings
