"""
OpenVINO Embedding Encoder
============================
Drop-in replacement for ``EmbeddingEncoder`` that runs the all-MiniLM-L6-v2
model via OpenVINO Runtime instead of PyTorch / sentence-transformers.

Why OpenVINO for embeddings?
    - Reduced latency on Intel CPUs via graph optimizations and threading
    - Access to iGPU and NPU for concurrent inference alongside CPU workloads
    - Smaller runtime footprint (no PyTorch dependency for inference)

Conversion workflow:
    1. Export sentence-transformers model to ONNX:
           optimum-cli export onnx \
               --model sentence-transformers/all-MiniLM-L6-v2 \
               models/onnx/all-MiniLM-L6-v2/

    2. Convert ONNX to OpenVINO IR:
           from src.openvino.model_converter import convert_onnx_to_ir
           convert_onnx_to_ir(
               onnx_path="models/onnx/all-MiniLM-L6-v2/model.onnx",
               output_dir="models/ov/all-MiniLM-L6-v2",
           )

    3. Use this encoder:
           ov_enc = OVEmbeddingEncoder(
               model_xml="models/ov/all-MiniLM-L6-v2/model.xml"
           )
           vectors = ov_enc.encode(["Hello world"])

Learning TODO (all implemented):
    1. ✅ Install openvino + openvino-dev.
    2. ✅ Complete the ONNX export and IR conversion above.
    3. ✅ Implement the tokeniser (use transformers.AutoTokenizer).
    4. ✅ Run inference with openvino.runtime.Core and compare to PyTorch output.
    5. ✅ Benchmark CPU vs iGPU latency.

Implementation notes:
    The sentence-transformers model outputs raw token-level embeddings from
    BERT. To get a single embedding per sentence, we must:
        1. Run the model to get token embeddings (batch × seq_len × 384)
        2. Apply **mean pooling**: average the token embeddings, but only
           over real tokens (not padding). We use the attention_mask to
           exclude padding tokens from the average.
        3. **L2-normalise** the result so that dot product = cosine similarity.
           This is critical because FAISS uses IndexFlatIP (inner product).
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384

# Default tokenizer name (same model the IR was exported from)
DEFAULT_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"


class OVEmbeddingEncoder:
    """
    OpenVINO-accelerated embedding encoder.

    Interface mirrors ``EmbeddingEncoder`` so it can be swapped in without
    changing retrieval or indexing code.

    Usage:
        encoder = OVEmbeddingEncoder(model_xml="models/ov/all-MiniLM-L6-v2/model.xml")
        vectors = encoder.encode(["Hello world", "Another sentence"])
        # vectors.shape == (2, 384), dtype == float32, L2-normalised
    """

    def __init__(
        self,
        model_xml: str = "",
        device: str = "CPU",
        dimension: int = EMBEDDING_DIM,
        tokenizer_name: str = DEFAULT_TOKENIZER,
    ):
        """
        Args:
            model_xml      : Path to the OpenVINO IR ``.xml`` file.
            device         : OpenVINO device string ("CPU", "GPU", "NPU").
            dimension      : Expected embedding dimension.
            tokenizer_name : HuggingFace tokenizer to use for text encoding.
        """
        self.model_xml = model_xml
        self.device = device
        self._dim = dimension
        self._compiled_model = None
        self._tokenizer = None
        self._input_names = []
        self._output_name = None

        # Load tokenizer
        self._load_tokenizer(tokenizer_name)

        # Load OpenVINO model if path was given
        if model_xml:
            self._try_load(model_xml, device)

    def _load_tokenizer(self, tokenizer_name: str) -> None:
        """
        Load the HuggingFace tokenizer.

        Why AutoTokenizer and not a custom tokenizer?
        ----------------------------------------------
        The ONNX/IR model was exported from a HuggingFace model, so it
        expects the exact same tokenization (WordPiece vocabulary, special
        tokens like [CLS] and [SEP], padding conventions). Using the same
        AutoTokenizer ensures token IDs match exactly.
        """
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info("Loaded tokenizer: %s", tokenizer_name)
        except ImportError:
            logger.warning(
                "transformers not installed — tokenizer unavailable. "
                "Install: pip install transformers"
            )
        except Exception as exc:
            logger.warning("Failed to load tokenizer '%s': %s", tokenizer_name, exc)

    def _try_load(self, model_xml: str, device: str) -> None:
        """
        Try to compile the IR model. Fails gracefully if openvino is
        not installed or the model file is missing.

        Compilation steps:
            1. Core.read_model() — parse the .xml graph definition and
               load .bin weights into memory.
            2. Core.compile_model() — optimize the graph for the target
               device (operator fusion, memory planning, etc.) and
               prepare it for inference.
        """
        try:
            from openvino.runtime import Core

            xml_path = Path(model_xml)
            if not xml_path.exists():
                logger.warning("Model file not found: %s", model_xml)
                return

            core = Core()
            model = core.read_model(model=str(xml_path))
            self._compiled_model = core.compile_model(
                model=model, device_name=device
            )

            # Discover input/output tensor names
            self._input_names = [
                inp.get_any_name() for inp in self._compiled_model.inputs
            ]
            self._output_name = self._compiled_model.output(0).get_any_name()

            logger.info(
                "Loaded OpenVINO embedding model on %s  "
                "inputs=%s  output=%s",
                device,
                self._input_names,
                self._output_name,
            )
        except ImportError:
            logger.warning("openvino not installed — using placeholder mode")
        except Exception as exc:
            logger.warning(
                "Failed to load OV model: %s — using placeholder mode", exc
            )

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into embedding vectors using OpenVINO inference.

        Pipeline (per batch):
            1. **Tokenize**: Convert text strings to token IDs and
               attention masks using the HuggingFace tokenizer.
            2. **Infer**: Run the compiled OpenVINO model. The model
               outputs a tensor of shape (batch, seq_len, hidden_dim)
               containing one embedding per token per sentence.
            3. **Mean pool**: Average the token embeddings for each
               sentence, using the attention mask to exclude padding
               tokens from the average.
            4. **L2 normalise**: Scale each embedding to unit length
               so that dot product equals cosine similarity.

        Args:
            texts         : List of strings to encode.
            batch_size    : Number of texts per inference batch.
            show_progress : If True, print progress to stderr.
            normalize     : If True, L2-normalise the output vectors.

        Returns:
            np.ndarray of shape (len(texts), dimension), dtype float32.
        """
        if self._compiled_model is None or self._tokenizer is None:
            logger.warning(
                "OVEmbeddingEncoder not fully initialized — "
                "returning ZERO vectors for %d texts. "
                "Ensure model_xml path is correct and transformers is installed.",
                len(texts),
            )
            return np.zeros((len(texts), self._dim), dtype=np.float32)

        batch_embeddings = []
        total = len(texts)

        # Progress bar setup
        if show_progress:
            try:
                from tqdm import tqdm
                batch_iterator = tqdm(
                    range(0, total, batch_size),
                    desc="Encoding",
                    unit="batch",
                    total=(total + batch_size - 1) // batch_size
                )
            except ImportError:
                batch_iterator = range(0, total, batch_size)
        else:
            batch_iterator = range(0, total, batch_size)

        for start in batch_iterator:
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            # ----------------------------------------------------------
            # Step 1: Tokenize
            # ----------------------------------------------------------
            # padding=True       → pad shorter sentences to the longest
            #                       in this batch
            # truncation=True    → truncate to model's max_length (512)
            # return_tensors="np"→ return numpy arrays (not PyTorch tensors)
            #
            # Output keys: input_ids, attention_mask, (token_type_ids)
            # Shapes: (batch_size, seq_len)
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # ----------------------------------------------------------
            # Step 2: Run OpenVINO inference
            # ----------------------------------------------------------
            # Build the input dict matching the model's expected input names.
            # For BERT-based models exported via optimum, these are typically:
            #   - "input_ids"
            #   - "attention_mask"
            #   - "token_type_ids" (optional, all zeros for single-sentence)
            infer_inputs = {
                self._input_names[0]: input_ids,
                self._input_names[1]: attention_mask,
            }

            # Add token_type_ids if the model expects it
            if "token_type_ids" in self._input_names:
                infer_inputs["token_type_ids"] = np.zeros_like(input_ids)

            # Run inference — returns a dict-like object
            outputs = self._compiled_model(infer_inputs)

            # The output is the last hidden state: (batch, seq_len, hidden_dim)
            # For optimum-exported models, it's typically the first output.
            token_embeddings = outputs[self._output_name]

            # ----------------------------------------------------------
            # Step 3: Mean pooling
            # ----------------------------------------------------------
            # We want ONE embedding per sentence, not per token.
            #
            # Naive approach: average all tokens.
            # Problem: padding tokens (zeros) would dilute the average.
            #
            # Solution: multiply each token embedding by its attention_mask
            # value (1 for real tokens, 0 for padding), sum, then divide
            # by the count of real tokens.
            #
            # attention_mask shape: (batch, seq_len)
            # Expand to:           (batch, seq_len, 1) for broadcasting
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)

            # Sum of (embedding * mask) along the sequence dimension
            sum_embeddings = np.sum(
                token_embeddings * mask_expanded, axis=1
            )  # (batch, hidden_dim)

            # Count of real tokens per sentence
            sum_mask = np.sum(mask_expanded, axis=1)  # (batch, 1)
            sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)  # avoid /0

            # Mean-pooled embeddings
            sentence_embeddings = sum_embeddings / sum_mask  # (batch, hidden_dim)

            batch_embeddings.append(sentence_embeddings)

        # Concatenate all batches into a single array.
        embeddings = np.concatenate(batch_embeddings, axis=0).astype(np.float32)

        # ----------------------------------------------------------
        # Step 4: L2 normalisation
        # ----------------------------------------------------------
        # Why normalise?
        # FAISS IndexFlatIP computes dot product. For unit vectors:
        #   dot(a, b) = cos(a, b)
        # So normalisation converts inner product to cosine similarity.
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-9, a_max=None)
            embeddings = embeddings / norms

        return embeddings

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text], normalize=normalize)[0]

    def benchmark(
        self,
        texts: List[str],
        batch_size: int = 32,
        n_runs: int = 5,
    ) -> dict:
        """
        Benchmark inference speed.

        Runs encoding multiple times and reports statistics.
        Useful for comparing CPU vs GPU performance.

        Args:
            texts      : Texts to encode (more = more realistic)
            batch_size : Batch size for encoding
            n_runs     : Number of timed runs (first run is warmup)

        Returns:
            Dict with timing stats.
        """
        # Warmup run (compilation, cache warming)
        _ = self.encode(texts, batch_size=batch_size)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = self.encode(texts, batch_size=batch_size)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times_arr = np.array(times)
        return {
            "device": self.device,
            "n_texts": len(texts),
            "batch_size": batch_size,
            "n_runs": n_runs,
            "mean_ms": float(times_arr.mean() * 1000),
            "std_ms": float(times_arr.std() * 1000),
            "min_ms": float(times_arr.min() * 1000),
            "max_ms": float(times_arr.max() * 1000),
            "texts_per_sec": float(len(texts) / times_arr.mean()),
        }


def list_available_devices() -> List[str]:
    """
    Return the OpenVINO devices available on this machine.
    Useful for the ``cli.py devices`` command.
    """
    try:
        from openvino.runtime import Core
        return Core().available_devices
    except ImportError:
        logger.warning("openvino not installed -- cannot list devices")
        return []
    except Exception as exc:
        logger.error("Failed to query OpenVINO devices: %s", exc)
        return []
