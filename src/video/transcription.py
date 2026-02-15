"""
Whisper Transcription
======================
Transcribes audio files using OpenAI's Whisper model (``small`` by default).

Produces timestamped transcript segments that can be aligned with OCR output
during the document-building phase.

Dependencies:
    pip install openai-whisper

Note: Whisper downloads model weights on first use (~461 MB for ``small``).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A single segment from Whisper's transcription output.

    Attributes
    ----------
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    text : str
        Transcribed text for this segment.
    """
    start: float
    end: float
    text: str


class WhisperTranscriber:
    """Transcribe audio files using Whisper.

    Usage::

        transcriber = WhisperTranscriber(model_size="small")
        segments = transcriber.transcribe("lecture_01.wav")
        for seg in segments:
            print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")

    Parameters
    ----------
    model_size : str
        Whisper model variant: ``"tiny"``, ``"base"``, ``"small"``,
        ``"medium"``, ``"large"``.  Default is ``"small"`` for a good
        balance of speed and accuracy.
    device : str
        PyTorch device string (``"cpu"`` or ``"cuda"``).
    language : str or None
        Force a specific language (e.g. ``"en"``).  ``None`` = auto-detect.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        language: Optional[str] = "en",
    ):
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        """Transcribe an audio file and return timestamped segments.

        Parameters
        ----------
        audio_path : str
            Path to the WAV/MP3/FLAC audio file.

        Returns
        -------
        List[TranscriptSegment]
            Chronologically ordered transcript segments with timestamps.
        """
        p = Path(audio_path)
        if not p.exists():
            logger.error("Audio file not found: %s", audio_path)
            return []

        model = self._get_model()
        if model is None:
            return []

        logger.info("Transcribing: %s (model=%s, device=%s)",
                     p.name, self.model_size, self.device)

        try:
            result = model.transcribe(
                str(p),
                language=self.language,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Transcription failed for %s: %s", audio_path, exc)
            return []

        segments: List[TranscriptSegment] = []
        for seg in result.get("segments", []):
            segments.append(TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            ))

        logger.info("Transcribed %d segments from %s (%.1f s total)",
                     len(segments), p.name,
                     segments[-1].end if segments else 0.0)
        return segments

    def chunk_segments(
        self,
        segments: List[TranscriptSegment],
        interval: float = 30.0,
    ) -> List[TranscriptSegment]:
        """Merge Whisper segments into fixed-duration chunks.

        Parameters
        ----------
        segments : list
            Raw Whisper transcript segments.
        interval : float
            Target chunk duration in seconds (default: 30s).

        Returns
        -------
        List[TranscriptSegment]
            Merged segments where each covers approximately *interval* seconds.
        """
        if not segments:
            return []

        chunks: List[TranscriptSegment] = []
        current_texts: list = []
        chunk_start = segments[0].start

        for seg in segments:
            current_texts.append(seg.text)

            if (seg.end - chunk_start) >= interval:
                chunks.append(TranscriptSegment(
                    start=chunk_start,
                    end=seg.end,
                    text=" ".join(current_texts),
                ))
                current_texts = []
                chunk_start = seg.end

        # Remaining text
        if current_texts:
            chunks.append(TranscriptSegment(
                start=chunk_start,
                end=segments[-1].end,
                text=" ".join(current_texts),
            ))

        logger.debug("Chunked %d segments into %d chunks (interval=%.0fs)",
                      len(segments), len(chunks), interval)
        return chunks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the Whisper model."""
        if self._model is not None:
            return self._model

        try:
            import whisper
        except ImportError:
            logger.error(
                "openai-whisper is not installed.  "
                "Install with: pip install openai-whisper"
            )
            return None

        logger.info("Loading Whisper model: %s (this may take a moment on first run)",
                     self.model_size)
        try:
            self._model = whisper.load_model(self.model_size, device=self.device)
        except Exception as exc:
            logger.error("Failed to load Whisper model '%s': %s",
                         self.model_size, exc)
            return None

        return self._model
