"""
Audio Extractor
================
Extracts audio tracks from video files using ``moviepy`` (backed by ffmpeg).

The extracted audio is saved as a WAV file suitable for downstream
transcription with Whisper.

Dependencies:
    pip install moviepy

Note: ``ffmpeg`` must be installed on the system
    (``sudo apt install ffmpeg`` on Ubuntu/WSL).
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extract audio from a video file.

    Usage::

        extractor = AudioExtractor(output_dir="data/processed/videos/audio")
        wav_path = extractor.extract("lecture_01.mp4")

    Parameters
    ----------
    output_dir : str
        Directory to store extracted WAV files.
    sample_rate : int
        Audio sample rate (default: 16000 -- Whisper's expected rate).
    """

    def __init__(self, output_dir: str, sample_rate: int = 16_000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate

    def extract(self, video_path: str, output_filename: Optional[str] = None) -> Optional[str]:
        """Extract audio from *video_path* and write a WAV file.

        Parameters
        ----------
        video_path : str
            Path to the source video file.
        output_filename : str, optional
            Explicit output filename.  Defaults to ``<stem>.wav``.

        Returns
        -------
        str or None
            Absolute path to the extracted WAV file, or ``None`` on failure.
        """
        vp = Path(video_path)
        if not vp.exists():
            logger.error("Video file not found: %s", video_path)
            return None

        if output_filename is None:
            output_filename = vp.stem + ".wav"
        wav_path = self.output_dir / output_filename

        # Skip re-extraction if the WAV already exists.
        if wav_path.exists():
            logger.debug("Audio already extracted: %s", wav_path)
            return str(wav_path)

        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            logger.error(
                "moviepy is not installed.  Install with: pip install moviepy"
            )
            return None

        try:
            logger.debug("Extracting audio: %s -> %s", video_path, wav_path)
            clip = VideoFileClip(str(vp))
            if clip.audio is None:
                logger.warning("Video has no audio track: %s", video_path)
                clip.close()
                return None

            clip.audio.write_audiofile(
                str(wav_path),
                fps=self.sample_rate,
                nbytes=2,          # 16-bit PCM
                codec="pcm_s16le",
                logger=None,       # suppress moviepy's own progress bar
            )
            clip.close()
            logger.info("Audio extracted: %s (%.1f KB)", wav_path.name,
                        wav_path.stat().st_size / 1024)
            return str(wav_path)

        except Exception as exc:
            logger.error("Audio extraction failed for %s: %s", video_path, exc)
            return None
