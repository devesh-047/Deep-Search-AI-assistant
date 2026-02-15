"""
Video processing subpackage -- extracting, transcribing, and interpreting video content.

Supports two ingestion modes:
    - **Caption mode** (MSR-VTT): VideoLoader → FrameSampler → FrameOCR → VideoDocumentBuilder
    - **Transcript mode**: VideoLoader → AudioExtractor → WhisperTranscriber → FrameSampler → FrameOCR → VideoDocumentBuilder
"""

from src.video.video_loader import VideoLoader, VideoFile
from src.video.frame_sampler import FrameSampler
from src.video.frame_ocr import FrameOCR
from src.video.video_document_builder import VideoDocumentBuilder

# Whisper / audio components are optional (not needed for caption-based datasets).
try:
    from src.video.audio_extractor import AudioExtractor
    from src.video.transcription import WhisperTranscriber
except ImportError:
    pass
