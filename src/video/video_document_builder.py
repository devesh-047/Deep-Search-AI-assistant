"""
Video Document Builder
=======================
Merges transcript segments, captions, and frame OCR results into a unified
video document JSON schema.  The output is fully compatible with the existing
``NormalizedDocument`` pipeline — video documents can be chunked, embedded,
and indexed alongside text/image documents.

Supports:
  - **MSR-VTT** style: multiple pre-written captions per video (no Whisper needed)
  - **Whisper** style: timestamped transcript segments + OCR results

Output directory:
    Deep-Search-AI-assistant/data/processed/videos/
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video chunk schema
# ---------------------------------------------------------------------------

@dataclass
class VideoChunk:
    """A single time-aligned chunk of video content.

    Combines caption/transcript and OCR text for one temporal window.
    """
    timestamp_start: float
    timestamp_end: float
    transcript: str = ""
    caption_text: str = ""
    ocr_text: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class VideoDocument:
    """A complete processed video document.

    This is the video equivalent of ``NormalizedDocument``.
    After building, the ``text`` property concatenates all chunk content
    for downstream chunking and embedding.
    """
    doc_id: str
    source: str
    modality: str = "video"
    video_path: str = ""
    duration: float = 0.0
    chunks: List[VideoChunk] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Concatenated caption/transcript + OCR text across all chunks."""
        parts = []
        for chunk in self.chunks:
            if chunk.caption_text:
                parts.append(chunk.caption_text)
            if chunk.transcript:
                parts.append(chunk.transcript)
            if chunk.ocr_text:
                parts.append(f"[OCR: {chunk.ocr_text}]")
        return "\n".join(parts)

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "modality": self.modality,
            "video_path": self.video_path,
            "duration": self.duration,
            "chunks": [asdict(c) for c in self.chunks],
            "metadata": self.metadata,
        }

    def to_normalized_dict(self) -> Dict:
        """Return a dict compatible with ``NormalizedDocument.from_dict()``.

        This allows video documents to enter the existing chunking →
        embedding → indexing pipeline seamlessly.
        """
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "doc_type": "video",
            "text": self.text,
            "image_path": "",
            "metadata": {
                **self.metadata,
                "modality": "video",
                "video_path": self.video_path,
                "duration": self.duration,
                "num_chunks": len(self.chunks),
            },
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _make_video_doc_id(video_id: str) -> str:
    """Deterministic document ID from video identifier."""
    return "video_" + hashlib.sha256(video_id.encode()).hexdigest()[:12]


class VideoDocumentBuilder:
    """Merge transcript/captions and OCR results into ``VideoDocument`` objects.

    Supports two modes:

    1. **Caption mode** (MSR-VTT): builds a document from a list of
       caption strings. No Whisper transcription needed.
    2. **Transcript mode** (Whisper): builds time-aligned chunks from
       ``TranscriptSegment`` objects and ``FrameOCRResult`` objects.

    Usage::

        builder = VideoDocumentBuilder(
            output_dir="data/processed/videos",
            chunk_interval=30.0,
        )

        # Caption mode (MSR-VTT)
        doc = builder.build_from_captions(
            video_id="video0",
            source="msrvtt",
            captions=["a man drives a car", "a person is driving"],
            duration=15.0,
        )

        # Transcript mode (Whisper)
        doc = builder.build(
            video_id="lecture_01",
            source="custom",
            transcript_segments=segments,
            ocr_results=ocr_results,
            duration=120.0,
        )

        builder.save(doc)

    Parameters
    ----------
    output_dir : str
        Directory to save per-video JSON files.
    chunk_interval : float
        Duration of each temporal chunk in seconds (default: 30).
    """

    def __init__(self, output_dir: str, chunk_interval: float = 30.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_interval = chunk_interval

    # ------------------------------------------------------------------
    # Public API — Caption mode (MSR-VTT)
    # ------------------------------------------------------------------

    def build_from_captions(
        self,
        video_id: str,
        source: str,
        captions: List[str],
        video_path: str = "",
        duration: float = 0.0,
        ocr_results=None,
        extra_metadata: Optional[Dict] = None,
    ) -> VideoDocument:
        """Build a ``VideoDocument`` from pre-written captions.

        This is the primary path for MSR-VTT and similar caption-based
        datasets.  All captions are merged into the text of a single
        chunk that spans the full video duration.

        If *ocr_results* are provided they are appended to the chunk.

        Parameters
        ----------
        video_id : str
            Unique video identifier (e.g. ``"video0"``).
        source : str
            Dataset source name (e.g. ``"msrvtt"``).
        captions : list of str
            Caption strings describing the video.
        video_path : str
            Original video file path.
        duration : float
            Video duration in seconds.  If 0, a synthetic duration of
            ``chunk_interval`` is used.
        ocr_results : list, optional
            List of ``FrameOCRResult`` objects.
        extra_metadata : dict, optional
            Additional metadata to include.

        Returns
        -------
        VideoDocument
        """
        doc_id = _make_video_doc_id(video_id)
        ocr_results = ocr_results or []

        # If duration unknown, use chunk_interval as a reasonable default.
        if duration <= 0:
            duration = self.chunk_interval

        # Merge all captions into a single rich text block.
        caption_text = " ".join(c.strip() for c in captions if c.strip())

        # Collect OCR text.
        ocr_text = " ".join(
            ocr.text for ocr in ocr_results if hasattr(ocr, "text")
        ).strip()

        chunks = []
        if caption_text or ocr_text:
            chunks.append(VideoChunk(
                timestamp_start=0.0,
                timestamp_end=round(duration, 2),
                caption_text=caption_text,
                ocr_text=ocr_text,
                metadata={
                    "chunk_index": 0,
                    "num_captions": len(captions),
                    "ocr_frames": len(ocr_results),
                },
            ))

        doc = VideoDocument(
            doc_id=doc_id,
            source=source,
            video_path=video_path,
            duration=round(duration, 2),
            chunks=chunks,
            metadata={
                **(extra_metadata or {}),
                "video_id": video_id,
                "num_captions": len(captions),
                "total_ocr_frames": len(ocr_results),
            },
        )

        logger.info(
            "Built caption-mode document: %s (%d captions, %.1fs)",
            doc_id, len(captions), duration,
        )
        return doc

    # ------------------------------------------------------------------
    # Public API — Transcript mode (Whisper)
    # ------------------------------------------------------------------

    def build(
        self,
        video_id: str,
        source: str,
        transcript_segments,
        ocr_results,
        video_path: str = "",
        duration: float = 0.0,
        extra_metadata: Optional[Dict] = None,
    ) -> VideoDocument:
        """Build a ``VideoDocument`` by aligning transcript and OCR results.

        Parameters
        ----------
        video_id : str
            Unique video identifier.
        source : str
            Dataset source name.
        transcript_segments : list
            List of ``TranscriptSegment`` objects (from Whisper).
        ocr_results : list
            List of ``FrameOCRResult`` objects (from frame OCR).
        video_path : str
            Original video file path.
        duration : float
            Total video duration in seconds.
        extra_metadata : dict, optional
            Additional metadata to include.

        Returns
        -------
        VideoDocument
        """
        doc_id = _make_video_doc_id(video_id)

        # Compute time windows based on chunk_interval.
        if duration <= 0 and transcript_segments:
            duration = transcript_segments[-1].end

        num_windows = max(1, int(duration / self.chunk_interval) + 1)

        chunks: List[VideoChunk] = []
        for i in range(num_windows):
            t_start = i * self.chunk_interval
            t_end = min((i + 1) * self.chunk_interval, duration)

            # Collect transcript text for this window.
            transcript_parts = []
            for seg in transcript_segments:
                # Segment overlaps with this window.
                if seg.end > t_start and seg.start < t_end:
                    transcript_parts.append(seg.text)

            # Collect OCR text for this window.
            ocr_parts = []
            for ocr in ocr_results:
                if t_start <= ocr.timestamp < t_end:
                    ocr_parts.append(ocr.text)

            transcript_text = " ".join(transcript_parts).strip()
            ocr_text = " ".join(ocr_parts).strip()

            # Only add non-empty chunks.
            if transcript_text or ocr_text:
                chunks.append(VideoChunk(
                    timestamp_start=round(t_start, 2),
                    timestamp_end=round(t_end, 2),
                    transcript=transcript_text,
                    ocr_text=ocr_text,
                    metadata={
                        "chunk_index": i,
                        "transcript_segments": len(transcript_parts),
                        "ocr_frames": len(ocr_parts),
                    },
                ))

        doc = VideoDocument(
            doc_id=doc_id,
            source=source,
            video_path=video_path,
            duration=round(duration, 2),
            chunks=chunks,
            metadata={
                **(extra_metadata or {}),
                "video_id": video_id,
                "total_transcript_segments": len(transcript_segments),
                "total_ocr_frames": len(ocr_results),
                "chunk_interval_seconds": self.chunk_interval,
            },
        )

        logger.info(
            "Built video document: %s (%d chunks, %.1fs duration)",
            doc_id, len(chunks), duration,
        )
        return doc

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, doc: VideoDocument) -> str:
        """Save a video document as a JSON file.

        Returns the path to the saved file.
        """
        out_path = self.output_dir / f"{doc.doc_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Saved video document: %s", out_path)
        return str(out_path)

    def save_batch(self, docs: List[VideoDocument]) -> List[str]:
        """Save a batch of video documents.

        Also writes a consolidated ``video_documents.jsonl`` file for
        integration with the existing normalised documents pipeline.
        """
        paths = []
        jsonl_path = self.output_dir / "video_documents.jsonl"

        pbar = tqdm(
            docs,
            desc="Saving video documents",
            unit="doc",
            leave=False,
        )

        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for doc in pbar:
                # Save detailed per-video JSON.
                path = self.save(doc)
                paths.append(path)

                # Write normalised-compatible line to JSONL.
                jf.write(json.dumps(doc.to_normalized_dict(), ensure_ascii=False) + "\n")

        logger.info(
            "Saved %d video documents + JSONL to %s",
            len(docs), self.output_dir,
        )
        return paths
