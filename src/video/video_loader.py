"""
Video Loader
=============
Discovers and loads video files from a dataset directory.

Supports:
  - **MSR-VTT** dataset structure (``annotation/MSR_VTT.json`` + ``videos/all/``)
  - Flat directories containing ``.mp4/.avi/.mkv`` files
  - Nested directories with metadata/transcript sidecars

Returns a list of ``VideoFile`` descriptor objects that downstream
pipeline stages consume.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Supported video extensions (lowercase).
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}

# Supported transcript sidecar extensions.
_TRANSCRIPT_EXTENSIONS = {".vtt", ".srt", ".txt", ".json"}


@dataclass
class VideoFile:
    """Descriptor for a single discovered video file.

    Attributes
    ----------
    video_path : str
        Absolute path to the video file.
    video_id : str
        Unique identifier derived from the filename (e.g. ``"video0"``).
    captions : List[str]
        Pre-loaded caption strings (from annotation JSON or sidecar).
    transcript_path : Optional[str]
        Path to a sidecar transcript file (if found alongside the video).
    metadata_path : Optional[str]
        Path to a sidecar metadata file (JSON/YAML if found).
    metadata : Dict
        Any pre-loaded metadata for the video.
    """
    video_path: str
    video_id: str
    captions: List[str] = field(default_factory=list)
    transcript_path: Optional[str] = None
    metadata_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class VideoLoader:
    """Discover video files under a root directory.

    The loader auto-detects the dataset layout:

    * **MSR-VTT format**: looks for ``annotation/MSR_VTT.json`` and loads
      captions from it, and scans ``videos/all/`` for ``.mp4`` files.
    * **Generic format**: recursively scans for video files and checks for
      sidecar transcript/metadata files using the same stem.

    Usage::

        loader = VideoLoader(
            root="/mnt/d/Openvino-project/data/raw/archive/data/MSRVTT/MSRVTT"
        )
        videos = loader.discover()
        print(f"Found {len(videos)} videos")

    Parameters
    ----------
    root : str
        Root directory to scan for video files.
    max_files : int
        Maximum number of video files to return (0 = all).
    """

    def __init__(self, root: str, max_files: int = 0):
        self.root = Path(root)
        self.max_files = max_files

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> List[VideoFile]:
        """Recursively scan *self.root* and return ``VideoFile`` descriptors.

        Auto-detects MSR-VTT vs generic directory layout.
        """
        if not self.root.exists():
            logger.warning("Video root directory does not exist: %s", self.root)
            return []

        # --- Try MSR-VTT format first ---------------------------------
        msrvtt_json = self._find_msrvtt_annotation()
        if msrvtt_json is not None:
            logger.info("Detected MSR-VTT dataset layout")
            return self._discover_msrvtt(msrvtt_json)

        # --- Fall back to generic scan --------------------------------
        logger.info("Using generic video discovery (no MSR-VTT annotation found)")
        return self._discover_generic()

    # ------------------------------------------------------------------
    # MSR-VTT discovery
    # ------------------------------------------------------------------

    def _find_msrvtt_annotation(self) -> Optional[Path]:
        """Look for MSR_VTT.json in common locations."""
        candidates = [
            self.root / "annotation" / "MSR_VTT.json",
            self.root / "MSR_VTT.json",
            self.root / "annotations" / "MSR_VTT.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _discover_msrvtt(self, annotation_path: Path) -> List[VideoFile]:
        """Load MSR-VTT videos using the master annotation JSON.

        The annotation file contains:
        - ``images``: list of ``{"id": "videoN"}``
        - ``annotations``: list of ``{"caption": "...", "image_id": "videoN"}``

        Each video typically has ~20 captions.
        """
        logger.info("Loading MSR-VTT annotations from: %s", annotation_path)

        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build caption lookup: video_id -> [caption_str, ...]
        captions_by_video: Dict[str, List[str]] = {}
        for ann in data.get("annotations", []):
            vid_id = ann.get("image_id", "")
            caption = ann.get("caption", "").strip()
            if vid_id and caption:
                captions_by_video.setdefault(vid_id, []).append(caption)

        logger.info(
            "Loaded %d captions for %d videos",
            sum(len(v) for v in captions_by_video.values()),
            len(captions_by_video),
        )

        # Locate the video files directory.
        video_dir = self._find_video_dir()
        if video_dir is None:
            logger.error(
                "Cannot find video directory under %s. "
                "Expected 'videos/all/' or similar.",
                self.root,
            )
            return []

        # Enumerate actual .mp4 files and pair with captions.
        videos: List[VideoFile] = []
        video_ids = sorted(data.get("images", []), key=lambda x: x.get("id", ""))

        for entry in video_ids:
            vid_id = entry.get("id", "")
            if not vid_id:
                continue

            # Find the video file.
            mp4_path = video_dir / f"{vid_id}.mp4"
            if not mp4_path.exists():
                logger.debug("Video file not found, skipping: %s", mp4_path)
                continue

            # Skip macOS resource fork files.
            if vid_id.startswith("._"):
                continue

            videos.append(VideoFile(
                video_path=str(mp4_path),
                video_id=vid_id,
                captions=captions_by_video.get(vid_id, []),
                metadata={
                    "dataset": "msrvtt",
                    "num_captions": len(captions_by_video.get(vid_id, [])),
                },
            ))

            if self.max_files and len(videos) >= self.max_files:
                logger.info(
                    "Reached max_files limit (%d); stopping discovery.",
                    self.max_files,
                )
                break

        logger.info(
            "Discovered %d MSR-VTT video file(s) with captions", len(videos)
        )
        return videos

    def _find_video_dir(self) -> Optional[Path]:
        """Locate the directory that contains .mp4 files."""
        candidates = [
            self.root / "videos" / "all",
            self.root / "videos",
            self.root / "video",
            self.root,
        ]
        for c in candidates:
            if c.is_dir():
                # Quick check: does it contain at least one .mp4?
                for item in c.iterdir():
                    if item.suffix.lower() == ".mp4" and not item.name.startswith("._"):
                        return c
        return None

    # ------------------------------------------------------------------
    # Generic discovery (flat / nested directories)
    # ------------------------------------------------------------------

    def _discover_generic(self) -> List[VideoFile]:
        """Recursively scan for video files (generic flat/nested layout)."""
        videos: List[VideoFile] = []

        for dirpath, _dirnames, filenames in os.walk(self.root):
            dirpath = Path(dirpath)
            for fname in sorted(filenames):
                fpath = dirpath / fname

                # Skip macOS resource forks.
                if fname.startswith("._"):
                    continue

                if fpath.suffix.lower() not in _VIDEO_EXTENSIONS:
                    continue

                video_id = fpath.stem
                transcript_path = self._find_sidecar(fpath, _TRANSCRIPT_EXTENSIONS)
                metadata_path = self._find_sidecar(fpath, {".json", ".yaml", ".yml"})

                metadata: Dict = {}
                if metadata_path:
                    metadata = self._try_load_metadata(metadata_path)

                videos.append(VideoFile(
                    video_path=str(fpath),
                    video_id=video_id,
                    transcript_path=transcript_path,
                    metadata_path=metadata_path,
                    metadata=metadata,
                ))

                if self.max_files and len(videos) >= self.max_files:
                    logger.info(
                        "Reached max_files limit (%d); stopping discovery.",
                        self.max_files,
                    )
                    return videos

        logger.info("Discovered %d video file(s) under %s", len(videos), self.root)
        return videos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_sidecar(video_path: Path, extensions: set) -> Optional[str]:
        """Look for a file with the same stem but a different extension."""
        for ext in extensions:
            candidate = video_path.with_suffix(ext)
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def _try_load_metadata(path: str) -> Dict:
        """Best-effort load of JSON or YAML metadata."""
        p = Path(path)
        try:
            if p.suffix == ".json":
                return json.loads(p.read_text(encoding="utf-8", errors="replace"))
            elif p.suffix in (".yaml", ".yml"):
                import yaml
                return yaml.safe_load(p.read_text(encoding="utf-8", errors="replace")) or {}
        except Exception as exc:
            logger.debug("Could not parse metadata %s: %s", path, exc)
        return {}
