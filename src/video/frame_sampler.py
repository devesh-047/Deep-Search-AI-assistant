"""
Frame Sampler
==============
Extracts frames from video files at configurable intervals using OpenCV.

Frames are saved as JPEG images and returned with their timestamps for
downstream OCR processing.

Dependencies:
    pip install opencv-python  (already in requirements.txt)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    """A single frame extracted from a video.

    Attributes
    ----------
    frame_path : str
        Absolute path to the saved frame image.
    timestamp : float
        Position in the video (seconds).
    frame_index : int
        Ordinal index of this sampled frame.
    """
    frame_path: str
    timestamp: float
    frame_index: int


class FrameSampler:
    """Extract frames from a video at fixed intervals.

    Usage::

        sampler = FrameSampler(output_dir="data/processed/videos/frames",
                               interval_seconds=5)
        frames = sampler.sample("lecture_01.mp4")
        for f in frames:
            print(f"Frame {f.frame_index} @ {f.timestamp:.1f}s -> {f.frame_path}")

    Parameters
    ----------
    output_dir : str
        Directory to save extracted frame images.
    interval_seconds : float
        Extract one frame every N seconds (default: 5).
    image_format : str
        Output image format (``"jpg"`` or ``"png"``).  JPG is recommended
        for storage efficiency; PNG for lossless OCR accuracy.
    """

    def __init__(
        self,
        output_dir: str,
        interval_seconds: float = 5.0,
        image_format: str = "jpg",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = max(0.5, interval_seconds)
        self.image_format = image_format.lower().lstrip(".")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, video_path: str, video_id: Optional[str] = None) -> List[SampledFrame]:
        """Extract frames from *video_path* at the configured interval.

        Parameters
        ----------
        video_path : str
            Path to the source video file.
        video_id : str, optional
            Identifier used in output filenames.  Defaults to video stem.

        Returns
        -------
        List[SampledFrame]
            Extracted frames with timestamps and file paths.
        """
        vp = Path(video_path)
        if not vp.exists():
            logger.error("Video file not found: %s", video_path)
            return []

        if video_id is None:
            video_id = vp.stem

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            logger.error("OpenCV could not open video: %s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        if duration <= 0:
            logger.warning("Video has zero duration: %s", video_path)
            cap.release()
            return []

        frame_interval = int(fps * self.interval_seconds)
        if frame_interval < 1:
            frame_interval = 1

        expected_samples = int(duration / self.interval_seconds) + 1

        # Create per-video subdirectory for frames.
        video_frame_dir = self.output_dir / video_id
        video_frame_dir.mkdir(parents=True, exist_ok=True)

        frames: List[SampledFrame] = []
        frame_count = 0
        sample_idx = 0

        logger.info("Sampling frames from %s (%.1fs, %.0f fps, interval=%.1fs)",
                     vp.name, duration, fps, self.interval_seconds)

        pbar = tqdm(
            total=expected_samples,
            desc=f"Extracting frames: {vp.name}",
            unit="frame",
            leave=False,
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                fname = f"{video_id}_frame_{sample_idx:05d}.{self.image_format}"
                fpath = video_frame_dir / fname

                # Skip if already extracted (caching).
                if not fpath.exists():
                    cv2.imwrite(str(fpath), frame)

                frames.append(SampledFrame(
                    frame_path=str(fpath),
                    timestamp=timestamp,
                    frame_index=sample_idx,
                ))
                sample_idx += 1
                pbar.update(1)

            frame_count += 1

        pbar.close()
        cap.release()

        logger.info("Extracted %d frames from %s", len(frames), vp.name)
        return frames

    def get_video_duration(self, video_path: str) -> float:
        """Return the duration of a video in seconds."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total / fps if fps > 0 else 0.0
