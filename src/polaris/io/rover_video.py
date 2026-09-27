"""Utilities for converting ROVER frame folders to video files."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def convert_image_sequence_to_video(image_dir: str, output_path: str, fps: int = 30, max_frames: int | None = None) -> None:
    """Convert timestamp-sorted PNG files to an MP4 video."""
    input_dir = Path(image_dir)
    image_files = sorted(input_dir.glob("*.png"))

    if max_frames is not None:
        image_files = image_files[:max_frames]

    if not image_files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        raise RuntimeError(f"Could not read first frame: {image_files[0]}")

    height, width, _ = first_image.shape
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not output.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    for image_path in tqdm(image_files, desc="Converting", unit="frame"):
        frame = cv2.imread(str(image_path))
        if frame is not None:
            output.write(frame)

    output.release()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for frame-to-video conversion."""
    parser = argparse.ArgumentParser(description="Convert ROVER image sequence to MP4")
    parser.add_argument("--input-dir", required=True, help="Directory containing PNG frames")
    parser.add_argument("--output-path", required=True, help="Output .mp4 path")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit")
    args = parser.parse_args(argv)

    convert_image_sequence_to_video(
        image_dir=args.input_dir,
        output_path=args.output_path,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    print(f"Saved video to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
