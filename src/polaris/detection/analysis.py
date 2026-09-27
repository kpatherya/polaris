"""Detection score analysis utilities."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from polaris.config import EXPANDED_GARDEN_QUERIES, RECOMMENDED_GARDEN_QUERIES, get_threshold


def analyze_detections(
    image_dir: Path,
    queries: list[str],
    num_samples: int = 20,
    device: str = "mps",
) -> dict[str, dict[str, float]]:
    """Analyze OWL-ViT score distributions for a list of object queries."""
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model.to(torch.device(device))

    image_paths = sorted([p for p in image_dir.glob("*.png") if not p.name.startswith("._")])
    if len(image_paths) > num_samples:
        step = max(1, len(image_paths) // num_samples)
        image_paths = image_paths[::step][:num_samples]

    category_scores: dict[str, list[float]] = defaultdict(list)

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=queries, images=image, return_tensors="pt")
        inputs = {key: value.to(torch.device(device)) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.0,
        )[0]

        for score, label in zip(results["scores"], results["labels"]):
            label_text = queries[label.item()]
            category_scores[label_text].append(score.item())

    stats: dict[str, dict[str, float]] = {}
    for category, scores in category_scores.items():
        if not scores:
            continue
        stats[category] = {
            "count": float(len(scores)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "p95": float(np.percentile(scores, 95)),
            "p90": float(np.percentile(scores, 90)),
            "p75": float(np.percentile(scores, 75)),
            "recommended_threshold": float(np.percentile(scores, 75)),
            "configured_threshold": float(get_threshold(category)),
        }

    return stats


def print_analysis(stats: dict[str, dict[str, float]], output_dir: Path) -> None:
    """Print and persist summary artifacts for detection analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_categories = sorted(stats.items(), key=lambda item: item[1]["max"], reverse=True)
    print("Category          Max    P95    P90    Median Config  P75")
    print("------------------------------------------------------------")

    threshold_updates: dict[str, float] = {}
    for category, stat in sorted_categories:
        print(
            f"{category:<16} {stat['max']:.3f}  {stat['p95']:.3f}  {stat['p90']:.3f}  "
            f"{stat['median']:.3f}  {stat['configured_threshold']:.3f}  {stat['p75']:.3f}"
        )
        threshold_updates[category] = stat["p75"]

    with (output_dir / "detection_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    with (output_dir / "recommended_thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump(threshold_updates, handle, indent=2)


def _parse_queries(mode: str, custom_queries: str | None) -> list[str]:
    if custom_queries:
        return [q.strip() for q in custom_queries.split(",") if q.strip()]
    if mode == "recommended":
        return RECOMMENDED_GARDEN_QUERIES
    if mode == "expanded":
        return EXPANDED_GARDEN_QUERIES
    return sorted(set(RECOMMENDED_GARDEN_QUERIES + EXPANDED_GARDEN_QUERIES))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for detection analysis."""
    parser = argparse.ArgumentParser(description="Analyze OWL-ViT detection score distributions")
    parser.add_argument("--image-dir", required=True, help="Directory containing RGB frames")
    parser.add_argument("--num-samples", type=int, default=20, help="Sample size")
    parser.add_argument("--output-dir", default="detection_analysis", help="Output folder")
    parser.add_argument("--device", default="mps", help="Inference device: mps/cuda/cpu")
    parser.add_argument(
        "--query-set",
        choices=["recommended", "expanded", "all"],
        default="recommended",
        help="Built-in query preset",
    )
    parser.add_argument(
        "--queries",
        default=None,
        help="Comma-separated custom query list; overrides --query-set",
    )
    args = parser.parse_args(argv)

    queries = _parse_queries(args.query_set, args.queries)
    stats = analyze_detections(
        image_dir=Path(args.image_dir),
        queries=queries,
        num_samples=args.num_samples,
        device=args.device,
    )
    print_analysis(stats, Path(args.output_dir))
    print(f"Saved analysis outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
