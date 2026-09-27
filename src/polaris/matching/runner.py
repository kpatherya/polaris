"""Shared matching workflows for baseline and full-pipeline experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from polaris.matching.primitives import DepthValidator, FastVLMAnalyzer, KeypointMatcher


@dataclass(frozen=True)
class StrategySpec:
    """Configuration for a matching strategy."""

    name: str
    needs_visual: bool
    needs_semantic: bool
    needs_depth: bool
    needs_geometric: bool
    default_log_file: str


STRATEGIES: dict[str, StrategySpec] = {
    "baseline-a": StrategySpec(
        name="baseline-a",
        needs_visual=True,
        needs_semantic=True,
        needs_depth=False,
        needs_geometric=False,
        default_log_file="artifacts/runs/latest/baseline_a_results.csv",
    ),
    "baseline-b": StrategySpec(
        name="baseline-b",
        needs_visual=False,
        needs_semantic=False,
        needs_depth=False,
        needs_geometric=True,
        default_log_file="artifacts/runs/latest/baseline_b_results.csv",
    ),
    "full-pipeline": StrategySpec(
        name="full-pipeline",
        needs_visual=True,
        needs_semantic=True,
        needs_depth=True,
        needs_geometric=True,
        default_log_file="artifacts/runs/latest/full_pipeline_results.csv",
    ),
}


class MatchingComponents:
    """Lazy-initialized wrappers around reusable matching primitives."""

    def __init__(
        self,
        strategy: StrategySpec,
        model_path: str,
        device: str,
        keypoint_method: str,
    ) -> None:
        self.strategy = strategy
        self.embedding_cache: dict[str, np.ndarray] = {}
        self.model_loaded = False
        self.analyzer = None
        self.depth_validator = None
        self.keypoint_matcher = None

        if strategy.needs_visual:
            self.analyzer = FastVLMAnalyzer(model_path=model_path, device=device)
            self.model_loaded = True

        if strategy.needs_depth:
            self.depth_validator = DepthValidator()

        if strategy.needs_geometric:
            self.keypoint_matcher = KeypointMatcher(method=keypoint_method)

    def get_visual_embedding(self, image_path: str) -> np.ndarray:
        """Return visual embedding with caching."""
        if image_path in self.embedding_cache:
            return self.embedding_cache[image_path]

        if not self.model_loaded or self.analyzer is None:
            return np.zeros(1)

        if not os.path.exists(image_path):
            return np.zeros(1)

        image = Image.open(image_path).convert("RGB")
        embedding = self.analyzer.get_vision_embedding(image)
        self.embedding_cache[image_path] = embedding
        return embedding

    @staticmethod
    def compute_visual_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        if emb1.shape != emb2.shape or len(emb1.shape) == 0:
            return 0.0

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    @staticmethod
    def compute_semantic_similarity(keywords1: set[str], keywords2: set[str]) -> float:
        """Jaccard similarity between keyword sets."""
        if not keywords1 and not keywords2:
            return 0.0
        union = keywords1 | keywords2
        if not union:
            return 0.0
        return float(len(keywords1 & keywords2) / len(union))

    def compute_depth_consistency(
        self,
        query_path: str,
        candidate_path: str,
        depth_dir_query: Path | None,
        depth_dir_candidate: Path | None,
    ) -> float:
        """Simple depth consistency score in [0, 1]."""
        if self.depth_validator is None:
            return 0.5

        if depth_dir_query is None or depth_dir_candidate is None:
            return 0.5

        query_depth_path = _find_depth_file(query_path, depth_dir_query)
        candidate_depth_path = _find_depth_file(candidate_path, depth_dir_candidate)
        if query_depth_path is None or candidate_depth_path is None:
            return 0.5

        query_depth = cv2.imread(str(query_depth_path), cv2.IMREAD_UNCHANGED)
        candidate_depth = cv2.imread(str(candidate_depth_path), cv2.IMREAD_UNCHANGED)
        if query_depth is None or candidate_depth is None:
            return 0.5

        query_stats = self.depth_validator.compute_depth_statistics(query_depth)
        candidate_stats = self.depth_validator.compute_depth_statistics(candidate_depth)

        q_med = query_stats.get("median", 0)
        c_med = candidate_stats.get("median", 0)
        if q_med <= 0 or c_med <= 0:
            return 0.5

        return float(min(q_med, c_med) / max(q_med, c_med))

    def compute_geometric_confidence(
        self,
        query_img: Image.Image,
        query_bbox: list[float],
        candidate_img: Image.Image,
        candidate_bbox: list[float],
    ) -> tuple[int, float]:
        """Keypoint/RANSAC confidence for bbox pair."""
        if self.keypoint_matcher is None:
            return 0, 0.0

        try:
            return self.keypoint_matcher.match_regions(
                img1=query_img,
                bbox1=query_bbox,
                img2=candidate_img,
                bbox2=candidate_bbox,
            )
        except (cv2.error, OSError, RuntimeError, TypeError, ValueError):
            return 0, 0.0


def _find_depth_file(rgb_path: str, depth_dir: Path) -> Path | None:
    rgb_filename = os.path.basename(rgb_path)
    preferred = depth_dir / rgb_filename
    if preferred.exists():
        return preferred

    stem = Path(rgb_filename).stem
    for ext in (".png", ".tiff", ".tif"):
        candidate = depth_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def load_detections(json_path: str) -> dict[str, Any]:
    """Load detection data from JSON."""
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_query_and_candidates(
    data: dict[str, Any],
    autumn_idx: int | None,
    winter_idx: int | None,
    limit: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    """Select query frame and candidate list for bi-directional matching."""
    autumn_frames = data.get("autumn", [])
    winter_frames = data.get("winter", [])

    if not autumn_frames or not winter_frames:
        raise ValueError("detections JSON must contain non-empty 'autumn' and 'winter' arrays")

    if autumn_idx is None and winter_idx is None:
        raise ValueError("Specify either --autumn-idx or --winter-idx")
    if autumn_idx is not None and winter_idx is not None:
        raise ValueError("Provide only one of --autumn-idx or --winter-idx")

    if autumn_idx is not None:
        if autumn_idx >= len(autumn_frames):
            raise IndexError(f"autumn index {autumn_idx} out of range")
        query_frame = autumn_frames[autumn_idx]
        candidate_frames = winter_frames
        query_season = "autumn"
        candidate_season = "winter"
    else:
        if winter_idx is None or winter_idx >= len(winter_frames):
            raise IndexError(f"winter index {winter_idx} out of range")
        query_frame = winter_frames[winter_idx]
        candidate_frames = autumn_frames
        query_season = "winter"
        candidate_season = "autumn"

    if limit is not None:
        candidate_frames = candidate_frames[:limit]

    return query_frame, candidate_frames, query_season, candidate_season


def get_semantic_keywords(frame_data: dict[str, Any]) -> set[str]:
    """Extract unique labels from frame detections."""
    return {det["label"] for det in frame_data.get("detections", []) if "label" in det}


def compute_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)

    if x2_int <= x1_int or y2_int <= y1_int:
        return 0.0

    intersection = (x2_int - x1_int) * (y2_int - y1_int)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return float(intersection / union) if union > 0 else 0.0


def evaluate_match_quality(query_frame: dict[str, Any], match_frame: dict[str, Any]) -> dict[str, Any]:
    """Evaluate semantic and spatial consistency between two matched frames."""
    query_dets = query_frame.get("detections", [])
    match_dets = match_frame.get("detections", [])

    query_labels = {det["label"] for det in query_dets}
    match_labels = {det["label"] for det in match_dets}

    label_union = query_labels | match_labels
    label_jaccard = float(len(query_labels & match_labels) / len(label_union)) if label_union else 0.0

    best_ious: list[float] = []
    same_label_ious: list[float] = []

    for q_det in query_dets:
        q_bbox = q_det["bbox"]
        q_label = q_det["label"]

        best_iou = 0.0
        best_same_label = 0.0
        for m_det in match_dets:
            iou = compute_iou(q_bbox, m_det["bbox"])
            best_iou = max(best_iou, iou)
            if q_label == m_det.get("label"):
                best_same_label = max(best_same_label, iou)

        best_ious.append(best_iou)
        same_label_ious.append(best_same_label)

    avg_iou = float(np.mean(best_ious)) if best_ious else 0.0
    avg_same_label_iou = float(np.mean(same_label_ious)) if same_label_ious else 0.0
    max_iou = float(max(best_ious)) if best_ious else 0.0
    good_matches = sum(1 for val in best_ious if val > 0.5)
    match_rate = float(good_matches / len(query_dets)) if query_dets else 0.0

    q_count = len(query_dets)
    m_count = len(match_dets)
    count_ratio = float(min(q_count, m_count) / max(q_count, m_count)) if max(q_count, m_count) > 0 else 0.0

    overall = float(
        0.3 * label_jaccard +
        0.3 * avg_iou +
        0.2 * avg_same_label_iou +
        0.1 * match_rate +
        0.1 * count_ratio
    )

    return {
        "label_jaccard": label_jaccard,
        "avg_iou": avg_iou,
        "avg_same_label_iou": avg_same_label_iou,
        "max_iou": max_iou,
        "match_rate": match_rate,
        "query_detection_count": q_count,
        "match_detection_count": m_count,
        "overall_quality_score": overall,
    }


def _run_baseline_a(
    components: MatchingComponents,
    query_frame: dict[str, Any],
    candidate_frames: list[dict[str, Any]],
    top_k: int,
    visual_weight: float,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    query_path = query_frame["image_path"]
    query_embedding = components.get_visual_embedding(query_path)
    if len(query_embedding) <= 1:
        return [], None

    query_keywords = get_semantic_keywords(query_frame)

    results: list[dict[str, Any]] = []
    for candidate in tqdm(candidate_frames, desc="Baseline A", unit="frame"):
        candidate_embedding = components.get_visual_embedding(candidate["image_path"])
        if len(candidate_embedding) <= 1:
            continue

        visual_score = components.compute_visual_similarity(query_embedding, candidate_embedding)
        semantic_score = components.compute_semantic_similarity(
            query_keywords,
            get_semantic_keywords(candidate),
        )
        final_score = (visual_weight * visual_score) + ((1.0 - visual_weight) * semantic_score)

        results.append(
            {
                "frame_id": candidate["frame_id"],
                "image_path": candidate["image_path"],
                "strategy": "baseline-a",
                "visual_score": visual_score,
                "semantic_score": semantic_score,
                "depth_score": 0.5,
                "geometric_confidence": 0.0,
                "num_inliers": 0,
                "is_invariant": False,
                "final_confidence": final_score,
                "frame_data": candidate,
            }
        )

    results.sort(key=lambda item: item["final_confidence"], reverse=True)
    return results[:top_k], query_embedding


def _run_baseline_b(
    components: MatchingComponents,
    query_frame: dict[str, Any],
    candidate_frames: list[dict[str, Any]],
    top_k: int,
    min_inliers: int,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    query_path = query_frame["image_path"]
    if not os.path.exists(query_path):
        return [], None

    query_img = Image.open(query_path).convert("RGB")
    query_dets = query_frame.get("detections", [])

    results: list[dict[str, Any]] = []
    for candidate in tqdm(candidate_frames, desc="Baseline B", unit="frame"):
        cand_path = candidate["image_path"]
        cand_dets = candidate.get("detections", [])
        if not os.path.exists(cand_path) or not query_dets or not cand_dets:
            continue

        cand_img = Image.open(cand_path).convert("RGB")
        best_inliers = 0
        best_conf = 0.0
        total_inliers = 0
        valid_matches = 0

        for q_det in query_dets:
            for c_det in cand_dets:
                inliers, conf = components.compute_geometric_confidence(
                    query_img=query_img,
                    query_bbox=q_det["bbox"],
                    candidate_img=cand_img,
                    candidate_bbox=c_det["bbox"],
                )
                if inliers >= min_inliers:
                    total_inliers += inliers
                    valid_matches += 1
                    if conf > best_conf:
                        best_conf = conf
                        best_inliers = inliers

        avg_inliers = float(total_inliers / valid_matches) if valid_matches else 0.0
        results.append(
            {
                "frame_id": candidate["frame_id"],
                "image_path": cand_path,
                "strategy": "baseline-b",
                "visual_score": 0.0,
                "semantic_score": 0.0,
                "depth_score": 0.5,
                "geometric_confidence": best_conf,
                "num_inliers": best_inliers,
                "avg_inliers": avg_inliers,
                "is_invariant": False,
                "final_confidence": best_conf,
                "frame_data": candidate,
            }
        )

    results.sort(key=lambda item: item["final_confidence"], reverse=True)
    return results[:top_k], None


def _run_full_pipeline(
    components: MatchingComponents,
    query_frame: dict[str, Any],
    candidate_frames: list[dict[str, Any]],
    top_k: int,
    min_inliers: int,
    depth_dir_query: Path | None,
    depth_dir_candidate: Path | None,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    query_path = query_frame["image_path"]
    query_embedding = components.get_visual_embedding(query_path)
    if len(query_embedding) <= 1:
        return [], None

    query_keywords = get_semantic_keywords(query_frame)
    query_dets = query_frame.get("detections", [])

    results: list[dict[str, Any]] = []
    for candidate in tqdm(candidate_frames, desc="Full pipeline", unit="frame"):
        cand_path = candidate["image_path"]
        cand_embedding = components.get_visual_embedding(cand_path)
        if len(cand_embedding) <= 1:
            continue

        visual_score = components.compute_visual_similarity(query_embedding, cand_embedding)
        semantic_score = components.compute_semantic_similarity(
            query_keywords,
            get_semantic_keywords(candidate),
        )
        depth_score = components.compute_depth_consistency(
            query_path,
            cand_path,
            depth_dir_query,
            depth_dir_candidate,
        )

        geometric_conf = 0.0
        num_inliers = 0

        cand_dets = candidate.get("detections", [])
        preliminary = 0.5 * visual_score + 0.5 * semantic_score
        if preliminary > 0.3 and query_dets and cand_dets and os.path.exists(cand_path):
            query_img = Image.open(query_path).convert("RGB")
            cand_img = Image.open(cand_path).convert("RGB")
            for q_det in query_dets[:3]:
                for c_det in cand_dets[:3]:
                    inliers, conf = components.compute_geometric_confidence(
                        query_img=query_img,
                        query_bbox=q_det["bbox"],
                        candidate_img=cand_img,
                        candidate_bbox=c_det["bbox"],
                    )
                    if inliers >= min_inliers and conf > geometric_conf:
                        geometric_conf = conf
                        num_inliers = inliers

        final_confidence = float(
            0.3 * visual_score
            + 0.2 * semantic_score
            + 0.2 * depth_score
            + 0.3 * geometric_conf
        )

        results.append(
            {
                "frame_id": candidate["frame_id"],
                "image_path": cand_path,
                "strategy": "full-pipeline",
                "visual_score": visual_score,
                "semantic_score": semantic_score,
                "depth_score": depth_score,
                "geometric_confidence": geometric_conf,
                "num_inliers": num_inliers,
                "is_invariant": depth_score > 0.7,
                "final_confidence": final_confidence,
                "frame_data": candidate,
            }
        )

    results.sort(key=lambda item: item["final_confidence"], reverse=True)
    return results[:top_k], query_embedding


def create_visualizations(
    query_frame: dict[str, Any],
    results: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Create side-by-side PNGs for top matches."""
    os.makedirs(output_dir, exist_ok=True)

    query_path = query_frame["image_path"]
    if not os.path.exists(query_path):
        return

    query_img = Image.open(query_path).convert("RGB")
    query_id = query_frame["frame_id"]

    for rank, result in enumerate(results[:5], start=1):
        match_path = result["image_path"]
        if not os.path.exists(match_path):
            continue

        match_img = Image.open(match_path).convert("RGB")
        canvas = _compose_side_by_side(
            query_img=query_img,
            match_img=match_img,
            query_frame=query_frame,
            match_frame=result["frame_data"],
            title=(
                f"{query_id} -> {result['frame_id']} | "
                f"final={result['final_confidence']:.3f} "
                f"v={result['visual_score']:.3f} "
                f"s={result['semantic_score']:.3f} "
                f"d={result['depth_score']:.3f} "
                f"g={result['geometric_confidence']:.3f}"
            ),
        )

        output_path = Path(output_dir) / f"{query_id}_rank{rank:02d}_{result['frame_id']}.png"
        canvas.save(output_path)


def _compose_side_by_side(
    query_img: Image.Image,
    match_img: Image.Image,
    query_frame: dict[str, Any],
    match_frame: dict[str, Any],
    title: str,
) -> Image.Image:
    q_width, q_height = query_img.size
    m_width, m_height = match_img.size

    canvas_width = q_width + m_width + 40
    canvas_height = max(q_height, m_height) + 100
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(query_img, (0, 60))
    canvas.paste(match_img, (q_width + 40, 60))

    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 16)
        text_font = ImageFont.truetype("Arial.ttf", 12)
    except OSError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    draw.text((10, 10), title, fill="black", font=title_font)
    draw.text((10, 38), f"Query: {query_frame['frame_id']}", fill="black", font=text_font)
    draw.text((q_width + 50, 38), f"Match: {match_frame['frame_id']}", fill="black", font=text_font)

    _draw_detections(draw, query_frame.get("detections", []), x_offset=0, y_offset=60, color="cyan")
    _draw_detections(draw, match_frame.get("detections", []), x_offset=q_width + 40, y_offset=60, color="lime")
    return canvas


def _draw_detections(draw: ImageDraw.ImageDraw, detections: list[dict[str, Any]], x_offset: int, y_offset: int, color: str) -> None:
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset], outline=color, width=2)


def annotate_quality(
    query_frame: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """Attach quality metrics to result rows."""
    for result in results:
        result["quality_metrics"] = evaluate_match_quality(query_frame, result["frame_data"])


def log_results_csv(
    query_id: str,
    results: list[dict[str, Any]],
    log_file: str,
) -> None:
    """Append match rows to CSV, creating a header if needed."""
    parent = Path(log_file).parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)

    file_exists = os.path.exists(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "query_id",
                    "rank",
                    "match_id",
                    "strategy",
                    "final_confidence",
                    "visual_score",
                    "semantic_score",
                    "depth_score",
                    "geometric_confidence",
                    "num_inliers",
                    "is_invariant",
                    "label_jaccard",
                    "avg_iou",
                    "match_rate",
                    "overall_quality_score",
                ]
            )

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for rank, result in enumerate(results, start=1):
            quality = result.get("quality_metrics", {})
            writer.writerow(
                [
                    timestamp,
                    query_id,
                    rank,
                    result["frame_id"],
                    result.get("strategy", ""),
                    f"{result.get('final_confidence', 0.0):.6f}",
                    f"{result.get('visual_score', 0.0):.6f}",
                    f"{result.get('semantic_score', 0.0):.6f}",
                    f"{result.get('depth_score', 0.0):.6f}",
                    f"{result.get('geometric_confidence', 0.0):.6f}",
                    int(result.get("num_inliers", 0)),
                    bool(result.get("is_invariant", False)),
                    f"{quality.get('label_jaccard', 0.0):.6f}",
                    f"{quality.get('avg_iou', 0.0):.6f}",
                    f"{quality.get('match_rate', 0.0):.6f}",
                    f"{quality.get('overall_quality_score', 0.0):.6f}",
                ]
            )


def print_summary(
    strategy: str,
    query_id: str,
    query_season: str,
    candidate_season: str,
    results: list[dict[str, Any]],
) -> None:
    """Print concise human-readable summary."""
    print("=" * 72)
    print(f"Strategy: {strategy} | Query: {query_id} ({query_season}) | Search: {candidate_season}")
    print("=" * 72)

    for idx, result in enumerate(results, start=1):
        quality = result.get("quality_metrics", {})
        print(
            f"#{idx} {result['frame_id']} "
            f"final={result['final_confidence']:.3f} "
            f"v={result['visual_score']:.3f} "
            f"s={result['semantic_score']:.3f} "
            f"d={result['depth_score']:.3f} "
            f"g={result['geometric_confidence']:.3f} "
            f"iou={quality.get('avg_iou', 0.0):.3f}"
        )


def run_matching_job(
    *,
    strategy_name: str,
    detections_file: str,
    model_path: str,
    device: str,
    autumn_idx: int | None,
    winter_idx: int | None,
    top_k: int,
    limit: int | None,
    keypoint_method: str,
    min_inliers: int,
    visual_weight: float,
    winter_depth: str | None,
    autumn_depth: str | None,
    log_file: str,
    visualize: bool,
    visualization_dir: str,
) -> list[dict[str, Any]]:
    """Execute one matching run and return top-k results."""
    strategy = STRATEGIES[strategy_name]
    data = load_detections(detections_file)

    query_frame, candidate_frames, query_season, candidate_season = select_query_and_candidates(
        data=data,
        autumn_idx=autumn_idx,
        winter_idx=winter_idx,
        limit=limit,
    )

    components = MatchingComponents(
        strategy=strategy,
        model_path=model_path,
        device=device,
        keypoint_method=keypoint_method,
    )

    query_id = query_frame["frame_id"]
    depth_query: Path | None = None
    depth_candidate: Path | None = None

    if strategy.needs_depth:
        if query_season == "autumn":
            depth_query = Path(autumn_depth) if autumn_depth and os.path.exists(autumn_depth) else None
            depth_candidate = Path(winter_depth) if winter_depth and os.path.exists(winter_depth) else None
        else:
            depth_query = Path(winter_depth) if winter_depth and os.path.exists(winter_depth) else None
            depth_candidate = Path(autumn_depth) if autumn_depth and os.path.exists(autumn_depth) else None

    if strategy_name == "baseline-a":
        results, _ = _run_baseline_a(
            components=components,
            query_frame=query_frame,
            candidate_frames=candidate_frames,
            top_k=top_k,
            visual_weight=visual_weight,
        )
    elif strategy_name == "baseline-b":
        results, _ = _run_baseline_b(
            components=components,
            query_frame=query_frame,
            candidate_frames=candidate_frames,
            top_k=top_k,
            min_inliers=min_inliers,
        )
    else:
        results, _ = _run_full_pipeline(
            components=components,
            query_frame=query_frame,
            candidate_frames=candidate_frames,
            top_k=top_k,
            min_inliers=min_inliers,
            depth_dir_query=depth_query,
            depth_dir_candidate=depth_candidate,
        )

    annotate_quality(query_frame, results)
    print_summary(
        strategy=strategy_name,
        query_id=query_id,
        query_season=query_season,
        candidate_season=candidate_season,
        results=results,
    )

    if visualize and results:
        create_visualizations(query_frame=query_frame, results=results, output_dir=visualization_dir)

    if results:
        log_results_csv(query_id=query_id, results=results, log_file=log_file)

    return results


def run_match_cli(argv: list[str] | None = None, default_strategy: str | None = None) -> int:
    """CLI adapter for strategy execution."""
    parser = argparse.ArgumentParser(description="Run cross-temporal matching experiments")
    parser.add_argument("--detections", default="data/detections.json", help="Path to detections JSON")

    if default_strategy is None:
        parser.add_argument(
            "--strategy",
            choices=sorted(STRATEGIES.keys()),
            default="full-pipeline",
            help="Matching strategy",
        )
    else:
        parser.set_defaults(strategy=default_strategy)

    parser.add_argument("--model-path", default="checkpoints/llava-fastvithd_0.5b_stage2", help="FastVLM model path")
    parser.add_argument("--device", default="mps", help="Inference device")

    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--autumn-idx", type=int, help="Autumn query index")
    query_group.add_argument("--winter-idx", type=int, help="Winter query index")

    parser.add_argument("--top-k", type=int, default=5, help="Top K matches to keep")
    parser.add_argument("--limit", type=int, default=None, help="Limit candidate frame count")
    parser.add_argument("--method", choices=["orb", "sift"], default="orb", help="Keypoint method")
    parser.add_argument("--min-inliers", type=int, default=4, help="RANSAC inlier threshold")
    parser.add_argument("--visual-weight", type=float, default=0.7, help="Baseline-A visual weight")
    parser.add_argument("--winter-depth", default=None, help="Winter depth directory")
    parser.add_argument("--autumn-depth", default=None, help="Autumn depth directory")
    parser.add_argument("--visualize", action="store_true", help="Write side-by-side PNG visualizations")
    parser.add_argument("--visualization-dir", default="experiments/visualizations", help="Visualization output directory")
    parser.add_argument("--log-file", default=None, help="CSV output file")

    args = parser.parse_args(argv)

    strategy_name: str = args.strategy
    strategy = STRATEGIES[strategy_name]
    log_file = args.log_file or strategy.default_log_file

    try:
        run_matching_job(
            strategy_name=strategy_name,
            detections_file=args.detections,
            model_path=args.model_path,
            device=args.device,
            autumn_idx=args.autumn_idx,
            winter_idx=args.winter_idx,
            top_k=args.top_k,
            limit=args.limit,
            keypoint_method=args.method,
            min_inliers=args.min_inliers,
            visual_weight=args.visual_weight,
            winter_depth=args.winter_depth,
            autumn_depth=args.autumn_depth,
            log_file=log_file,
            visualize=args.visualize,
            visualization_dir=args.visualization_dir,
        )
    except (FileNotFoundError, ImportError, IndexError, RuntimeError, ValueError) as exc:
        print(f"Matching failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(run_match_cli())
