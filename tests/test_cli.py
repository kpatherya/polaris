from __future__ import annotations

from polaris.config import get_threshold
from polaris.detection import analysis as detection_analysis
from polaris.io.rover_video import convert_image_sequence_to_video
from polaris.matching.runner import compute_iou, evaluate_match_quality, select_query_and_candidates


def test_imports_and_basic_config() -> None:
    assert get_threshold("house") > 0
    assert callable(detection_analysis.analyze_detections)
    assert callable(convert_image_sequence_to_video)


def test_compute_iou() -> None:
    iou = compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert 0 < iou < 1


def test_select_query_and_quality() -> None:
    data = {
        "autumn": [
            {
                "frame_id": "autumn_0000",
                "image_path": "a.png",
                "detections": [{"label": "house", "bbox": [0, 0, 10, 10]}],
            }
        ],
        "winter": [
            {
                "frame_id": "winter_0000",
                "image_path": "w.png",
                "detections": [{"label": "house", "bbox": [1, 1, 11, 11]}],
            }
        ],
    }

    query, candidates, query_season, candidate_season = select_query_and_candidates(
        data=data,
        autumn_idx=0,
        winter_idx=None,
        limit=None,
    )

    assert query_season == "autumn"
    assert candidate_season == "winter"
    assert len(candidates) == 1

    quality = evaluate_match_quality(query, candidates[0])
    assert quality["label_jaccard"] == 1.0
    assert quality["avg_iou"] > 0
