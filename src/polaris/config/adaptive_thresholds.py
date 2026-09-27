"""Adaptive threshold and query presets for OWL-ViT detection."""

CATEGORY_THRESHOLDS = {
    "house": 0.089,
    "building": 0.038,
    "shed": 0.040,
    "door": 0.049,
    "window": 0.020,
    "roof": 0.041,
    "tree": 0.065,
    "bush": 0.019,
    "grass": 0.012,
    "flowers": 0.017,
    "plant": 0.011,
}

DEFAULT_THRESHOLD = 0.12


def get_threshold(label: str) -> float:
    """Return the configured threshold for an object label."""
    return CATEGORY_THRESHOLDS.get(label.lower(), DEFAULT_THRESHOLD)


HIGH_CONFIDENCE_QUERIES = [
    "house",
    "tree",
    "window",
    "roof",
    "door",
    "shed",
]

MEDIUM_CONFIDENCE_QUERIES = [
    "bush",
    "grass",
    "fountain",
    "flowers",
    "building",
    "plant",
]

LOW_CONFIDENCE_QUERIES = [
    "lamp",
    "shrub",
    "leaves",
    "wall",
    "rock",
    "fence",
    "hedge",
    "branch",
]

UNRELIABLE_QUERIES = [
    "puddle",
    "snow",
    "mud",
    "stone",
    "path",
    "walkway",
    "post",
    "bench",
    "statue",
    "planter",
    "gate",
    "ice",
    "foliage",
    "pot",
]

STRUCTURAL_QUERIES = ["house", "building", "shed", "door", "window", "roof"]

VEGETATION_QUERIES = ["tree", "bush", "shrub", "grass", "flowers", "plant", "leaves"]

RECOMMENDED_GARDEN_QUERIES = [
    "house",
    "tree",
    "window",
    "door",
    "bush",
    "grass",
]

EXPANDED_GARDEN_QUERIES = HIGH_CONFIDENCE_QUERIES + ["bush", "grass", "flowers", "plant"]

ULTRA_FAST_QUERIES = ["house", "tree", "window"]
