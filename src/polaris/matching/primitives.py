"""Reusable model and geometry primitives for cross-temporal matching."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch
from PIL import Image


class FastVLMAnalyzer:
    """FastVLM wrapper for semantic/image embedding extraction."""

    def __init__(self, model_path: str, device: str = "mps") -> None:
        try:
            from llava.constants import (
                DEFAULT_IM_END_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IMAGE_TOKEN,
                IMAGE_TOKEN_INDEX,
            )
            from llava.conversation import conv_templates
            from llava.mm_utils import (
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
            )
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
        except ImportError as exc:
            raise ImportError(
                "FastVLMAnalyzer requires local llava package modules to be importable. "
                "Run from repository root or install llava package dependencies."
            ) from exc

        self._default_image_token = DEFAULT_IMAGE_TOKEN
        self._default_im_end_token = DEFAULT_IM_END_TOKEN
        self._default_im_start_token = DEFAULT_IM_START_TOKEN
        self._image_token_index = IMAGE_TOKEN_INDEX
        self._conv_templates = conv_templates
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token

        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            None,
            model_name,
            device=device,
        )
        self.device = device
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _format_prompt(self, prompt: str) -> str:
        if self.model.config.mm_use_im_start_end:
            return (
                f"{self._default_im_start_token}{self._default_image_token}"
                f"{self._default_im_end_token}\n{prompt}"
            )
        return f"{self._default_image_token}\n{prompt}"

    def describe_object(self, image: Image.Image) -> str:
        """Generate a semantic object description from an image crop."""
        prompt = (
            "Describe this object in detail, including its material, color, shape, texture, "
            "and identifying features useful for re-identification across seasons."
        )
        return self._generate(image, prompt=prompt, temperature=0.2)

    def _generate(self, image: Image.Image, prompt: str, temperature: float = 0.2) -> str:
        formatted_prompt = self._format_prompt(prompt)
        conv = self._conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], formatted_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = self._tokenizer_image_token(
            prompt_text,
            self.tokenizer,
            self._image_token_index,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        image_tensor = self._process_images([image], self.image_processor, self.model.config)[0]

        with torch.inference_mode():
            kwargs = {
                "images": image_tensor.unsqueeze(0).to(self.device, dtype=torch.float16),
                "image_sizes": [image.size],
                "do_sample": temperature > 0,
                "max_new_tokens": 256,
                "use_cache": True,
            }
            if temperature > 0:
                kwargs["temperature"] = temperature
            output_ids = self.model.generate(input_ids, **kwargs)

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def get_vision_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract a fixed-size image embedding from FastVLM vision tower."""
        image_tensor = self._process_images([image], self.image_processor, self.model.config)[0]

        with torch.no_grad():
            vision_tower = self.model.get_vision_tower()
            features = vision_tower(image_tensor.unsqueeze(0).to(self.device, dtype=torch.float16))

            if isinstance(features, dict):
                features = features.get("image_features", features["feats"])

            if features.ndim == 4:
                batch, channels, height, width = features.shape
                features = features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)

            if features.ndim == 3:
                return features.mean(dim=1)[0].cpu().numpy().astype(np.float32)

            if features.ndim == 2:
                return features[0].cpu().numpy().astype(np.float32)

        raise ValueError(f"Unexpected feature shape from vision tower: {features.shape}")


class DepthValidator:
    """Depth-map helpers for spatial consistency checks."""

    @staticmethod
    def compute_depth_statistics(depth: np.ndarray) -> dict[str, float]:
        """Compute aggregate stats for a full depth map."""
        valid = depth[depth > 0]
        if valid.size == 0:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "valid_ratio": 0.0}
        return {
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "std": float(np.std(valid)),
            "valid_ratio": float(valid.size / depth.size),
        }

    @staticmethod
    def extract_depth_stats(depth: np.ndarray, bbox: Sequence[float]) -> dict[str, float]:
        """Compute depth statistics for a bounding box region."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth.shape[1] - 1, x2)
        y2 = min(depth.shape[0] - 1, y2)

        crop = depth[y1:y2, x1:x2]
        if crop.size == 0:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "valid_ratio": 0.0}

        valid = crop[crop > 0]
        if valid.size == 0:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "valid_ratio": 0.0}

        return {
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "std": float(np.std(valid)),
            "valid_ratio": float(valid.size / crop.size),
        }


class KeypointMatcher:
    """Keypoint detector/matcher wrapper for geometric verification."""

    def __init__(self, method: str = "orb") -> None:
        self.method = method.lower()
        if self.method == "sift":
            self.detector = cv2.SIFT_create()
            flann_index_kdtree = 1
            index_params = {"algorithm": flann_index_kdtree, "trees": 5}
            search_params = {"checks": 50}
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.method == "akaze":
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.detector = cv2.ORB_create(nfeatures=500)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match_regions(
        self,
        img1: Image.Image,
        bbox1: Sequence[float],
        img2: Image.Image,
        bbox2: Sequence[float],
        ratio_threshold: float = 0.75,
    ) -> tuple[int, float]:
        """Return (inlier_count, inlier_ratio) for keypoint matches between bbox crops."""
        cv_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        cv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

        crop1 = self._crop(cv_img1, bbox1)
        crop2 = self._crop(cv_img2, bbox2)
        if crop1.size == 0 or crop2.size == 0:
            return 0, 0.0

        kp1, des1 = self.detector.detectAndCompute(crop1, None)
        kp2, des2 = self.detector.detectAndCompute(crop2, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return 0, 0.0

        matches = self.matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for pair in matches:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < ratio_threshold * second.distance:
                good_matches.append(first)

        if len(good_matches) < 4:
            return 0, 0.0

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if mask is None:
            return 0, 0.0

        inliers = int(np.sum(mask))
        confidence = float(inliers / len(good_matches)) if good_matches else 0.0
        return inliers, confidence

    @staticmethod
    def _crop(image: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        return image[y1:y2, x1:x2]


__all__ = ["DepthValidator", "FastVLMAnalyzer", "KeypointMatcher"]
