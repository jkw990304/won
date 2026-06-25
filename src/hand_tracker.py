from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import hand_landmarker

from gesture_config import FEATURE_SIZE, HAND_LANDMARKER_PATH


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def create_landmarker():
    model_path = Path(HAND_LANDMARKER_PATH)
    if not model_path.exists():
        raise RuntimeError(
            f"{HAND_LANDMARKER_PATH} 파일이 없습니다. README의 모델 다운로드 명령을 먼저 실행하세요."
        )

    base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
    options = hand_landmarker.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return hand_landmarker.HandLandmarker.create_from_options(options)


def detect_hands(landmarker, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect(image)


def extract_features(result):
    features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    if not result.hand_landmarks:
        return features

    values = []
    for hand in result.hand_landmarks[:2]:
        wrist = hand[0]
        for point in hand:
            values.extend([point.x - wrist.x, point.y - wrist.y, point.z - wrist.z])

    values = values[:FEATURE_SIZE]
    features[: len(values)] = values
    return features


def draw_landmarks(frame, result):
    if not result.hand_landmarks:
        return

    height, width = frame.shape[:2]
    for hand in result.hand_landmarks:
        points = [(int(point.x * width), int(point.y * height)) for point in hand]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 210, 255), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 255, 0), -1)
