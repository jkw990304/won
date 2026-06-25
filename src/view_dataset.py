from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from gesture_config import DATA_PATH, FEATURE_SIZE, SEQUENCE_LENGTH


CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def to_points(frame_features, hand_index, width, height):
    offset = hand_index * 63
    coords = frame_features[offset : offset + 63].reshape(21, 3)
    if np.allclose(coords, 0):
        return None

    xy = coords[:, :2]
    scale = min(width, height) * 1.8
    center = np.array([width * (0.35 + 0.3 * hand_index), height * 0.55])
    points = center + xy * scale
    return points.astype(int)


def draw_hand(canvas, points):
    for start, end in CONNECTIONS:
        cv2.line(canvas, tuple(points[start]), tuple(points[end]), (0, 220, 255), 3)
    for point in points:
        cv2.circle(canvas, tuple(point), 5, (0, 255, 120), -1)


def render_sample(sample, label, index, frame_number):
    canvas = np.zeros((720, 960, 3), dtype=np.uint8)
    frame_features = sample[frame_number]

    for hand_index in range(2):
        points = to_points(frame_features, hand_index, canvas.shape[1], canvas.shape[0])
        if points is not None:
            draw_hand(canvas, points)

    cv2.putText(
        canvas,
        f"sample {index} | label: {label} | frame {frame_number + 1}/{SEQUENCE_LENGTH}",
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        "n next  p previous  q quit",
        (24, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 220, 255),
        2,
    )
    return canvas


def print_summary(labels):
    counts = Counter(labels)
    print("dataset summary")
    print(f"total: {len(labels)}")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")


def main():
    path = Path(DATA_PATH)
    if not path.exists():
        raise RuntimeError("data/samples.npz가 없습니다. 먼저 collect_data.py로 데이터를 모아주세요.")

    data = np.load(path, allow_pickle=True)
    samples = data["x"]
    labels = data["y"]

    if samples.shape[1:] != (SEQUENCE_LENGTH, FEATURE_SIZE):
        raise RuntimeError(f"예상과 다른 데이터 형태입니다: {samples.shape}")

    print_summary(labels)
    if len(samples) == 0:
        return

    index = 0
    frame_number = 0
    while True:
        canvas = render_sample(samples[index], labels[index], index, frame_number)
        cv2.imshow("Dataset Viewer", canvas)

        key = cv2.waitKey(80) & 0xFF
        frame_number = (frame_number + 1) % SEQUENCE_LENGTH

        if key == ord("q"):
            break
        if key == ord("n"):
            index = (index + 1) % len(samples)
            frame_number = 0
        if key == ord("p"):
            index = (index - 1) % len(samples)
            frame_number = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
