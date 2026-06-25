from pathlib import Path
from collections import deque

import cv2
import numpy as np

from gesture_config import DATA_PATH, FEATURE_SIZE, LABELS, SEQUENCE_LENGTH
from hand_tracker import create_landmarker, detect_hands, draw_landmarks, extract_features


def load_existing_samples():
    path = Path(DATA_PATH)
    if not path.exists():
        return [], []

    data = np.load(path, allow_pickle=True)
    return list(data["x"]), list(data["y"])


def save_samples(samples, labels):
    Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_PATH, x=np.asarray(samples), y=np.asarray(labels))


def main():
    samples, labels = load_existing_samples()
    current_key = "1"
    sequence = deque(maxlen=SEQUENCE_LENGTH)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    with create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라 프레임을 읽지 못했습니다.")
                break

            frame = cv2.flip(frame, 1)
            result = detect_hands(landmarker, frame)
            sequence.append(extract_features(result))
            draw_landmarks(frame, result)

            label = LABELS[current_key]
            cv2.putText(
                frame,
                f"label: {label} | samples: {len(samples)}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "1 fire_alarm  2 doorbell  3 clap  0 none  SPACE save  q quit",
                (16, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Collect hand gesture samples", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if chr(key) in LABELS:
                current_key = chr(key)
            if key == 32 and len(sequence) == SEQUENCE_LENGTH:
                samples.append(np.asarray(sequence, dtype=np.float32))
                labels.append(label)
                save_samples(samples, labels)
                print(f"saved {label}: total={len(samples)}")

    cap.release()
    cv2.destroyAllWindows()
    save_samples(samples, labels)


if __name__ == "__main__":
    main()
