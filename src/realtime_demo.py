from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from gesture_config import FEATURE_SIZE, MODEL_PATH, SEQUENCE_LENGTH
from hand_tracker import create_landmarker, detect_hands, draw_landmarks, extract_features


def main():
    if not Path(MODEL_PATH).exists():
        raise RuntimeError("models/gesture_model.joblib가 없습니다. 먼저 train_model.py를 실행하세요.")

    model = joblib.load(MODEL_PATH)
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    label = "warming_up"
    confidence = 0.0

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

            if len(sequence) == SEQUENCE_LENGTH:
                x = np.asarray(sequence, dtype=np.float32).reshape(1, -1)
                probabilities = model.predict_proba(x)[0]
                best_index = int(np.argmax(probabilities))
                label = model.classes_[best_index]
                confidence = float(probabilities[best_index])

            color = (0, 255, 0) if confidence >= 0.65 else (0, 200, 255)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
            )
            cv2.putText(
                frame,
                "q quit",
                (16, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Realtime hand gesture demo", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
