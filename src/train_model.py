from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gesture_config import DATA_PATH, MODEL_PATH


def main():
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        raise RuntimeError("data/samples.npz가 없습니다. 먼저 collect_data.py로 데이터를 모아주세요.")

    data = np.load(data_path, allow_pickle=True)
    x = data["x"]
    y = data["y"]

    if len(x) < 20:
        raise RuntimeError("샘플이 너무 적습니다. 최소 20개 이상 모은 뒤 학습하세요.")

    x = x.reshape(len(x), -1)
    counts = Counter(y)
    print("class counts:", dict(counts))

    stratify = y if min(counts.values()) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions))

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
