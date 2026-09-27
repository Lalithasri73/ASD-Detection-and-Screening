import pickle
from pathlib import Path

import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


AUTISTIC_DIR = Path("audio_data") / "autistic"
NORMAL_DIR = Path("audio_data") / "normal"
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "audio_classifier.pkl"
ENCODER_PATH = MODEL_DIR / "audio_label_encoder.pkl"


def extract_mfcc_mean(file_path: Path, n_mfcc: int = 40) -> np.ndarray:
    """Load audio and return mean MFCC feature vector."""
    y, sr = librosa.load(str(file_path), sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def load_audio_features() -> tuple[np.ndarray, np.ndarray]:
    """Load all .wav files from autistic and normal folders."""
    file_label_pairs = []

    file_label_pairs.extend((p, "autistic") for p in AUTISTIC_DIR.glob("*.wav"))
    file_label_pairs.extend((p, "normal") for p in NORMAL_DIR.glob("*.wav"))

    if not file_label_pairs:
        raise FileNotFoundError(
            "No .wav files found. Expected files in 'audio_data/autistic' and/or 'audio_data/normal'."
        )

    features = []
    labels = []

    for file_path, label in file_label_pairs:
        try:
            feat = extract_mfcc_mean(file_path, n_mfcc=40)
            features.append(feat)
            labels.append(label)
        except Exception as exc:
            print(f"Skipping {file_path} due to error: {exc}")

    if not features:
        raise RuntimeError("No valid audio files could be processed.")

    return np.array(features), np.array(labels)


def main() -> None:
    X, y_text = load_audio_features()

    # Explicit mapping as requested: Autistic=1, Normal=0
    label_mapping = {"normal": 0, "autistic": 1}
    y = np.array([label_mapping[label] for label in y_text], dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Label mapping:", {"normal": 0, "autistic": 1})

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_mapping, f)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved label encoder to: {ENCODER_PATH}")


if __name__ == "__main__":
    main()
