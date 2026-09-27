"""
Standalone audio processing module for autism-related acoustic analysis.

Features extracted from .wav audio:
- MFCC (40 coefficients, mean + std)
- Spectral Centroid (mean + std)
- Pitch via librosa.piptrack (mean + std + variance over voiced frames)

Trains an audio-only Random Forest classifier using:
- audio_data/autistic  -> label 1
- audio_data/normal    -> label 0

Saves trained artifact to:
- model/audio_only_model.pkl

Provides:
- train_audio_model(...)
- predict_audio(file_path, ...)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


AUDIO_ROOT = Path("audio_data")
AUTISTIC_DIR = AUDIO_ROOT / "autistic"
NORMAL_DIR = AUDIO_ROOT / "normal"

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "audio_only_model.pkl"


def _safe_stats(x: np.ndarray) -> Tuple[float, float]:
    """Return mean/std with safe fallbacks for empty arrays."""
    if x.size == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x))


def extract_audio_features(file_path: str | Path, n_mfcc: int = 40) -> np.ndarray:
    """
    Extract combined acoustic feature vector from a .wav file.

    Returns:
        np.ndarray of shape (2*n_mfcc + 7,)
        [mfcc_mean(40), mfcc_std(40), centroid_mean, centroid_std,
         pitch_mean, pitch_std, pitch_var, voiced_ratio, rms_mean]
    """
    file_path = Path(file_path)
    y, sr = librosa.load(str(file_path), sr=None)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    centroid_mean, centroid_std = _safe_stats(centroid)

    # Pitch via piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for col in range(pitches.shape[1]):
        idx = np.argmax(magnitudes[:, col])
        val = pitches[idx, col]
        if val > 0:
            pitch_values.append(val)
    pitch_values = np.asarray(pitch_values, dtype=np.float32)
    pitch_mean, pitch_std = _safe_stats(pitch_values)
    pitch_var = float(np.var(pitch_values)) if pitch_values.size > 0 else 0.0
    voiced_ratio = float(pitch_values.size / max(1, pitches.shape[1]))

    # Energy proxy
    rms = librosa.feature.rms(y=y).flatten()
    rms_mean, _ = _safe_stats(rms)

    return np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            np.array(
                [
                    centroid_mean,
                    centroid_std,
                    pitch_mean,
                    pitch_std,
                    pitch_var,
                    voiced_ratio,
                    rms_mean,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def _load_dataset(
    autistic_dir: Path = AUTISTIC_DIR, normal_dir: Path = NORMAL_DIR
) -> Tuple[np.ndarray, np.ndarray]:
    """Load all wav files and return features X and labels y."""
    X: List[np.ndarray] = []
    y: List[int] = []

    for p in autistic_dir.glob("*.wav"):
        try:
            X.append(extract_audio_features(p))
            y.append(1)
        except Exception as exc:
            print(f"Skipping {p} due to error: {exc}")

    for p in normal_dir.glob("*.wav"):
        try:
            X.append(extract_audio_features(p))
            y.append(0)
        except Exception as exc:
            print(f"Skipping {p} due to error: {exc}")

    if not X:
        raise FileNotFoundError(
            "No valid .wav files found in audio_data/autistic or audio_data/normal."
        )

    return np.vstack(X), np.array(y, dtype=np.int64)


def train_audio_model(
    autistic_dir: str | Path = AUTISTIC_DIR,
    normal_dir: str | Path = NORMAL_DIR,
    model_path: str | Path = MODEL_PATH,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Train audio-only Random Forest model and save artifact to disk.

    Returns:
        Dict with simple training summary.
    """
    autistic_dir = Path(autistic_dir)
    normal_dir = Path(normal_dir)
    model_path = Path(model_path)

    X, y = _load_dataset(autistic_dir, normal_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    acc = float(clf.score(X_test, y_test))

    # Keep normal-class reference stats for anomaly rules in prediction.
    X_normal = X[y == 0]
    normal_mean = np.mean(X_normal, axis=0)
    normal_std = np.std(X_normal, axis=0) + 1e-8

    artifact = {
        "model": clf,
        "feature_config": {"n_mfcc": 40},
        "normal_mean": normal_mean.astype(np.float32),
        "normal_std": normal_std.astype(np.float32),
        "labels": {"normal": 0, "autistic": 1},
        "meta": {
            "num_samples": int(len(y)),
            "num_autistic": int(np.sum(y == 1)),
            "num_normal": int(np.sum(y == 0)),
            "test_accuracy": acc,
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved audio model to: {model_path}")
    print(f"Test accuracy: {acc:.4f}")

    return artifact["meta"]


def _detect_anomalies(feature_vec: np.ndarray, normal_mean: np.ndarray, normal_std: np.ndarray) -> List[str]:
    """
    Rule-based anomaly tags from normalized feature deviations.

    Feature tail order:
    [-7] centroid_mean
    [-6] centroid_std
    [-5] pitch_mean
    [-4] pitch_std
    [-3] pitch_var
    [-2] voiced_ratio
    [-1] rms_mean
    """
    z = np.abs((feature_vec - normal_mean) / normal_std)
    anomalies: List[str] = []

    centroid_std_z = float(z[-6])
    pitch_mean_z = float(z[-5])
    pitch_std_z = float(z[-4])
    pitch_var_z = float(z[-3])
    voiced_ratio_z = float(z[-2])
    rms_mean_z = float(z[-1])

    if pitch_std_z > 2.0 or pitch_var_z > 2.0:
        anomalies.append("High Variance")
    if pitch_mean_z > 2.0 or voiced_ratio_z > 2.0:
        anomalies.append("Atypical Prosody")
    if centroid_std_z > 2.0:
        anomalies.append("Unusual Spectral Dynamics")
    if rms_mean_z > 2.0:
        anomalies.append("Atypical Loudness Pattern")

    if not anomalies:
        anomalies.append("No strong acoustic anomalies detected")

    return anomalies


def predict_audio(file_path: str | Path, model_path: str | Path = MODEL_PATH) -> Dict[str, object]:
    """
    Predict autism likelihood from a single audio file.

    Returns:
        {
          "prediction": 0/1,
          "label": "Normal"/"Autistic",
          "likelihood_score": float in [0,1],  # P(autistic)
          "detected_acoustic_anomalies": [str, ...]
        }
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train_audio_model() first.")

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    clf: RandomForestClassifier = artifact["model"]
    normal_mean = artifact["normal_mean"]
    normal_std = artifact["normal_std"]
    n_mfcc = artifact.get("feature_config", {}).get("n_mfcc", 40)

    feat = extract_audio_features(file_path, n_mfcc=n_mfcc)
    X = feat.reshape(1, -1)

    pred = int(clf.predict(X)[0])
    if hasattr(clf, "predict_proba"):
        likelihood = float(clf.predict_proba(X)[0][1])  # P(autistic=1)
    else:
        likelihood = float(pred)

    anomalies = _detect_anomalies(feat, normal_mean, normal_std)

    return {
        "prediction": pred,
        "label": "Autistic" if pred == 1 else "Normal",
        "likelihood_score": likelihood,
        "detected_acoustic_anomalies": anomalies,
    }


if __name__ == "__main__":
    # Run training when executed directly.
    train_audio_model()
