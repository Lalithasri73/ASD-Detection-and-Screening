from flask import Flask, request, render_template, redirect, url_for, session, send_file
import base64
import io
import os
import pickle
import tempfile
from datetime import datetime, timezone
import json
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import warnings
from werkzeug.utils import secure_filename
from audio_processor import predict_audio

# Hide noisy, non-fatal sklearn warnings coming from:
# 1) Loading pickled models trained with a different sklearn version
# 2) A feature-names warning during predict()
try:
    from sklearn.exceptions import InconsistentVersionWarning  # type: ignore

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    # Fallback: regex-match the warning message prefix
    warnings.filterwarnings("ignore", message=r"Trying to unpickle estimator.*")

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names.*",
    category=UserWarning,
)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# --- 1. MODEL LOADING & ENCODERS ---
try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    best_model_path = os.path.join(APP_DIR, "model", "best_model.pkl")
    adaboost_model_path = os.path.join(APP_DIR, "model", "adaboost_model.pkl")
    audio_model_path = os.path.join(APP_DIR, "model", "audio_classifier.pkl")

    model_rf = pickle.load(open(best_model_path, "rb"))
    label_encoders = pickle.load(open(os.path.join(APP_DIR, "model", "encoder.pkl"), "rb"))
    TRAIN_CSV_PATH = os.path.join(APP_DIR, "train.csv")
except Exception as e:
    print(f"Initialization Error: {str(e)}")
    raise

USERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")


def _load_users():
    if not os.path.exists(USERS_PATH):
        return {"admin": "password"}
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data:
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {"admin": "password"}


def _save_users(users_dict):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users_dict, f, indent=2)


users = _load_users()

# --- 2. VISUALIZATION UTILITY ---
FEATURE_ORDER = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score",
    "age",
    "gender",
    "ethnicity",
    "jaundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "result",
    "relation",
]


def _safe_transform(le, value, fallback=None):
    value = str(value or "").strip()
    classes = getattr(le, "classes_", [])
    if value in classes:
        return int(le.transform([value])[0])
    if fallback is not None and fallback in classes:
        return int(le.transform([fallback])[0])
    return int(le.transform([classes[0]])[0])


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _prepare_encoded_xy():
    df = pd.read_csv(TRAIN_CSV_PATH)
    X = df[FEATURE_ORDER].copy()
    y = df["Class/ASD"].astype(int)

    for col, le in label_encoders.items():
        X[col] = X[col].astype(str).map(lambda v: v.strip())
        if col == "ethnicity":
            X[col] = X[col].replace({"Middle Eastern": "Middle Eastern "})
        X[col] = X[col].map(lambda v: v if v in le.classes_ else le.classes_[0])
        X[col] = le.transform(X[col])
    return X, y


def _load_or_train_adaboost(X_train, y_train):
    # If a real model exists on disk, load it; otherwise train a new one.
    try:
        if os.path.exists(adaboost_model_path) and os.path.getsize(adaboost_model_path) > 0:
            return pickle.load(open(adaboost_model_path, "rb"))
    except Exception:
        pass

    clf = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.5,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Best-effort save so next run is faster.
    try:
        with open(adaboost_model_path, "wb") as f:
            pickle.dump(clf, f)
    except Exception:
        pass

    return clf


def _compute_metrics_for_model(model_obj, X_train, X_test, y_train, y_test):
    model_obj.fit(X_train, y_train)
    y_pred = model_obj.predict(X_test)

    # Overall metrics (treat class 1 = Autistic as positive)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_pos = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    recall_pos = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1_pos = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    # Metrics per class (0 = Non-autistic, 1 = Autistic)
    precision_neg = float(precision_score(y_test, y_pred, pos_label=0, zero_division=0))
    recall_neg = float(recall_score(y_test, y_pred, pos_label=0, zero_division=0))
    f1_neg = float(f1_score(y_test, y_pred, pos_label=0, zero_division=0))

    metrics = {
        "accuracy": accuracy,
        # Overall (positive class = Autistic)
        "precision": precision_pos,
        "recall": recall_pos,
        "f1": f1_pos,
        # Explicit non-autistic vs autistic metrics
        "precision_0": precision_neg,
        "recall_0": recall_neg,
        "f1_0": f1_neg,
        "precision_1": precision_pos,
        "recall_1": recall_pos,
        "f1_1": f1_pos,
        "report": classification_report(y_test, y_pred, digits=4, zero_division=0),
        "cm": confusion_matrix(y_test, y_pred),
    }

    # Confusion matrix plot
    fig_cm = plt.figure(figsize=(5.2, 4.2))
    ax = fig_cm.add_subplot(111)
    sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="viridis", cbar=True, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["NO", "YES"])
    ax.set_yticklabels(["NO", "YES"])
    cm_b64 = _fig_to_b64(fig_cm)

    # Distribution chart (test split)
    fig_dist = plt.figure(figsize=(5.2, 4.0))
    ax2 = fig_dist.add_subplot(111)
    vals, counts = np.unique(y_test, return_counts=True)
    counts_map = {int(v): int(c) for v, c in zip(vals, counts)}
    ax2.pie(
        [counts_map.get(0, 0), counts_map.get(1, 0)],
        labels=["NO", "YES"],
        autopct="%1.0f%%",
        colors=["#ff4b5c", "#00d2ff"],
    )
    ax2.set_title("Classification Results Distribution (Test Split)")
    dist_b64 = _fig_to_b64(fig_dist)

    # Feature importance chart
    importance_b64 = ""
    if hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
        idx = np.argsort(importances)[::-1][:12]
        fig_imp = plt.figure(figsize=(8.4, 4.2))
        ax3 = fig_imp.add_subplot(111)
        ax3.barh([FEATURE_ORDER[i] for i in idx][::-1], importances[idx][::-1], color="#6a11cb")
        ax3.set_title("Top Feature Importances")
        ax3.set_xlabel("Importance")
        importance_b64 = _fig_to_b64(fig_imp)

    metrics["cm_plot"] = cm_b64
    metrics["dist_plot"] = dist_b64
    metrics["importance_plot"] = importance_b64
    return metrics


X_ALL, Y_ALL = _prepare_encoded_xy()
X_train, X_test, y_train, y_test = train_test_split(
    X_ALL, Y_ALL, test_size=0.2, random_state=42, stratify=Y_ALL
)

model_ab = _load_or_train_adaboost(X_train, y_train)

MODELS = {
    "rf": model_rf,
    "ab": model_ab,
}

MODEL_LABELS = {
    "rf": "Random Forest (Best Model)",
    "ab": "AdaBoost",
}

METRICS_BY_MODEL = {
    "rf": _compute_metrics_for_model(MODELS["rf"], X_train, X_test, y_train, y_test),
    "ab": _compute_metrics_for_model(MODELS["ab"], X_train, X_test, y_train, y_test),
}

# Optional audio model for multimodal prediction.
# Optional audio model for multimodal prediction.
AUDIO_MODEL = None
print(f"DEBUG: Checking for audio model at: {audio_model_path}")

if os.path.exists(audio_model_path):
    try:
        if os.path.getsize(audio_model_path) > 0:
            with open(audio_model_path, "rb") as f:
                AUDIO_MODEL = pickle.load(f)
            print("✅ SUCCESS: Audio model loaded into memory.")
        else:
            print("❌ ERROR: audio_classifier.pkl is empty.")
    except Exception as e:
        print(f"❌ ERROR: Failed to unpickle audio model: {e}")
        AUDIO_MODEL = None
else:
    print("⚠️ WARNING: audio_classifier.pkl not found. App will run in Behavioral-only mode.")

def _extract_audio_mfcc_mean_from_upload(file_storage, n_mfcc: int = 40) -> np.ndarray:
    if file_storage is None or not file_storage.filename:
        raise ValueError("No audio file provided.")

    # 1. Create a temporary file path
    import uuid
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_audio_upload_{uuid.uuid4().hex}.wav")

    # 2. Save the uploaded bytes directly to that path
    file_storage.save(temp_path)

    try:
        # 3. Load the file with librosa (now that it is saved and closed)
        y, sr = librosa.load(temp_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # 4. Clean up the temporary file immediately after reading
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return np.mean(mfcc, axis=1)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

# --- 3. ROUTES ---
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('homepage'))
        return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        username = request.args.get("username", "")
        return render_template("signup.html", username=username, error=None)

    username = (request.form.get("username") or "").strip()
    password = (request.form.get("password") or "").strip()
    if not username or not password:
        return render_template("signup.html", username=username, error="Username and password required.")
    if username in users:
        return render_template("signup.html", username=username, error="Username already exists.")

    users[username] = password
    _save_users(users)
    return redirect(url_for("login", signup_success=1, username=username))

@app.route('/homepage')
def homepage():
    if 'username' not in session: return redirect(url_for('login'))
    return render_template('homepage.html', username=session['username'])


@app.route("/about")
def about():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("about.html", username=session["username"])

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Set standard default values at the top so they are always defined
    default_features = {f"A{i}_Score": 0 for i in range(1, 11)}
    default_features.update({
        "age": "N/A", "gender": "N/A", "ethnicity": "N/A",
        "jaundice": "N/A", "austim": "N/A", "contry_of_res": "N/A",
        "used_app_before": "N/A", "relation": "N/A", "result": "N/A", "aq10_total": 0
    })

    res_data = {
        "behavioral_prediction": None,
        "behavioral_confidence": "0%",
        "audio_result": None,
        "hybrid_mode": "Single-Model",
        "prediction": "N/A",
        "confidence": "0%",
        "features": default_features,
        "model_name": "Unknown",
        "cm_plot": "",
        "dist_plot": ""
    }

    # --- 1. GET: This renders the form ---
    if request.method == 'GET':
        # This list ensures the A1-A10 questions appear!
        q_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return render_template("predict.html", 
                               username=session.get('username'), 
                               q_numbers=q_numbers)

    # --- 2. POST: This handles the prediction logic ---
    try:
        raw = request.form.to_dict(flat=True)
        mode = raw.get("prediction_mode", "behavior")
        audio_file = request.files.get("audio_file")

        if mode == "behavior":
            # Extract questions safely using .get(key, 0)
            input_data = {f"A{i}_Score": int(raw.get(f"A{i}_Score", 0)) for i in range(1, 11)}
            
            # Create DataFrame in the exact order model expects
            input_df = pd.DataFrame([{
                **{f"A{i}_Score": input_data[f"A{i}_Score"] for i in range(1, 11)},
                "age": float(raw.get("age", 0)),
                "gender": _safe_transform(label_encoders["gender"], raw.get("gender", "m")),
                "ethnicity": _safe_transform(label_encoders["ethnicity"], raw.get("ethnicity", "Others")),
                "jaundice": _safe_transform(label_encoders["jaundice"], raw.get("jaundice", "no")),
                "austim": _safe_transform(label_encoders["austim"], raw.get("austim", "no")),
                "contry_of_res": _safe_transform(label_encoders["contry_of_res"], raw.get("contry_of_res", "India")),
                "used_app_before": _safe_transform(label_encoders["used_app_before"], "no"),
                "result": float(raw.get("result", 0)),
                "relation": _safe_transform(label_encoders["relation"], raw.get("relation", "Self")),
            }])

            # Get selected model
            model_id = session.get("selected_model", "rf")
            model_obj = MODELS.get(model_id, MODELS["rf"])

            # Run Behavioral Model
            pred = int(model_obj.predict(input_df[FEATURE_ORDER].values)[0])
            proba = model_obj.predict_proba(input_df[FEATURE_ORDER].values)[0][1]
            
            res_data.update({
                "prediction": "YES" if pred == 1 else "NO",
                "confidence": f"{proba*100:.2f}%",
                "behavioral_prediction": "YES" if pred == 1 else "NO",
                "behavioral_confidence": f"{proba*100:.2f}%",
                "hybrid_mode": "Behavioral-Only",
                "features": {
                    **input_data,
                    "age": raw.get("age", "0"),
                    "gender": raw.get("gender", "m"),
                    "ethnicity": raw.get("ethnicity", "Others"),
                    "jaundice": raw.get("jaundice", "no"),
                    "austim": raw.get("austim", "no"),
                    "contry_of_res": raw.get("contry_of_res", "India"),
                    "used_app_before": "no",
                    "result": raw.get("result", "0"),
                    "relation": raw.get("relation", "Self"),
                    "aq10_total": sum(input_data.values())
                },
                "model_name": MODEL_LABELS.get(model_id, "Unknown Model"),
                "cm_plot": METRICS_BY_MODEL[model_id]["cm_plot"],
                "dist_plot": METRICS_BY_MODEL[model_id]["dist_plot"]
            })

            session["latest_assessment"] = {
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "username": session.get("username"),
                "prediction": res_data["prediction"],
                "confidence": res_data["confidence"],
                "features": input_df.iloc[0].to_dict(),
                "model_id": model_id,
                "model_name": MODEL_LABELS.get(model_id, "Unknown Model")
            }

        elif mode == "audio":
            if audio_file and audio_file.filename:
                try:
                    import uuid
                    from werkzeug.utils import secure_filename
                    upload_dir = os.path.join(APP_DIR, "static", "uploads")
                    os.makedirs(upload_dir, exist_ok=True)
                    fname = secure_filename(audio_file.filename) or "audio_upload.wav"
                    stamped = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{fname}"
                    saved_path = os.path.join(upload_dir, stamped)
                    audio_file.save(saved_path)
                    
                    # Run Audio Model directly from saved file
                    y, sr = librosa.load(saved_path, sr=None)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                    feat = np.mean(mfcc, axis=1)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    pred = int(AUDIO_MODEL.predict(np.array([feat]))[0])
                    proba = AUDIO_MODEL.predict_proba(np.array([feat]))[0][1]
                    
                    audio_url = url_for("static", filename=f"uploads/{stamped}")
                    
                    res_data.update({
                        "prediction": "YES" if pred == 1 else "NO",
                        "confidence": f"{proba*100:.2f}%",
                        "audio_result": {"prediction": "YES" if pred == 1 else "NO", "confidence": f"{proba*100:.2f}%", "status": "Success"},
                        "hybrid_mode": "Audio-Only",
                        "model_name": "Audio Classifier",
                        "features": None,
                        "audio_url": audio_url,
                        "audio_filename": audio_file.filename
                    })

                    session["latest_assessment"] = {
                        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        "username": session.get("username"),
                        "prediction": res_data["prediction"],
                        "confidence": res_data["confidence"],
                        "features": {},
                        "model_id": "audio",
                        "model_name": "Audio Classifier",
                        "hybrid_mode": "Audio-Only",
                        "audio_filename": audio_file.filename,
                        "audio_saved_path": saved_path,
                        "audio_duration": duration
                    }
                except Exception as e:
                    res_data.update({
                        "prediction": "Error processing audio file",
                        "confidence": "0%",
                        "audio_result": {"prediction": "Error", "confidence": "0%", "status": "Failed", "error": str(e)},
                        "hybrid_mode": "Audio-Only",
                        "model_name": "Audio Classifier",
                        "features": None
                    })
                    session["latest_assessment"] = {
                        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        "username": session.get("username"),
                        "prediction": res_data["prediction"],
                        "confidence": res_data["confidence"],
                        "features": {},
                        "model_id": "audio",
                        "model_name": "Audio Classifier",
                        "hybrid_mode": "Audio-Only",
                        "audio_filename": audio_file.filename
                    }
            else:
                res_data.update({
                    "prediction": "Error: No audio file provided",
                    "confidence": "0%",
                    "hybrid_mode": "Audio-Only",
                    "model_name": "Audio Classifier",
                    "features": None
                })
        else:
            res_data.update({
                "prediction": "Error: Unknown prediction mode",
                "confidence": "0%",
                "hybrid_mode": "Unknown"
            })

        return render_template("result.html", username=session.get('username'), **res_data)

    except Exception as e:
        print(f"Error during prediction: {e}")
        res_data["prediction"] = f"Error: {e}"
        return render_template("result.html", username=session.get('username'), **res_data)


@app.get("/download/csv")
def download_result_csv():
    if "username" not in session:
        return redirect(url_for("login"))
    assessment = session.get("latest_assessment")
    if not assessment:
        return redirect(url_for("predict_page"))

    flat = {
        "timestamp_utc": assessment.get("timestamp_utc"),
        "username": assessment.get("username"),
        "prediction": assessment.get("prediction"),
        "confidence": assessment.get("confidence"),
        **(assessment.get("features") or {}),
    }
    df = pd.DataFrame([flat])
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    bio.seek(0)
    filename = f"autism_assessment_{session['username']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(bio, mimetype="text/csv", as_attachment=True, download_name=filename)


@app.get("/download/pdf")
def download_result_pdf():
    if "username" not in session:
        return redirect(url_for("login"))
    assessment = session.get("latest_assessment")
    if not assessment:
        return redirect(url_for("predict_page"))

    model_id = assessment.get("model_id", session.get("selected_model", "rf"))
    if model_id not in METRICS_BY_MODEL:
        model_id = "rf"
    metrics_for_model = METRICS_BY_MODEL[model_id]

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    except Exception as e:
        return f"PDF dependency missing. Install reportlab. Error: {e}", 500

    def _b64_to_image_buffer(b64_str: str):
        data = base64.b64decode(b64_str.encode("utf-8"))
        buf = io.BytesIO(data)
        buf.seek(0)
        return buf

    features = assessment.get("features") or {}
    bio = io.BytesIO()
    doc = SimpleDocTemplate(
        bio,
        pagesize=A4,
        title="Autism Screening Result",
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleBlue",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#1f4e8c"),
        spaceAfter=4,
    )
    section_style = ParagraphStyle(
        "SectionBlue",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#0f3b71"),
        spaceBefore=4,
        spaceAfter=6,
    )
    subtle_style = ParagraphStyle(
        "Subtle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#666666"),
    )

    story = []
    story.append(Paragraph("Autism Prediction System", title_style))
    story.append(Paragraph("Comprehensive Screening Report", styles["Heading2"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("This document is generated automatically from the latest assessment.", subtle_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Name/Username: <b>{assessment.get('username','')}</b>", styles["Normal"]))
    story.append(Paragraph(f"Date (UTC): <b>{assessment.get('timestamp_utc','')}</b>", styles["Normal"]))
    story.append(Paragraph(f"Model: <b>{assessment.get('model_name','')}</b>", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(f"Prediction: <b>{assessment.get('prediction','')}</b>", section_style))
    story.append(Paragraph(f"Confidence: <b>{assessment.get('confidence','')}</b>", styles["Normal"]))
    story.append(Spacer(1, 10))

    if assessment.get('hybrid_mode') == 'Audio-Only':
        story.append(Paragraph("Audio Analysis Report", section_style))
        audio_rows = [
            ["Audio Source Filename", str(assessment.get("audio_filename", "N/A"))],
            ["Duration (seconds)", f"{assessment.get('audio_duration', 0):.2f}"],
            ["Analysis Status", "Completed" if assessment.get("prediction") not in ["N/A", "Error processing audio file"] else "Failed"]
        ]
        audio_table = Table(audio_rows, hAlign="LEFT", colWidths=[2.2 * inch, 3.0 * inch])
        audio_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2575fc")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(audio_table)
        story.append(Spacer(1, 12))
        
        saved_path = assessment.get("audio_saved_path")
        if saved_path and os.path.exists(saved_path):
            try:
                import librosa.display
                y, sr = librosa.load(saved_path, sr=None)
                fig_wave = plt.figure(figsize=(6.0, 2.5))
                ax_wave = fig_wave.add_subplot(111)
                librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="#6a11cb")
                ax_wave.set_title("Waveform Visualization")
                wave_b64 = _fig_to_b64(fig_wave)
                
                story.append(Paragraph("Waveform Visualization", section_style))
                wave_img = Image(_b64_to_image_buffer(wave_b64), width=6.0 * inch, height=2.5 * inch)
                story.append(wave_img)
                story.append(Spacer(1, 12))
            except Exception as e:
                print(f"Failed to generate waveform: {e}")
        
        story.append(Paragraph("Digital evidence attached. Audio playback is available on the secure web portal.", subtle_style))
        story.append(Spacer(1, 12))
    else:
        # AQ10 table
        aq_rows = [["Question", "Value", "Yes/No"]]
        for i in range(1, 11):
            v = int(features.get(f"A{i}_Score", 0))
            aq_rows.append([f"A{i}_Score", str(v), "Yes" if v == 1 else "No"])
        aq_rows.append(["AQ10 Total", str(features.get("aq10_total", "")), ""])
        aq_table = Table(aq_rows, hAlign="LEFT", colWidths=[2.0 * inch, 1.0 * inch, 1.2 * inch])
        aq_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6a11cb")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("ALIGN", (1, 1), (1, -1), "CENTER"),
                    ("ALIGN", (2, 1), (2, -1), "CENTER"),
                ]
            )
        )
        story.append(Paragraph("Behavioral Responses (AQ-10)", section_style))
        story.append(aq_table)
        story.append(Spacer(1, 12))

        # Demographics table
        demo_rows = [
            ["Age", str(features.get("age", ""))],
            ["Gender", str(features.get("gender", ""))],
            ["Ethnicity", str(features.get("ethnicity", ""))],
            ["Jaundice", str(features.get("jaundice", ""))],
            ["Family History (Autism)", str(features.get("austim", ""))],
            ["Country", str(features.get("contry_of_res", ""))],
            ["Used App Before", str(features.get("used_app_before", ""))],
            ["Relation", str(features.get("relation", ""))],
            ["Previous Screening Score (result)", str(features.get("result", ""))],
        ]
        demo_table = Table([["Field", "Value"]] + demo_rows, hAlign="LEFT", colWidths=[2.2 * inch, 3.0 * inch])
        demo_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2575fc")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(Paragraph("Demographic Information", section_style))
        story.append(demo_table)
        story.append(Spacer(1, 12))

    # Performance summary
    if assessment.get('hybrid_mode') != 'Audio-Only':
        story.append(Paragraph("Model Performance (evaluation split)", section_style))
        perf_rows = [
            ["Accuracy (overall)", f"{metrics_for_model['accuracy']:.4f}"],
            ["Precision (Autistic=1)", f"{metrics_for_model['precision_1']:.4f}"],
            ["Recall (Autistic=1)", f"{metrics_for_model['recall_1']:.4f}"],
            ["Precision (Non-autistic=0)", f"{metrics_for_model['precision_0']:.4f}"],
            ["Recall (Non-autistic=0)", f"{metrics_for_model['recall_0']:.4f}"],
            ["F1 (Autistic=1)", f"{metrics_for_model['f1_1']:.4f}"],
            ["F1 (Non-autistic=0)", f"{metrics_for_model['f1_0']:.4f}"],
        ]
        perf_table = Table([["Metric", "Value"]] + perf_rows, hAlign="LEFT", colWidths=[3.0 * inch, 1.4 * inch])
        perf_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4fc3f7")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(perf_table)
        story.append(Spacer(1, 12))

        # Images
        story.append(Paragraph("Confusion Matrix", section_style))
        cm_img = Image(_b64_to_image_buffer(metrics_for_model["cm_plot"]), width=5.6 * inch, height=3.4 * inch)
        story.append(cm_img)
        story.append(Spacer(1, 10))

        story.append(Paragraph("Distribution Chart", section_style))
        dist_img = Image(_b64_to_image_buffer(metrics_for_model["dist_plot"]), width=5.2 * inch, height=3.4 * inch)
        story.append(dist_img)
        story.append(Spacer(1, 10))

        if metrics_for_model.get("importance_plot"):
            story.append(Paragraph("Feature Importance", section_style))
            imp_img = Image(_b64_to_image_buffer(metrics_for_model["importance_plot"]), width=6.2 * inch, height=3.2 * inch)
            story.append(imp_img)
        story.append(Spacer(1, 12))
        
    story.append(
        Paragraph(
            "Disclaimer: This report is for screening support only and is not a clinical diagnosis. "
            "Please consult a qualified healthcare professional for formal evaluation.",
            subtle_style,
        )
    )

    def _draw_footer(canvas, document):
        canvas.saveState()
        footer = "Autism Prediction System"
        page_text = f"Page {document.page}"
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(document.leftMargin, 20, footer)
        canvas.drawRightString(A4[0] - document.rightMargin, 20, page_text)
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    bio.seek(0)
    filename = f"autism_assessment_{session['username']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(bio, mimetype="application/pdf", as_attachment=True, download_name=filename)

@app.route('/performance')
def performance():
    if 'username' not in session: return redirect(url_for('login'))
    model_id = (request.args.get("model") or session.get("selected_model") or "rf").strip()
    if model_id not in METRICS_BY_MODEL:
        model_id = "rf"
    session["selected_model"] = model_id
    comparison_rows = []
    for mid, label in MODEL_LABELS.items():
        m = METRICS_BY_MODEL[mid]
        comparison_rows.append(
            {
                "model": label,
                "accuracy": m["accuracy"],
                "precision_1": m["precision_1"],
                "recall_1": m["recall_1"],
                "f1_1": m["f1_1"],
            }
        )
    return render_template(
        "performance.html",
        username=session["username"],
        metrics=METRICS_BY_MODEL[model_id],
        cm_plot=METRICS_BY_MODEL[model_id]["cm_plot"],
        model_options=[{"id": k, "label": v} for k, v in MODEL_LABELS.items()],
        selected_model=model_id,
        comparison_rows=comparison_rows,
    )

@app.route('/chart')
def chart():
    if 'username' not in session: return redirect(url_for('login'))
    model_id = (request.args.get("model") or session.get("selected_model") or "rf").strip()
    if model_id not in METRICS_BY_MODEL:
        model_id = "rf"
    session["selected_model"] = model_id
    return render_template(
        "chart.html",
        username=session["username"],
        metrics=METRICS_BY_MODEL[model_id],
        dist_plot=METRICS_BY_MODEL[model_id]["dist_plot"],
        importance_plot=METRICS_BY_MODEL[model_id].get("importance_plot", ""),
        model_options=[{"id": k, "label": v} for k, v in MODEL_LABELS.items()],
        selected_model=model_id,
    )

@app.route("/audio-detect", methods=["GET", "POST"])
def audio_detect():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "GET":
        return render_template("audio_detect.html", username=session["username"], error=None)

    try:
        audio_file = request.files.get("audio_file")
        if audio_file is None or not audio_file.filename:
            return render_template(
                "audio_detect.html",
                username=session["username"],
                error="Please upload a .wav file.",
            )

        fname = secure_filename(audio_file.filename)
        if not fname.lower().endswith(".wav"):
            return render_template(
                "audio_detect.html",
                username=session["username"],
                error="Only .wav files are supported.",
            )

        upload_dir = os.path.join(APP_DIR, "static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        stamped = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{fname}"
        saved_path = os.path.join(upload_dir, stamped)
        audio_file.save(saved_path)

        audio_out = predict_audio(saved_path)
        audio_url = url_for("static", filename=f"uploads/{stamped}")

        return render_template(
            "audio_result.html",
            username=session["username"],
            audio_url=audio_url,
            audio_result=audio_out,
        )
    except Exception as ex:
        return render_template(
            "audio_detect.html",
            username=session["username"],
            error=f"Audio processing failed: {ex}",
        )

@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)