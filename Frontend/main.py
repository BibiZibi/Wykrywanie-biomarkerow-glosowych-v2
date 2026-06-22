import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import librosa
import opensmile
import shap

SAMPLE_RATE = 16000

# =========================
# AUDIO
# =========================
def load_audio(filepath, sr=SAMPLE_RATE):
    try:
        y, sr = librosa.load(filepath, sr=sr, mono=True)
        return y, sr
    except:
        return None, None

def normalize_audio(y):
    peak = np.abs(y).max()
    return y / peak if peak > 0 else y

# =========================
# FEATURES
# =========================
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_features_from_signal(y, sr):
    try:
        feats = smile.process_signal(y, sr)
        return feats.iloc[0]
    except:
        return None

# =========================
# PREDYKCJA
# =========================
def predict_from_audio(filepath, model, scaler):
    y, sr = load_audio(filepath)

    if y is None:
        return None

    y = normalize_audio(y)
    features = extract_features_from_signal(y, sr)

    if features is None:
        return None

    X = pd.DataFrame([features])
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0][1]
    pred = model.predict(X_scaled)[0]

    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        shap_vals = shap_values[0]
    except:
        shap_vals = np.zeros(X_scaled.shape[1])


    shap_vals = shap_values[0]
    feature_names = X.columns

    idxs = np.argsort(np.abs(shap_vals))[::-1][:5]

    top_features = []
    for i in idxs:
        top_features.append({
            "name": feature_names[i],
            "impact": shap_vals[i]
        })

    return {
        "prediction": pred,
        "probability": proba,
        "feature": top_features[0]["name"],
        "impact": top_features[0]["impact"],
        "top_features": top_features
    }
