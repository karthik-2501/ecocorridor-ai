"""
Machine Learning Module for Wildlife Species Classification.
Implements explainable supervised classification using Scikit-Learn Random Forest.
Features include spatio-temporal metadata, sensor sequence parameters, and location embeddings
with strict sequence-group split leakage prevention and interactive inference.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import streamlit as st

import config
from modules.preprocessing import build_ml_preprocessor, engineer_temporal_features, engineer_sequence_features


@st.cache_data(show_spinner=True)
def prepare_ml_data(
    df: pd.DataFrame,
    target_mode: str = "top_species",
    sample_size: int = 25000,
    random_state: int = 42,
    return_groups: bool = False
) -> Any:
    """
    Prepare feature matrix X and target y with strict leakage prevention.

    Target Modes:
    - 'top_species': Multi-class classification of top 7 frequent wildlife species + empty
    - 'wildlife_vs_empty': Binary classification (Wildlife Present vs Empty Trigger)
    - 'all_animals': Multi-class classification on animal-only records

    Features Included:
    - Hour, Month, Day of Week, Cyclical sine/cosine
    - Location identifier
    - Frame number, Sequence length, Frame ratio
    - Image width, height, aspect ratio

    Features Excluded:
    - image_id, file_name, rights_holder, labels (prevent target leakage)
    """
    data = df.copy()

    # Temporal and sequence engineering
    data = engineer_temporal_features(data)
    data = engineer_sequence_features(data)

    # Filter by target mode
    if target_mode == "wildlife_vs_empty":
        data = data[data["category"].isin(config.VALID_ANIMAL_SPECIES + ["empty"])].copy()
        data["target"] = data["category"].apply(lambda c: "Empty" if c == "empty" else "Wildlife Present")
    elif target_mode == "all_animals":
        data = data[data["category"].isin(config.VALID_ANIMAL_SPECIES)].copy()
        # Keep species with at least 500 samples
        valid_cats = data["category"].value_counts()[lambda x: x >= 500].index.tolist()
        data = data[data["category"].isin(valid_cats)].copy()
        data["target"] = data["category"]
    else:
        # Default: Top 7 species + empty
        top_cats = ["empty", "opossum", "coyote", "rabbit", "deer", "raccoon", "bird", "bobcat"]
        data = data[data["category"].isin(top_cats)].copy()
        data["target"] = data["category"]

    # Sample if dataset is very large for interactive responsiveness
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=random_state, stratify=data["target"])

    numeric_features = [
        "hour", "month", "day_of_week",
        "frame_num", "seq_num_frames", "frame_ratio",
        "width", "height", "aspect_ratio",
        "hour_sin", "hour_cos", "month_sin", "month_cos"
    ]
    # Keep only available numeric features
    numeric_features = [f for f in numeric_features if f in data.columns]

    categorical_features = ["location"]
    categorical_features = [f for f in categorical_features if f in data.columns]

    X = data[numeric_features + categorical_features].copy()
    y = data["target"].astype(str)
    groups = data["seq_id"] if "seq_id" in data.columns else None
    class_names = sorted(y.unique().tolist())

    if return_groups:
        return X, y, numeric_features, categorical_features, class_names, groups
    return X, y, numeric_features, categorical_features, class_names


def train_wildlife_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    n_estimators: int = 100,
    max_depth: Optional[int] = 15,
    test_size: float = 0.2,
    random_state: int = 42,
    groups: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Train a Random Forest Classifier with scikit-learn pipeline,
    and compute evaluation metrics, confusion matrix, and feature importances.
    Uses GroupShuffleSplit when sequence groups are available to prevent temporal burst leakage.
    """
    split_method = "Stratified Random Split"
    if groups is not None and len(groups) == len(X):
        # Sequence-aware group split to avoid burst correlation between train and test
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            split_method = "Sequence Group Split (Leakage-Free)"
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    # Build preprocessing pipeline
    preprocessor = build_ml_preprocessor(numeric_features, categorical_features)

    # Fit preprocessor
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Model training
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model.fit(X_train_proc, y_train)

    # Predictions
    y_pred = model.predict(X_test_proc)
    classes = model.classes_.tolist()

    # Performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Extract transformed feature names for importance mapping
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names
    importances = model.feature_importances_

    return {
        "model": model,
        "preprocessor": preprocessor,
        "classes": classes,
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "feature_names": all_feature_names,
        "feature_importances": importances,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "split_method": split_method
    }


def predict_sample(
    trained_artifacts: Dict[str, Any],
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Run inference for a single user-configured observation input."""
    preprocessor = trained_artifacts["preprocessor"]
    model = trained_artifacts["model"]
    classes = trained_artifacts["classes"]

    df_in = pd.DataFrame([input_data])
    # Add cyclical features if hour or month present
    if "hour" in df_in.columns:
        df_in["hour_sin"] = np.sin(2 * np.pi * df_in["hour"] / 24.0)
        df_in["hour_cos"] = np.cos(2 * np.pi * df_in["hour"] / 24.0)
    if "month" in df_in.columns:
        df_in["month_sin"] = np.sin(2 * np.pi * df_in["month"] / 12.0)
        df_in["month_cos"] = np.cos(2 * np.pi * df_in["month"] / 12.0)
    if "width" in df_in.columns and "height" in df_in.columns:
        df_in["aspect_ratio"] = (df_in["width"] / df_in["height"]).round(3)
    if "frame_num" in df_in.columns and "seq_num_frames" in df_in.columns:
        df_in["frame_ratio"] = (df_in["frame_num"] / df_in["seq_num_frames"]).clip(0.0, 1.0)

    # Ensure all required features are present
    for f in trained_artifacts["numeric_features"]:
        if f not in df_in.columns:
            df_in[f] = 0.0
    for f in trained_artifacts["categorical_features"]:
        if f not in df_in.columns:
            df_in[f] = "unknown"

    proc = preprocessor.transform(df_in[trained_artifacts["numeric_features"] + trained_artifacts["categorical_features"]])
    probs = model.predict_proba(proc)[0]
    pred_class = classes[int(np.argmax(probs))]

    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    return {
        "predicted_class": pred_class,
        "confidence": float(np.max(probs)),
        "probabilities": sorted_probs
    }


def check_vision_model_status() -> Dict[str, Any]:
    """
    Check the operational status of deep-learning computer vision models
    (such as MegaDetector V6) and explain the distinction between
    metadata-driven machine learning and raw-pixel computer vision.
    """
    status = {
        "vision_framework": "MegaDetector V6 (PyTorch-Wildlife / Ultralytics)",
        "framework_installed": False,
        "weights_available": False,
        "device": "CPU / CUDA",
        "description": (
            "MegaDetector V6 is an animal detection model architecture (YOLOv10 / RT-DETR) "
            "that outputs bounding boxes on raw camera-trap photographs. When raw image files "
            "are not downloaded locally, deep learning pixel inference is offline, and the system "
            "relies on verified dataset metadata and the Scikit-Learn Random Forest model."
        )
    }

    try:
        import torch
        status["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        status["device"] = "PyTorch not loaded in active environment"

    try:
        from src.megadetector_core import MegaDetectorV6
        status["framework_installed"] = True
    except (ImportError, Exception):
        status["framework_installed"] = False

    return status
