import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC


LOGGER = logging.getLogger("brain_tumor.main")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_BUNDLE_PATH = os.path.join(ARTIFACT_DIR, "model_bundle.joblib")
TRAINING_REPORT_PATH = os.path.join(ARTIFACT_DIR, "training_report.json")
MODEL_VERSION = "2.1.1"

DATASET_LABELS = {
    "no_tumor": 0,
    "pituitary_tumor": 1,
    "meningioma_tumor": 2,
    "glioma_tumor": 3,
}
LABEL_TO_NAME = {value: key for key, value in DATASET_LABELS.items()}
IMAGE_SIZE = (200, 200)

RANDOM_STATE = 10
TEST_SIZE = 0.3
PCA_VARIANCE = 0.98
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "70"))
ENABLE_CLAHE_DEFAULT = os.getenv("ENABLE_CLAHE", "0") == "1"
ENABLE_TRAIN_AUGMENTATION = os.getenv("ENABLE_TRAIN_AUGMENTATION", "0") == "1"


def _dataset_class_path(class_name):
    return os.path.join(BASE_DIR, class_name)


def _compute_dataset_fingerprint():
    digest = hashlib.sha256()

    for class_name in sorted(DATASET_LABELS.keys()):
        class_path = _dataset_class_path(class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in sorted(os.listdir(class_path)):
            file_path = os.path.join(class_path, filename)
            if not os.path.isfile(file_path):
                continue

            stat_result = os.stat(file_path)
            digest.update(
                f"{class_name}|{filename}|{stat_result.st_size}|{int(stat_result.st_mtime)}".encode("utf-8")
            )

    return digest.hexdigest()


def _load_dataset():
    images = []
    labels = []

    for class_name, class_id in DATASET_LABELS.items():
        class_path = _dataset_class_path(class_name)
        if not os.path.isdir(class_path):
            LOGGER.warning("Missing dataset directory: %s", class_path)
            continue

        for filename in sorted(os.listdir(class_path)):
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                LOGGER.warning("Skipping unreadable image: %s", image_path)
                continue

            image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            images.append(image)
            labels.append(class_id)

    if not images:
        raise ValueError("No valid training images were found in dataset folders.")

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


def _evaluate_model(y_true, y_pred):
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1_score),
    }


def _class_distribution(labels):
    values, counts = np.unique(labels, return_counts=True)
    distribution = {}
    for value, count in zip(values.tolist(), counts.tolist()):
        distribution[LABEL_TO_NAME[int(value)]] = int(count)
    return distribution


def _augment_training_split(images, labels):
    if not ENABLE_TRAIN_AUGMENTATION:
        return images, labels, {
            "enabled": False,
            "strategies": ["original_only"],
            "input_samples": int(len(images)),
            "output_samples": int(len(images)),
        }

    flipped_images = np.flip(images, axis=2).copy()
    clahe_images = np.array([_apply_clahe(image) for image in images], dtype=np.uint8)

    augmented_images = np.concatenate([images, flipped_images, clahe_images], axis=0)
    augmented_labels = np.concatenate([labels, labels, labels], axis=0)

    rng = np.random.default_rng(RANDOM_STATE)
    shuffled_indices = rng.permutation(len(augmented_labels))
    augmented_images = augmented_images[shuffled_indices]
    augmented_labels = augmented_labels[shuffled_indices]

    report = {
        "enabled": True,
        "strategies": ["original", "horizontal_flip", "clahe"],
        "input_samples": int(len(images)),
        "output_samples": int(len(augmented_images)),
    }
    return augmented_images, augmented_labels, report


def _train_model(enable_grid_search=True):
    train_start = time.perf_counter()

    x, y = _load_dataset()
    x_train_images, x_test_images, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_images_before_augmentation = int(len(x_train_images))
    x_train_images, y_train, augmentation_report = _augment_training_split(x_train_images, y_train)

    x_train = x_train_images.reshape(len(x_train_images), -1).astype(np.float32) / 255.0
    x_test = x_test_images.reshape(len(x_test_images), -1).astype(np.float32) / 255.0

    pca = PCA(PCA_VARIANCE, random_state=RANDOM_STATE)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    logistic_model = LogisticRegression(max_iter=3000, C=1.0, random_state=RANDOM_STATE)
    logistic_model.fit(x_train_pca, y_train)
    logistic_predictions = logistic_model.predict(x_test_pca)
    logistic_metrics = _evaluate_model(y_test, logistic_predictions)

    svm_model = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    best_params = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": None}
    cv_best_score = None
    search_space_size = 1

    if enable_grid_search:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        param_grid = {
            "C": [0.5, 1, 5, 10, 20, 40],
            "gamma": ["scale", "auto", 0.001, 0.003, 0.005, 0.01],
            "class_weight": [None, "balanced"],
        }
        search_space_size = len(param_grid["C"]) * len(param_grid["gamma"]) * len(param_grid["class_weight"])

        grid_search = GridSearchCV(
            estimator=SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE),
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
        )
        grid_search.fit(x_train_pca, y_train)

        best_params = {
            "kernel": "rbf",
            "C": float(grid_search.best_params_["C"]),
            "gamma": grid_search.best_params_["gamma"],
            "class_weight": grid_search.best_params_["class_weight"],
        }

        svm_model = SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
            C=best_params["C"],
            gamma=best_params["gamma"],
            class_weight=best_params["class_weight"],
        )
        svm_model.fit(x_train_pca, y_train)

        cv_best_score = float(grid_search.best_score_)
    else:
        svm_model.fit(x_train_pca, y_train)

    svm_predictions = svm_model.predict(x_test_pca)
    svm_probabilities = svm_model.predict_proba(x_test_pca)
    svm_metrics = _evaluate_model(y_test, svm_predictions)

    confusion = confusion_matrix(y_test, svm_predictions, labels=sorted(LABEL_TO_NAME.keys())).tolist()

    calibration_samples = []
    for i, predicted_class in enumerate(svm_predictions):
        top_confidence = float(np.max(svm_probabilities[i]))
        calibration_samples.append(
            {
                "confidence": top_confidence,
                "correct": int(predicted_class == y_test[i]),
                "predicted_class": LABEL_TO_NAME[int(predicted_class)],
                "true_class": LABEL_TO_NAME[int(y_test[i])],
            }
        )

    elapsed_ms = round((time.perf_counter() - train_start) * 1000, 2)

    training_report = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "pca_variance": PCA_VARIANCE,
        "pca_components": int(pca.n_components_),
        "pca_explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_.tolist()],
        "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "dataset": {
            "total_images": int(len(x)),
            "train_images": train_images_before_augmentation,
            "train_images_after_augmentation": int(len(x_train)),
            "test_images": int(len(x_test)),
            "class_distribution": _class_distribution(y),
        },
        "augmentation": augmentation_report,
        "baseline_logistic_regression": {
            "metrics": logistic_metrics,
        },
        "svm": {
            "search_space_size": int(search_space_size),
            "best_params": best_params,
            "cv_best_accuracy": cv_best_score,
            "metrics": svm_metrics,
            "confusion_matrix": confusion,
        },
        "calibration_samples": calibration_samples,
        "timing": {
            "training_total_ms": elapsed_ms,
        },
    }

    LOGGER.info(
        "Training complete. SVM accuracy=%.4f, Logistic accuracy=%.4f, PCA components=%s, train_after_augmentation=%s",
        svm_metrics["accuracy"],
        logistic_metrics["accuracy"],
        pca.n_components_,
        len(x_train),
    )

    return pca, svm_model, training_report


def _save_training_report(training_report):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(TRAINING_REPORT_PATH, "w", encoding="utf-8") as report_file:
        json.dump(training_report, report_file, indent=2)


def _load_training_report():
    if not os.path.exists(TRAINING_REPORT_PATH):
        return {}

    try:
        with open(TRAINING_REPORT_PATH, "r", encoding="utf-8") as report_file:
            return json.load(report_file)
    except (json.JSONDecodeError, OSError):
        LOGGER.warning("Training report file is unreadable, will regenerate.")
        return {}


def _save_model_bundle(pca_model, svm_model, metadata):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(
        {
            "pca": pca_model,
            "svm": svm_model,
            "metadata": metadata,
        },
        MODEL_BUNDLE_PATH,
    )


def _load_model_bundle(dataset_fingerprint):
    if not os.path.exists(MODEL_BUNDLE_PATH):
        return None

    try:
        bundle = joblib.load(MODEL_BUNDLE_PATH)
    except Exception:
        LOGGER.warning("Model bundle exists but could not be loaded. Retraining.")
        return None

    metadata = bundle.get("metadata", {})
    if metadata.get("model_version") != MODEL_VERSION:
        LOGGER.info("Model bundle version mismatch. Retraining.")
        return None

    if metadata.get("dataset_fingerprint") != dataset_fingerprint:
        LOGGER.info("Dataset fingerprint changed. Retraining.")
        return None

    if "pca" not in bundle or "svm" not in bundle:
        LOGGER.info("Model bundle missing required keys. Retraining.")
        return None

    LOGGER.info("Loaded persisted model bundle from disk.")
    return bundle


def _build_metadata(dataset_fingerprint):
    return {
        "model_version": MODEL_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "image_size": [IMAGE_SIZE[0], IMAGE_SIZE[1]],
        "labels": DATASET_LABELS,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
    }


def _initialize_models():
    dataset_fingerprint = _compute_dataset_fingerprint()
    force_retrain = os.getenv("FORCE_RETRAIN", "0") == "1"

    if not force_retrain:
        bundle = _load_model_bundle(dataset_fingerprint)
        if bundle is not None:
            metadata = bundle["metadata"]
            training_report = _load_training_report()
            return bundle["pca"], bundle["svm"], metadata, training_report

    pca_model, svm_model, training_report = _train_model(enable_grid_search=True)
    metadata = _build_metadata(dataset_fingerprint)
    _save_model_bundle(pca_model, svm_model, metadata)
    _save_training_report(training_report)
    return pca_model, svm_model, metadata, training_report


def _apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def _preprocess_uploaded_image(file, enable_clahe=None):
    preprocess_start = time.perf_counter()
    image_bytes = file.read()

    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    image_buffer = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to decode uploaded image.")

    if enable_clahe is None:
        enable_clahe = ENABLE_CLAHE_DEFAULT

    if enable_clahe:
        image = _apply_clahe(image)

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = image.reshape(1, -1).astype(np.float32) / 255.0

    elapsed_ms = round((time.perf_counter() - preprocess_start) * 1000, 2)
    LOGGER.info("Preprocessing finished in %sms (clahe=%s)", elapsed_ms, bool(enable_clahe))

    return image


def get_tumor_prediction(file, enable_clahe=None):
    inference_start = time.perf_counter()

    processed_image = _preprocess_uploaded_image(file, enable_clahe=enable_clahe)
    transformed = PCA_MODEL.transform(processed_image)

    prediction = SVM_MODEL.predict(transformed)
    probabilities = SVM_MODEL.predict_proba(transformed)[0]

    predicted_index = int(prediction[0])
    predicted_label = LABEL_TO_NAME[predicted_index]
    confidence = float(np.max(probabilities))

    elapsed_ms = round((time.perf_counter() - inference_start) * 1000, 2)
    LOGGER.info("Inference finished in %sms", elapsed_ms)

    return {
        "tumor_type": predicted_label,
        "confidence": confidence,
        "low_confidence": confidence * 100 < LOW_CONFIDENCE_THRESHOLD,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "model_version": MODEL_METADATA.get("model_version", MODEL_VERSION),
        "inference_ms": elapsed_ms,
    }


def get_tumor_type(file):
    result = get_tumor_prediction(file)
    return result["tumor_type"]


def get_training_report():
    return TRAINING_REPORT


def get_model_metadata():
    return MODEL_METADATA


PCA_MODEL, SVM_MODEL, MODEL_METADATA, TRAINING_REPORT = _initialize_models()
