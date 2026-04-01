"""
Generate project visualizations:
1) PCA cumulative variance plot
2) Model performance comparison chart
3) SVM confusion matrix heatmap
4) Confidence distribution plot
5) Confidence calibration curve
"""

import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
TRAINING_REPORT_PATH = os.path.join(ARTIFACT_DIR, "training_report.json")

DATASET_LABELS = {
    "no_tumor": 0,
    "pituitary_tumor": 1,
    "meningioma_tumor": 2,
    "glioma_tumor": 3,
}
CLASS_ORDER = ["no_tumor", "pituitary_tumor", "meningioma_tumor", "glioma_tumor"]
IMAGE_SIZE = (200, 200)


def _load_training_report():
    if not os.path.exists(TRAINING_REPORT_PATH):
        raise FileNotFoundError(
            "Training report not found. Start the app once so model artifacts are created first."
        )

    with open(TRAINING_REPORT_PATH, "r", encoding="utf-8") as report_file:
        return json.load(report_file)


def _load_dataset():
    images = []
    labels = []

    for class_name, class_id in DATASET_LABELS.items():
        class_path = os.path.join(BASE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in sorted(os.listdir(class_path)):
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(class_id)

    if not images:
        raise ValueError("No training images found.")

    return np.array(images), np.array(labels)


def _legacy_pca_cumulative_curve():
    x, _ = _load_dataset()
    x_flat = x.reshape(len(x), -1) / 255.0
    pca = PCA(0.98)
    pca.fit(x_flat)
    return np.cumsum(pca.explained_variance_ratio_)


def generate_pca_plot(report):
    print("Generating PCA cumulative variance plot...")

    explained_variance_ratio = report.get("pca_explained_variance_ratio", [])
    if explained_variance_ratio:
        cumulative = np.cumsum(np.array(explained_variance_ratio, dtype=np.float64))
    else:
        # Backward compatibility for older training reports that do not include PCA vectors.
        cumulative = _legacy_pca_cumulative_curve()

    threshold_idx = int(np.argmax(cumulative >= 0.98))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(range(1, len(cumulative) + 1), cumulative, color="#0f766e", linewidth=2)
    ax.axhline(0.98, color="#dc2626", linestyle="--", linewidth=1.5)
    ax.axvline(threshold_idx + 1, color="#0284c7", linestyle=":", linewidth=1.5)

    ax.set_title("PCA Cumulative Variance Plot", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)

    ax.annotate(
        f"{threshold_idx + 1} components\nfor 98% variance",
        xy=(threshold_idx + 1, cumulative[threshold_idx]),
        xytext=(threshold_idx + 15, 0.86),
        arrowprops={"arrowstyle": "->", "color": "#0f766e"},
        bbox={"boxstyle": "round,pad=0.4", "fc": "#f8fafc", "ec": "#0f766e"},
    )

    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "pca_scree_plot.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def generate_performance_chart(report):
    print("Generating model performance comparison chart...")
    logistic_accuracy = report["baseline_logistic_regression"]["metrics"]["accuracy"] * 100
    svm_accuracy = report["svm"]["metrics"]["accuracy"] * 100

    models = ["Logistic Regression", "SVM (RBF)"]
    values = [logistic_accuracy, svm_accuracy]
    colors = ["#0f766e", "#0284c7"]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    bars = ax.bar(models, values, color=colors, edgecolor="#0f172a", linewidth=1.5)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.4,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "performance_comparison.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def generate_confusion_matrix_plot(report):
    print("Generating confusion matrix heatmap...")
    confusion = np.array(report["svm"]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    image = ax.imshow(confusion, cmap="Blues")
    plt.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(CLASS_ORDER)))
    ax.set_yticks(np.arange(len(CLASS_ORDER)))
    ax.set_xticklabels([label.replace("_", " ") for label in CLASS_ORDER], rotation=30, ha="right")
    ax.set_yticklabels([label.replace("_", " ") for label in CLASS_ORDER])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("SVM Confusion Matrix", fontsize=14, fontweight="bold")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", color="#0f172a", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def generate_confidence_distribution_plot(report):
    print("Generating confidence distribution plot...")
    samples = report.get("calibration_samples", [])
    confidences = [sample["confidence"] * 100 for sample in samples]
    correctness = [sample["correct"] for sample in samples]

    correct_conf = [c for c, ok in zip(confidences, correctness) if ok == 1]
    wrong_conf = [c for c, ok in zip(confidences, correctness) if ok == 0]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    if correct_conf:
        ax.hist(correct_conf, bins=10, alpha=0.7, color="#047857", label="Correct")
    if wrong_conf:
        ax.hist(wrong_conf, bins=10, alpha=0.7, color="#dc2626", label="Incorrect")

    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Number of Predictions")
    ax.set_title("Confidence Distribution by Prediction Correctness", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "confidence_distribution.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def generate_calibration_curve_plot(report):
    print("Generating confidence calibration curve...")
    samples = report.get("calibration_samples", [])
    if not samples:
        raise ValueError("Calibration samples are unavailable in training report.")

    confidences = np.array([sample["confidence"] for sample in samples])
    correctness = np.array([sample["correct"] for sample in samples])

    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(confidences, bins) - 1

    bin_centers = []
    empirical_accuracy = []

    for idx in range(len(bins) - 1):
        in_bin = bin_indices == idx
        if np.any(in_bin):
            bin_centers.append((bins[idx] + bins[idx + 1]) / 2)
            empirical_accuracy.append(np.mean(correctness[in_bin]))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot([0, 1], [0, 1], "--", color="#64748b", label="Perfect Calibration")
    ax.plot(bin_centers, empirical_accuracy, "o-", color="#0284c7", linewidth=2, label="Model Calibration")

    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Observed Accuracy")
    ax.set_title("Confidence Calibration Curve", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "calibration_curve.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(STATIC_DIR, exist_ok=True)

    print("=" * 72)
    print("Generating visualization outputs for Brain Tumor Detection project")
    print("=" * 72)

    report = _load_training_report()

    generate_pca_plot(report)
    generate_performance_chart(report)
    generate_confusion_matrix_plot(report)
    generate_confidence_distribution_plot(report)
    generate_calibration_curve_plot(report)

    print("=" * 72)
    print("Visualization generation complete.")
    print("Saved files:")
    print("- static/pca_scree_plot.png")
    print("- static/performance_comparison.png")
    print("- static/confusion_matrix.png")
    print("- static/confidence_distribution.png")
    print("- static/calibration_curve.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
