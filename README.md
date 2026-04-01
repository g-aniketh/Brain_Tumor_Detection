# Brain Tumor Detection

A full-stack classical machine learning project for brain MRI classification.

The application predicts one of four classes:

- no_tumor
- glioma_tumor
- meningioma_tumor
- pituitary_tumor

Implemented capabilities:

- Flask web interface for MRI upload
- PCA + SVM classification pipeline
- Logistic Regression baseline comparison
- Confidence score and low-confidence triage flag
- Secure file upload handling and cleanup
- JSON API (single + batch inference)
- File-based prediction history with filtering
- Model persistence with automatic cache loading
- Metrics and explainability visualization outputs

## Project Structure

- app.py: Flask app, web routes, API routes, history, security headers
- main.py: dataset loading, training, model persistence, inference
- templates/: web pages
- static/: styles and generated plots
- artifacts/: saved model bundle and training report
- tests/: automated regression tests

## Core Pipeline

1. Load labeled MRI images from class folders.
2. Convert to grayscale and resize to 200x200.
3. Flatten and normalize pixel values.
4. Reduce dimensionality with PCA (98% variance).
5. Train Logistic Regression baseline and SVM classifier.
6. Tune SVM with GridSearchCV + Stratified K-Fold CV.
7. Persist trained PCA/SVM bundle and metadata.
8. Serve browser and API inference using cached models.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate.fish
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open:

- Web UI: http://127.0.0.1:5000

## Environment Variables

- FLASK_DEBUG: set 1 for debug mode (default: 0)
- FORCE_RETRAIN: set 1 to retrain model and overwrite artifacts
- ENABLE_CLAHE: set 1 to enable CLAHE by default
- LOW_CONFIDENCE_THRESHOLD: percentage threshold for referral flag (default: 70)
- MAX_UPLOAD_MB: max upload size in MB (default: 10)
- MAX_BATCH_SIZE: API batch limit (default: 10)
- RATE_LIMIT_REQUESTS: max protected requests per window (default: 30)
- RATE_LIMIT_WINDOW_SECONDS: rate-limit window size (default: 60)
- ALLOW_CORS_ORIGIN: optional CORS origin value
- PREDICTION_HISTORY_FILE: path to JSONL history file

## API Endpoints

### POST /api/predict

Single image payload:

```json
{
  "image": "<base64_image>",
  "filename": "scan.jpg",
  "enable_clahe": false
}
```

Batch payload:

```json
{
  "images": [
    { "filename": "scan1.jpg", "data": "<base64_image_1>" },
    { "filename": "scan2.jpg", "data": "<base64_image_2>" }
  ],
  "enable_clahe": true
}
```

### GET /api/history

Query params:

- page, per_page
- class
- source (web or api)
- status (success or error)
- min_confidence, max_confidence
- low_confidence (true or false)
- start_date, end_date (ISO format)

Alias: GET /history

### GET /api/metrics

Returns the training report with:

- dataset stats
- PCA components and explained variance
- Logistic and SVM metrics
- confusion matrix
- CV best score
- calibration samples

## Visualization Generation

After at least one training run:

```bash
python generate_visualizations.py
```

Generated files:

- static/pca_scree_plot.png
- static/performance_comparison.png
- static/confusion_matrix.png
- static/confidence_distribution.png
- static/calibration_curve.png

## Automated Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Current Example Metrics (latest run)

From artifacts/training_report.json:

- Logistic Regression accuracy: 84.17%
- SVM accuracy: 88.33%
- PCA components retained: 214

## Disclaimer

This tool is for educational and research support. It is not a standalone clinical diagnostic system. Final diagnosis must be made by qualified medical professionals.
