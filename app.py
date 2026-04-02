import base64
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlencode

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from main import get_model_metadata, get_training_report, get_tumor_prediction


LOGGER = logging.getLogger("brain_tumor.app")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(BASE_DIR, "user_request"))
HISTORY_FILE = os.getenv("PREDICTION_HISTORY_FILE", os.path.join(BASE_DIR, "prediction_history.jsonl"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "10"))
ALLOW_CORS_ORIGIN = os.getenv("ALLOW_CORS_ORIGIN", "")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
RATE_STATE = {}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE) or BASE_DIR, exist_ok=True)


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _safe_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except OSError as error:
        LOGGER.warning("Failed to delete temporary file %s: %s", path, error)


def _new_upload_name(original_filename):
    safe_name = secure_filename(original_filename or "")
    extension = os.path.splitext(safe_name)[1].lower()
    if not extension:
        extension = ".img"
    return f"{uuid.uuid4().hex}{extension}"


def _append_history(record):
    with open(HISTORY_FILE, "a", encoding="utf-8") as history:
        history.write(json.dumps(record, ensure_ascii=True) + "\n")


def _read_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    records = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as history:
        for line in history:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _parse_float(value, default_value=None):
    if value is None:
        return default_value
    try:
        return float(value)
    except (TypeError, ValueError):
        return default_value


def _parse_int(value, default_value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default_value


def _parse_iso_datetime(value):
    if not value:
        return None

    try:
        sanitized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(sanitized)
    except ValueError:
        return None


def _decode_base64_image(encoded_data):
    if not isinstance(encoded_data, str) or not encoded_data.strip():
        raise ValueError("Image data is missing or invalid.")

    payload = encoded_data.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        return base64.b64decode(payload, validate=True)
    except Exception as error:
        raise ValueError("Invalid base64 image payload.") from error


def _client_key():
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.remote_addr or "unknown"


def _is_rate_limited(client_id):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    existing = RATE_STATE.get(client_id, [])
    active = [timestamp for timestamp in existing if timestamp >= window_start]

    if len(active) >= RATE_LIMIT_REQUESTS:
        RATE_STATE[client_id] = active
        return True

    active.append(now)
    RATE_STATE[client_id] = active
    return False


def _predict_from_bytes(image_bytes, enable_clahe=False):
    if not image_bytes:
        raise ValueError("Image payload is empty.")

    with io.BytesIO(image_bytes) as image_stream:
        return get_tumor_prediction(image_stream, enable_clahe=enable_clahe)


def _history_response(records, page, per_page):
    total = len(records)
    start = (page - 1) * per_page
    end = start + per_page
    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": (total + per_page - 1) // per_page,
        "items": records[start:end],
    }


def _request_wants_html():
    if request.args.get("format") == "json":
        return False
    if request.args.get("view") == "html":
        return True

    best = request.accept_mimetypes.best
    return best == "text/html" and request.accept_mimetypes["text/html"] >= request.accept_mimetypes["application/json"]


def _build_query_url(path, params):
    compact = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        compact[key] = value

    query = urlencode(compact)
    if not query:
        return path
    return f"{path}?{query}"


def _build_history_payload():
    records = _read_history()

    class_name = request.args.get("class")
    source = request.args.get("source")
    status = request.args.get("status")
    min_confidence = _parse_float(request.args.get("min_confidence"))
    max_confidence = _parse_float(request.args.get("max_confidence"))
    low_confidence = request.args.get("low_confidence")
    start_date = _parse_iso_datetime(request.args.get("start_date"))
    end_date = _parse_iso_datetime(request.args.get("end_date"))

    filtered = []
    for record in records:
        if class_name and record.get("tumor_type") != class_name:
            continue
        if source and record.get("source") != source:
            continue
        if status and record.get("status") != status:
            continue

        confidence = _parse_float(record.get("confidence"))
        if min_confidence is not None and (confidence is None or confidence < min_confidence):
            continue
        if max_confidence is not None and (confidence is None or confidence > max_confidence):
            continue

        if low_confidence in {"true", "false"}:
            wanted = low_confidence == "true"
            if bool(record.get("low_confidence", False)) != wanted:
                continue

        if start_date or end_date:
            record_time = _parse_iso_datetime(record.get("timestamp"))
            if record_time is None:
                continue
            if start_date and record_time < start_date:
                continue
            if end_date and record_time > end_date:
                continue

        filtered.append(record)

    filtered.sort(key=lambda item: item.get("timestamp", ""), reverse=True)

    page = max(1, _parse_int(request.args.get("page"), 1))
    per_page = max(1, min(100, _parse_int(request.args.get("per_page"), 20)))

    payload = _history_response(filtered, page=page, per_page=per_page)
    filters = {
        "class": class_name or "",
        "source": source or "",
        "status": status or "",
        "min_confidence": request.args.get("min_confidence", ""),
        "max_confidence": request.args.get("max_confidence", ""),
        "low_confidence": low_confidence or "",
        "start_date": request.args.get("start_date", ""),
        "end_date": request.args.get("end_date", ""),
    }
    return payload, filters


@app.after_request
def _apply_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline';"

    if ALLOW_CORS_ORIGIN:
        response.headers["Access-Control-Allow-Origin"] = ALLOW_CORS_ORIGIN
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Vary"] = "Origin"

    return response


@app.errorhandler(413)
def _payload_too_large(_):
    message = f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB."
    if request.path.startswith("/api/"):
        return jsonify({"error": message}), 413
    return render_template("result.html", error=message), 413


@app.before_request
def _enforce_rate_limit():
    protected_paths = {"/result", "/api/predict"}
    if request.path not in protected_paths:
        return None

    if request.method not in {"POST", "OPTIONS"}:
        return None

    client_id = _client_key()
    if _is_rate_limited(client_id):
        message = "Rate limit exceeded. Please retry after a short delay."
        if request.path.startswith("/api/"):
            return jsonify({"error": message}), 429
        return render_template("result.html", error=message), 429

    return None


@app.route("/")
def home():
    return render_template("home.html", max_batch_size=MAX_BATCH_SIZE)


@app.route("/result", methods=["POST"])
def result():
    request_start = time.perf_counter()

    files = [item for item in request.files.getlist("image") if item and item.filename]
    if not files:
        return render_template("result.html", error="Please upload an image file.")

    if len(files) > MAX_BATCH_SIZE:
        return render_template("result.html", error=f"You can upload up to {MAX_BATCH_SIZE} files at once.")

    enable_clahe = request.form.get("enhance") == "on"

    batch_results = []

    for file in files:
        filename = secure_filename(file.filename) or "uploaded_image"
        item_start = time.perf_counter()

        if not _allowed_file(file.filename):
            item_error = "Unsupported file type. Upload a valid image."
            batch_results.append(
                {
                    "filename": filename,
                    "status": "error",
                    "error": item_error,
                }
            )
            _append_history(
                {
                    "id": uuid.uuid4().hex,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "web",
                    "filename": filename,
                    "enhancement": bool(enable_clahe),
                    "status": "error",
                    "error": item_error,
                }
            )
            continue

        upload_name = _new_upload_name(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_name)
        file.save(file_path)

        prediction = None
        error_message = None

        try:
            with open(file_path, "rb") as uploaded_file:
                prediction = get_tumor_prediction(uploaded_file, enable_clahe=enable_clahe)
        except ValueError as error:
            error_message = str(error)
        finally:
            _safe_remove(file_path)

        elapsed_ms = round((time.perf_counter() - item_start) * 1000, 2)

        record = {
            "id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "web",
            "filename": filename,
            "enhancement": bool(enable_clahe),
            "status": "success" if prediction else "error",
            "latency_ms": elapsed_ms,
        }

        if prediction:
            result_item = {
                "filename": filename,
                "status": "success",
                "tumor_type": prediction["tumor_type"],
                "confidence": round(prediction["confidence"] * 100, 2),
                "low_confidence": prediction["low_confidence"],
                "threshold": prediction["low_confidence_threshold"],
                "model_version": prediction["model_version"],
                "latency_ms": elapsed_ms,
            }
            batch_results.append(result_item)
            record.update(
                {
                    "tumor_type": prediction["tumor_type"],
                    "confidence": round(float(prediction["confidence"]) * 100, 4),
                    "low_confidence": bool(prediction["low_confidence"]),
                    "model_version": prediction["model_version"],
                }
            )
        else:
            result_item = {
                "filename": filename,
                "status": "error",
                "error": error_message or "Unable to process image.",
                "latency_ms": elapsed_ms,
            }
            batch_results.append(result_item)
            record["error"] = result_item["error"]

        _append_history(record)

    total_elapsed_ms = round((time.perf_counter() - request_start) * 1000, 2)

    if len(batch_results) == 1:
        single = batch_results[0]
        if single.get("status") != "success":
            return render_template("result.html", error=single.get("error", "Unable to process image."))

        return render_template(
            "result.html",
            tumor_result=single["tumor_type"],
            confidence=single["confidence"],
            low_confidence=single["low_confidence"],
            threshold=single["threshold"],
            model_version=single["model_version"],
            latency_ms=single["latency_ms"],
        )

    success_count = sum(1 for item in batch_results if item.get("status") == "success")
    first_success = next((item for item in batch_results if item.get("status") == "success"), None)
    model_version = first_success.get("model_version") if first_success else get_model_metadata().get("model_version")
    threshold = first_success.get("threshold") if first_success else get_model_metadata().get("low_confidence_threshold")

    return render_template(
        "result.html",
        batch_results=batch_results,
        success_count=success_count,
        total_count=len(batch_results),
        threshold=threshold,
        model_version=model_version,
        latency_ms=total_elapsed_ms,
    )


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def api_predict():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    enable_clahe = bool(payload.get("enable_clahe", False))
    items = []

    if isinstance(payload.get("images"), list):
        items = payload["images"]
    elif "image" in payload:
        items = [{"filename": payload.get("filename", "image_1"), "data": payload.get("image")}]

    if not items:
        return jsonify({"error": "Provide either 'image' or 'images' payload."}), 400

    if len(items) > MAX_BATCH_SIZE:
        return jsonify({"error": f"Batch size exceeds limit ({MAX_BATCH_SIZE})."}), 400

    request_start = time.perf_counter()
    results = []

    for index, item in enumerate(items):
        filename = secure_filename(str(item.get("filename", f"image_{index + 1}"))) or f"image_{index + 1}"
        encoded_data = item.get("data")

        try:
            image_bytes = _decode_base64_image(encoded_data)
            prediction = _predict_from_bytes(image_bytes, enable_clahe=enable_clahe)

            result_item = {
                "filename": filename,
                "status": "success",
                "tumor_type": prediction["tumor_type"],
                "confidence": round(prediction["confidence"] * 100, 2),
                "low_confidence": prediction["low_confidence"],
                "threshold": prediction["low_confidence_threshold"],
                "model_version": prediction["model_version"],
                "inference_ms": prediction["inference_ms"],
            }

            results.append(result_item)

            _append_history(
                {
                    "id": uuid.uuid4().hex,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "api",
                    "filename": filename,
                    "enhancement": bool(enable_clahe),
                    "status": "success",
                    "tumor_type": prediction["tumor_type"],
                    "confidence": round(float(prediction["confidence"]) * 100, 4),
                    "low_confidence": bool(prediction["low_confidence"]),
                    "model_version": prediction["model_version"],
                    "latency_ms": prediction["inference_ms"],
                }
            )
        except ValueError as error:
            message = str(error)
            results.append(
                {
                    "filename": filename,
                    "status": "error",
                    "error": message,
                }
            )
            _append_history(
                {
                    "id": uuid.uuid4().hex,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "api",
                    "filename": filename,
                    "enhancement": bool(enable_clahe),
                    "status": "error",
                    "error": message,
                }
            )

    elapsed_ms = round((time.perf_counter() - request_start) * 1000, 2)

    return jsonify(
        {
            "model": get_model_metadata(),
            "count": len(results),
            "success_count": sum(1 for item in results if item.get("status") == "success"),
            "elapsed_ms": elapsed_ms,
            "predictions": results,
        }
    )


@app.route("/api/history", methods=["GET"])
def api_history():
    payload, filters = _build_history_payload()

    if _request_wants_html():
        labels = list(get_model_metadata().get("labels", {}).keys())
        classes = labels if labels else sorted({item.get("tumor_type") for item in payload["items"] if item.get("tumor_type")})

        query_base = {
            "class": filters["class"],
            "source": filters["source"],
            "status": filters["status"],
            "min_confidence": filters["min_confidence"],
            "max_confidence": filters["max_confidence"],
            "low_confidence": filters["low_confidence"],
            "start_date": filters["start_date"],
            "end_date": filters["end_date"],
            "per_page": payload["per_page"],
            "view": "html",
        }

        prev_url = None
        next_url = None
        if payload["page"] > 1:
            prev_url = _build_query_url("/api/history", {**query_base, "page": payload["page"] - 1})
        if payload["page"] < payload["pages"]:
            next_url = _build_query_url("/api/history", {**query_base, "page": payload["page"] + 1})

        return render_template(
            "api_history.html",
            data=payload,
            filters=filters,
            class_options=classes,
            prev_url=prev_url,
            next_url=next_url,
        )

    return jsonify(payload)


@app.route("/history", methods=["GET"])
def history_alias():
    return api_history()


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    report = get_training_report() or {}

    if _request_wants_html():
        labels = list(get_model_metadata().get("labels", {}).keys())
        static_assets = [
            {"name": "PCA Scree Plot", "path": "pca_scree_plot.png"},
            {"name": "Model Comparison", "path": "performance_comparison.png"},
            {"name": "Confusion Matrix", "path": "confusion_matrix.png"},
            {"name": "Confidence Distribution", "path": "confidence_distribution.png"},
            {"name": "Calibration Curve", "path": "calibration_curve.png"},
        ]
        return render_template("api_metrics.html", report=report, class_labels=labels, static_assets=static_assets)

    return jsonify(report)


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1")
