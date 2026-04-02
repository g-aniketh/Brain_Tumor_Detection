import base64
import io
import os
import unittest

from app import MAX_BATCH_SIZE, app
from werkzeug.datastructures import MultiDict


class BrainTumorAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.sample_image_path = cls._find_sample_image()

    @staticmethod
    def _find_sample_image():
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folders = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]

        for folder in folders:
            folder_path = os.path.join(base, folder)
            if not os.path.isdir(folder_path):
                continue
            for name in os.listdir(folder_path):
                path = os.path.join(folder_path, name)
                if os.path.isfile(path):
                    return path
        raise RuntimeError("No sample image found for tests")

    def test_home_page_is_available(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_result_route_rejects_invalid_extension(self):
        payload = {"image": (io.BytesIO(b"not-an-image"), "invalid.txt")}
        response = self.client.post("/result", data=payload, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Unsupported file type", response.data)

    def test_api_predict_single_success(self):
        with open(self.sample_image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")

        response = self.client.post(
            "/api/predict",
            json={"image": encoded, "filename": os.path.basename(self.sample_image_path)},
        )
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("predictions", data)
        self.assertGreaterEqual(data.get("success_count", 0), 1)

    def test_api_predict_batch_limit_enforced(self):
        with open(self.sample_image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")

        images = [{"filename": f"image_{i}.jpg", "data": encoded} for i in range(MAX_BATCH_SIZE + 1)]
        response = self.client.post("/api/predict", json={"images": images})
        self.assertEqual(response.status_code, 400)

    def test_api_history_returns_paginated_payload(self):
        response = self.client.get("/api/history?per_page=2&page=1")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("items", data)
        self.assertIn("total", data)

    def test_api_metrics_contains_svm_block(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("svm", data)

    def test_result_route_accepts_multiple_images(self):
        with open(self.sample_image_path, "rb") as image_file:
            image_bytes = image_file.read()

        payload = MultiDict(
            [
                ("image", (io.BytesIO(image_bytes), "first.jpg")),
                ("image", (io.BytesIO(image_bytes), "second.jpg")),
            ]
        )
        response = self.client.post("/result", data=payload, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Batch processed", response.data)
        self.assertIn(b"first.jpg", response.data)
        self.assertIn(b"second.jpg", response.data)


if __name__ == "__main__":
    unittest.main()
