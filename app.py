from flask import Flask, render_template, request
from main import get_tumor_prediction
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./user_request"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template('result.html', error="Please upload an image file.")

        if not _allowed_file(file.filename):
            return render_template('result.html', error="Unsupported file type. Upload a valid image.")

        if file:
            filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filename)

            try:
                with open(filename, 'rb') as f:
                    prediction = get_tumor_prediction(f)
            except ValueError as e:
                os.remove(filename)
                return render_template('result.html', error=str(e))

            os.remove(filename)

            return render_template(
                'result.html',
                tumor_result=prediction['tumor_type'],
                confidence=round(prediction['confidence'] * 100, 2)
            )
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
