from flask import Flask, render_template, request
from main import get_tumor_type
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./user_request"

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filename)

            with open(filename, 'rb') as f:
                tumor_result = get_tumor_type(f)

            os.remove(filename)

            return render_template('result.html', tumor_result=tumor_result)
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
