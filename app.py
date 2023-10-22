from flask import Flask, render_template, request
from predict import make_prediction
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_file = request.files["image"]
        img = Image.open(img_file).convert('L') # Convertir a escala de grises
        img = img.resize((28, 28))
        img_data = np.array(img).reshape(1, 784).T
        prediction = make_prediction(img_data/255) # Normalizar los datos
        return render_template("index.html", prediction=prediction)
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
