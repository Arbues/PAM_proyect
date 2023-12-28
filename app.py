from flask import Flask, render_template, request
from keras.models import load_model
import joblib

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

modelo_MLG = joblib.load('clf_entrenado_MLG.pkl')
modelo_tSFL1 = load_model('clf_entrenado_tSFL1.h5')
modelo_tSFL2 = load_model('clf_entrenado_tSFL2.h5')

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        img_file = request.files["image"]
        if img_file:
            img = Image.open(img_file).convert('L')
            img = img.resize((28, 28))
            img_data = np.array(img) / 255

            img_vector = 1 - img_data.flatten()

            plt.imshow(img_vector.reshape(28, 28), cmap='gray')
            plt.title("Imagen Ingresada")
            plt.show()

            model_choice = request.form['model']
            if model_choice == 'MLG':
                prediction = modelo_MLG.predict([img_vector])[0]
            elif model_choice == 'tSFL1':
                prediction = modelo_tSFL1.predict(img_vector.reshape(1, -1))[0]
                prediction = np.argmax(prediction)
            elif model_choice == 'tSFL2':
                prediction = modelo_tSFL2.predict(img_vector.reshape(1, -1))[0]
                prediction = np.argmax(prediction)

            

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
