import os
import joblib
import numpy as np

from flask import Flask, render_template, request


app = Flask(__name__)

MODEL_PATH = "model/iris_knn_model.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Please run 'python train_model.py' first."
        )

    model_data = joblib.load(MODEL_PATH)
    return model_data


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    input_values = {
        "sepal_length": "",
        "sepal_width": "",
        "petal_length": "",
        "petal_width": ""
    }

    try:
        model_data = load_model()
        model = model_data["model"]
        target_names = model_data["target_names"]
        accuracy = model_data["accuracy"]

    except Exception as e:
        model = None
        target_names = None
        accuracy = None
        error = str(e)

    if request.method == "POST" and model is not None:
        try:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            input_values = {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }

            input_data = np.array([[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]])

            predicted_class = model.predict(input_data)[0]
            prediction = target_names[predicted_class].capitalize()

        except ValueError:
            error = "Please enter valid numeric values."
        except Exception as e:
            error = f"Prediction error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        input_values=input_values,
        accuracy=accuracy
    )


if __name__ == "__main__":
    app.run(debug=True)