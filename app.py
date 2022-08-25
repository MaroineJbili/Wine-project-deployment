from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
# Load model
CLASSIFIER = joblib.load("model.joblib")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
    # return jsonify({"msg": "Please find here the documentation of the project"})

@app.route("/predict", methods=["POST", "GET"])
def predict():
    # Check if request has a JSON content
    if request.json:
        # Get the JSON as dictionnary
        req = request.get_json()
        # Check mandatory key
        if "input" in req.keys():
            # Predict
            prediction = CLASSIFIER.predict(req["input"])
            # Return the result as JSON but first we need to transform the
            # result so as to be serializable by jsonify()
            # prediction = str(prediction)

            return jsonify({"prediction_" + str(i) : str(p) for p, i in zip(prediction, range(len(prediction)))}), 200
    return jsonify({"msg": "Error: not a JSON or no email key in your request"})


# @app.route("/predict", methods=["GET"])
# def prediction():
#     return jsonify({"prediction_" + str(i) : str(p) for p, i in zip(prediction, range(len(prediction)))})


if __name__ == "__main__":
    app.run(debug=True)
