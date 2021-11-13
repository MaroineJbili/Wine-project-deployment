from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
# Load model
CLASSIFIER = joblib.load("model.joblib")

@app.route("/", methods=["GET"])
def index():
    return jsonify({"msg": "Voici la documentation du model"})

@app.route("/predict", methods=["POST","GET"])
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
            return jsonify({'prediction_' + str(i): str(p) for p, i in zip(prediction, range(len(prediction)))}), 200
    return jsonify({"msg": "Error: not a JSON or no input key in your request"})


if __name__ == "__main__":
    app.run(debug=True)
