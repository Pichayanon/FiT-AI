from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("squat_tcn_model.h5")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "sequence" not in data:
            return jsonify({"error": "Missing 'sequence' key in JSON payload"}), 400

        sequence = data["sequence"]

        sequence_np = np.array(sequence)
        if sequence_np.shape != (30, 48):
            return jsonify({"error": f"Invalid input shape: expected (30, 48), got {sequence_np.shape}"}), 400

        sequence_np = sequence_np.reshape(1, 30, 48)
        prediction = model.predict(sequence_np)[0][0]

        print(f"prediction: {int(prediction > 0.5)}")
        print(f"confidence: {float(prediction)}")
              
        return jsonify({
            "prediction": int(prediction > 0.5),
            "confidence": float(prediction)
        })


    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
