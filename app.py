"""
Portfolio backend — serves the portfolio site (index.html) and powers its
"live project" section: a real scikit-learn model trained at startup and
served through a small API, so a visitor can interact with an actual
working ML system rather than a screenshot of one.

Routes:
    GET  /                -> index.html
    POST /api/predict     -> {"species": str, "confidence": float, "probabilities": {...}}
    GET  /api/health       -> {"status": "ok", "model": "..."}
    GET  /api/meta         -> species list, slider ranges, dataset points, test accuracy
"""

import os

from flask import Flask, jsonify, request, send_from_directory
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

# ---------------------------------------------------------------------------
# Train the model once, at process startup. Iris is tiny (150 rows) so this
# takes a fraction of a second and needs no persisted model file.
# ---------------------------------------------------------------------------
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

test_accuracy = round(accuracy_score(y_test, model.predict(X_test)) * 100, 1)
SPECIES = [name.replace("_", " ").title() for name in iris.target_names]

FEATURE_RANGES = {
    "sepal_length": {"min": 4.0, "max": 8.0, "default": 5.8},
    "sepal_width": {"min": 2.0, "max": 4.5, "default": 3.0},
    "petal_length": {"min": 1.0, "max": 7.0, "default": 4.3},
    "petal_width": {"min": 0.1, "max": 2.6, "default": 1.3},
}


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify(status="ok", model="RandomForestClassifier", test_accuracy=test_accuracy)


@app.route("/api/meta")
def meta():
    """Lets the front end draw sliders/scatter bounds without hardcoding them twice."""
    return jsonify(
        species=SPECIES,
        feature_ranges=FEATURE_RANGES,
        test_accuracy=test_accuracy,
        n_train=len(X_train),
        n_test=len(X_test),
        dataset_points=[
            {
                "sepal_length": float(row[0]),
                "sepal_width": float(row[1]),
                "petal_length": float(row[2]),
                "petal_width": float(row[3]),
                "species": SPECIES[int(label)],
            }
            for row, label in zip(iris.data, iris.target)
        ],
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    try:
        features = [
            float(payload["sepal_length"]),
            float(payload["sepal_width"]),
            float(payload["petal_length"]),
            float(payload["petal_width"]),
        ]
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Send sepal_length, sepal_width, petal_length, petal_width as numbers."), 400

    probabilities = model.predict_proba([features])[0]
    best_index = int(probabilities.argmax())

    return jsonify(
        species=SPECIES[best_index],
        confidence=round(float(probabilities[best_index]) * 100, 1),
        probabilities={
            SPECIES[i]: round(float(p) * 100, 1) for i, p in enumerate(probabilities)
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
