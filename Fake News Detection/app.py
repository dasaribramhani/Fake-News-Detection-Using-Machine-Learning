from flask import Flask, render_template, request
from waitress import serve
import joblib

app = Flask(__name__)

model = joblib.load("model/fake_news_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text", "").strip()

    if not news_text:
        result = "Please enter some news text."
    else:
        prediction = model.predict([news_text])[0]

        print("----------------------------------------")
        print("DEBUG - Raw Prediction Value:", prediction)
        print("----------------------------------------")

        # 0 = REAL NEWS, 1 = FAKE NEWS
        if int(prediction) == 0:
            result = "REAL NEWS"
        else:
            result = "FAKE NEWS"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    print("Server running on http://localhost:5000")
    serve(app, host='0.0.0.0', port=5000)