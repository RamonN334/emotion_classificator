from flask import Flask

app = Flask(__name__)


@app.route("/emotion_classificator/1.0")
def classify():
    return "hello"

if __name__ == "__ main__":
    app.run(debug=True)
