import base64
import io
import uuid
from datetime import datetime, timedelta
from copy import deepcopy
from PIL import Image
from flask import Flask, make_response, request, json

import emotion_gender_processor as eg_processor

app = Flask(__name__)

DEBUG = False
JSON_TITLE = "JSON"


@app.route("/emotion_classificator/1.0", methods=["POST"])
def classify():
    if (request.method == "POST" and
        "image" in request.json and
            "minAccuracy" in request.json):
        if DEBUG:
            img = Image.open("test.jpeg", mode="r")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()

            start_time = datetime.now()

            prediction_result = eg_processor.process_image(img_byte_arr)

            end_time = datetime.now()
            delta = end_time - start_time
        else:
            img_b64 = request.json["image"]
            minAccuracy = request.json["minAccuracy"]
            img_body = getImgBody(img_b64)
            if (img_body is None):
                return json.jsonify(getJSONResponse(msg="Image not found"))

            start_time = datetime.now()

            prediction_result = eg_processor.process_image(img_body)

            delta = datetime.now() - start_time

        print(delta.total_seconds() * 1000.0)
        print(prediction_result)
        if (prediction_result == []):
            return json.jsonify(getJSONResponse())
    else:
        return json.jsonify(getJSONResponse(msg="Invalid request"))

    return json.jsonify(getJSONResponse(prediction_result))


def getImgBody(img_b64):
    if "data:image" in img_b64:
        img_encoded = img_b64.split(",")[1]
        return base64.decodebytes(img_encoded.encode("utf-8"))
    else:
        return None


def getJSONResponse(result=None, msg=None):
    json = {
        "success": False
    }

    if msg is not None:
        json["message"] = msg
        return json

    json["data"] = []

    if result is None:
        return json

    for item in result:
        d = {
            "faceRectangle": {
                "left": item["face_bound"][0],
                "top": item["face_bound"][1],
                "width": item["face_bound"][2],
                "heigth": item["face_bound"][3]
            },
            "emotion": item["emotion"]
        }
        json["data"].append(deepcopy(d))

    json["success"] = True
    return json

if __name__ == "__main__":
    # eg_processor.load_models()
    app.run(port="8080", debug=True)
