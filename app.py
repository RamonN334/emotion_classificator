import base64
import io
from copy import deepcopy
from PIL import Image
from flask import Flask, make_response, request, json

import emotion_gender_processor as eg_processor

app = Flask(__name__)

DEBUG = True


@app.route("/emotion_classificator/1.0", methods=["POST"])
def classify():
    if request.method == "POST" and "image" in request.json:
        if DEBUG:
            img = Image.open("test.jpeg", mode="r")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            prediction_result = eg_processor.process_image(img_byte_arr)
        else:
            img_b64 = request.json["image"]
            img_body = getImgBody(img_b64)
            prediction_result = eg_processor.process_image(img_body)

        print(prediction_result)

    return json.jsonify({"JSON": getJSONResponse(prediction_result)})


def getImgBody(img_b64):
    if "data:image" in img_b64:
        return img_b64.split(",")[1]


def getJSONResponse(result):
    json = [
        {
            "success": True,
            "data": []
        }
    ]
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
        json[0]["data"].append(deepcopy(d))

    return json

if __name__ == "__main__":
    app.run(port="8080", debug=True)
