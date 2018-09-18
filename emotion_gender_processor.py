import os
import sys
import logging
import copy
from datetime import datetime

import cv2
import tensorflow as tf
from keras.models import load_model
from keras import backend
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

backend.clear_session()
_SAVE_DIR = 'static/result'
_DETECTION_MODEL_PATH = './trained_models/detection_models/haarcascade_frontalface_default.xml'
_EMOTION_MODEL_PATH = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
_FACE_DETECTION = load_detection_model(_DETECTION_MODEL_PATH)
_EMOTION_CLASSIFIER = load_model(_EMOTION_MODEL_PATH, compile=False)
_GRAPH = tf.get_default_graph()


def process_image(image):
    # backend.clear_session()
    # session = backend.get_session()
    detected_peoples = []
    json_info = {}
    try:
        # parameters for loading data and images
        # gender_model_path = './trained_models/gender_models/simple_CNN.81-0.96.hdf5'
        emotion_labels = get_labels('fer2013')
        # gender_labels = get_labels('imdb')
        # font = cv2.FONT_HERSHEY_SIMPLEX

        # gender_keys = list(gender_labels.values())
        emotion_keys = list(emotion_labels.values())

        # print(gender_keys)
        # print(emotion_keys)

        # print(json_info)

        # hyper-parameters for bounding boxes shape
        # gender_offsets = (30, 60)
        # gender_offsets = (10, 10)
        emotion_offsets = (20, 40)
        emotion_offsets = (0, 0)

        # loading models
        # gender_classifier = load_model(gender_model_path, compile=False)

        # getting input model shapes for inference
        emotion_target_size = _EMOTION_CLASSIFIER.input_shape[1:3]
        # gender_target_size = gender_classifier.input_shape[1:3]

        # loading images
        image_array = np.fromstring(image, np.uint8)
        unchanged_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

        rgb_image = cv2.cvtColor(unchanged_image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(unchanged_image, cv2.COLOR_BGR2GRAY)

        faces = detect_faces(_FACE_DETECTION, gray_image)
        for face_coordinates in faces:
            # x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
            # rgb_face = rgb_image[y1:y2, x1:x2]

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                # rgb_face = cv2.resize(rgb_face, (gender_target_size))
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            # rgb_face = preprocess_input(rgb_face, False)
            # rgb_face = np.expand_dims(rgb_face, 0)
            # gender_prediction = gender_classifier.predict(rgb_face)
            # gender_label_arg = np.argmax(gender_prediction)
            # gender_text = gender_labels[gender_label_arg]

            start_time = datetime.now()
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            with _GRAPH.as_default():
                emotion_prediction = _EMOTION_CLASSIFIER.predict(gray_face)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]

            # print(gender_label_arg)
            # print(emotion_label_arg)
            # json_info['gender'] = dict(zip(gender_keys, gender_prediction
            #                                             .astype(np.float16)
            #                                             .flat))
            json_info['emotion'] = dict(zip(emotion_keys,
                                            emotion_prediction
                                            .astype(np.float16)
                                            .flat))
            json_info['face_bound'] = list(face_coordinates
                                           .astype(np.int)
                                           .flat)
            # print(json_info['face_bound'])
            # json_info['result'] = {
            #     # 'gender': gender_text,
            #     'emotion': emotion_text
            # }

            # for key, value in json_info["gender"].items():
            #     json_info["gender"][key] = str(value)

            for key, value in json_info["emotion"].items():
                json_info["emotion"][key] = str(value)

            for i in range(len(json_info['face_bound'])):
                json_info['face_bound'][i] = str(json_info['face_bound'][i])
            detected_peoples.append(copy.deepcopy(json_info))

            delta = datetime.now() - start_time
            print("Delta in eg_processor {0}"
                  .format(delta.total_seconds() * 1000.0))
            # print(face_coordinates)

            # if gender_text == gender_labels[0]:
            #     color = (0, 0, 255)
            # else:
            #     color = (255, 0, 0)
            color = (0, 0, 255)

            draw_bounding_box(face_coordinates, rgb_image, color)
            # draw_text(face_coordinates, rgb_image,
            #           gender_text, color, 0, -20, 1, 2)
            draw_text(face_coordinates, rgb_image,
                      emotion_text, color, 0, -10, 1, 2)
    except Exception as err:
        logging.error('Error in emotion gender processor: "{0}"'.format(err))

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    if not os.path.exists(_SAVE_DIR):
        os.mkdir(_SAVE_DIR)

    recognition_datetime = str(datetime.now()).replace(' ', '_')
    filepath = os.path.join(_SAVE_DIR, 'predicted_image_{0}.png'
                                       .format(recognition_datetime))
    cv2.imwrite(filepath, bgr_image)

    return detected_peoples
