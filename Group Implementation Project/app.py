from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import numpy as np
import cv2
import base64
from keras.models import load_model
import tensorflow as tf
import keras
from keras.models import model_from_json

app = Flask(__name__)

global model
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    proba = 0
    if request.is_json:
        jsondata = request.json.get('data')
        img = base64.b64decode(jsondata)
        filename = 'number.jpg'
        with open(filename, 'wb') as f:
            f.write(img)
        img = cv2.imread('number.jpg')
        img = cv2.resize(img, (0,0), fx = 0.1, fy = 0.1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape
        img = img.flatten().astype('float32')
        img /= 255
        keras.backend.clear_session()
        model = loadModel()
        proba = predict_img(img, model)
    return render_template('index.html')

def predict_img(img, model):
    graph = tf.get_default_graph()
    with graph.as_default():
        proba = model.predict(np.array( [img,] ))
    y_classes = model.predict_classes(np.array( [img,] ))
    print(proba[0])
    return proba[0][0]

def loadModel():
    model = load_model('mnist_simple_ann.h5')
    return model

if __name__ == '__main__':
    app.run(debug=True, port=8000)









    # y_classes = model.predict_classes(np.array( [img,] ))
    # y_predicted = text_labels[np.argmax(y)]
    # y_classes = proba.argmax(axis=-1)
    # proba = proba[0]
    #
    # print(y_classes[0])
    # predict_result = {}
    # for i in range(len(proba)):
    #     predict_result[y_classes[i]] = proba[i]
    # for i in range(len(proba)):
    #     predict_result[i] = proba[i]
        # if i == 0:
        #     predict_result[len(proba)-1] = proba[i]
        # else: predict_result[i+1] = proba[i]
    # {k: v for k, v in sorted(predict_result.items(), key=lambda item: item[1])}
    # print(predict_result)
    # for key in predict_result:
    #     print("Number predicted: " + str(key))
    #     print("Confidence: " + str(predict_result[key]))
