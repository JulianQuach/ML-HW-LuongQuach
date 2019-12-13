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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.is_json:
        jsondata = request.json.get('data')
        img = base64.b64decode(jsondata)
        filename = 'number.jpg'
        with open(filename, 'wb') as f:
            f.write(img)
        img = cv2.imread('number.jpg')
        img = cv2.resize(255 - img, (28,28), interpolation=cv2.INTER_AREA)
        cv2.imwrite("numberafterprocessing.png", img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape
        img = img.flatten().astype('float32')
        img /= 255
        keras.backend.clear_session()
        model = loadModel()
        proba, output = predict_img(img, model)
    return render_template('index.html')

def predict_img(img, model):
    proba = model.predict(np.array( [img,] ))
    output = proba.argmax()
    print(str(np.amax(proba[0])*100) + " %")
    print(output)
    return str(np.amax(proba[0])*100) + " %", str(output)

def loadModel():
    model = load_model('mnist_simple_ann.h5')
    return model

if __name__ == '__main__':
    app.run(debug=True, port=8000)
