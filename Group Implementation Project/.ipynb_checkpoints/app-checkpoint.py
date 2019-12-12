from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import numpy as np
import cv2
import base64
from keras.models import load_model

app = Flask(__name__)


@app.route('/')
def index():
    loadModel()
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
        img = cv2.resize(img, (0,0), fx = 0.1, fy = 0.1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape
        print(height)
        print(width)
        predict_img(img)
    return render_template('index.html')

def predict_img(img):
    height, width = img.shape
    num_pixels = width * height
    img = img.reshape(img)
    print(img.shape)
    return

def loadModel():
    model = load_model('mnist_simple_cnn.h5')
    return

if __name__ == '__main__':
    app.run(debug=True, port=8000)
