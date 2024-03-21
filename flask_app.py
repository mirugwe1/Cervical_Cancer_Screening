from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import keras
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = keras.models.load_model('./models/cervical_cancer_image_classifier.h5')

print('Model loaded. Check http://127.0.0.1:5000/')
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        class_idx = np.argmax(preds, axis=1)[0]
        class_labels = ['Negative', 'Positive', 'Suspected']
        class_label = class_labels[class_idx]

        return class_label

    return None


if __name__ == '__main__':
    app.run(debug=True)
