from flask import Flask, render_template, request, \
    send_from_directory, redirect, url_for, session
from flask_session import Session

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os

from src.align import align_dataset_mtcnn as mtcnn
from src import classifier
from src import compare
from celery import Celery

app = Flask(__name__)
app.secret_key                   = 'super secret key'
app.config["SESSION_PERMANENT"]  = False
app.config["SESSION_TYPE"]       = "filesystem"
app.config['UPLOAD_FOLDER']      = './static/uploads/'

pretrained_models = "models/20180408-102900.pb"
classifier_models = "models/my_classifier.pkl"

train_raw_folder     = "data/images/train_raw"
train_aligned_folder = "data/images/train_aligned"

test_raw_folder      = "data/images/test_raw"
test_aligned_folder  = "data/images/test_aligned"


def predict_label(img_path):
    loaded_img = load_img(img_path, target_size=(256, 256))
    img_array = img_to_array(loaded_img) / 255.0
    img_array = expand_dims(img_array, 0)
    predicted_bit = np.round(model.predict(img_array)[0][0]).astype('int')
    return class_dict[predicted_bit]

@app.route('/compare', methods=['GET', 'POST'])
def compare_route():
    argv = []
    argv.append(pretrained_models)
    argv.append(classifier_models)
    argv.append(session.get('img_path'))
    result = compare.main(compare.parse_arguments(argv))

    session['detected']   = result['detected']
    session['person']     = result['person']
    session['confidence'] = result['confidence']

    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image    = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)

            session['img_path']  = img_path
            session['img']       = image.filename

            return render_template('loading.html', url = url_for('compare_route'))

    if not 'detected' in session:
        return render_template('index.html')
    else:
        return render_template('index.html', session = session)

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/align_train', methods=['POST', 'GET'])
def align_train_route():
    argv = []
    argv.append(train_raw_folder)
    argv.append(train_aligned_folder)
    argv.append("--image_size=160")
    argv.append("--type_data=True")

    mtcnn.main(mtcnn.parse_arguments(argv))

    # mtcnn.main(args)
    # img = request.files['img']
    # cred = g.cred
    # img.save("upload/train.jpg")
    # datas = trainImage(f"{cred.user_id}", f"{cred.nip}")
    # return jsonify(message=str(datas)+" Wajahh Berhasil di Registrasi"), 200
    return render_template('loading.html', url = url_for('index'))

@app.route('/align_test', methods=['POST', 'GET'])
def align_test_route():
    argv = []
    argv.append(test_raw_folder)
    argv.append(test_aligned_folder)
    argv.append("--image_size=160")

    mtcnn.main(mtcnn.parse_arguments(argv))

    return render_template('loading.html', url = url_for('index'))

@app.route('/train', methods=['POST', 'GET'])
def train_route():
    argv = []
    argv.append("TRAIN")
    argv.append(train_aligned_folder)
    argv.append(pretrained_models)
    argv.append(classifier_models)

    classifier.main(classifier.parse_arguments(argv))

    return render_template('loading.html', url = url_for('index'))

@app.route('/classify', methods=['POST', 'GET'])
def classify_route():
    argv = []
    argv.append("CLASSIFY")
    argv.append(test_aligned_folder)
    argv.append(pretrained_models)
    argv.append(classifier_models)

    classifier.main(classifier.parse_arguments(argv))

    return render_template('loading.html', url = url_for('index'))

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))