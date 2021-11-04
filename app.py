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

app = Flask(__name__)
app.secret_key                  = 'super secret key'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"]      = "filesystem"
app.config['UPLOAD_FOLDER']     = './static/uploads/'

def predict_label(img_path):
    loaded_img = load_img(img_path, target_size=(256, 256))
    img_array = img_to_array(loaded_img) / 255.0
    img_array = expand_dims(img_array, 0)
    predicted_bit = np.round(model.predict(img_array)[0][0]).astype('int')
    return class_dict[predicted_bit]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image    = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            session['img_path'] = img_path
            print("Request Berjalan")

            return redirect(url_for('loading', operation = 'comparing'))
    return render_template('index.html')

@app.route('/compare', methods=['GET', 'POST'])
def compare_route():
    argv = []
    argv.append("models/20180408-102900.pb")
    argv.append("models/my_classifier.pkl")
    argv.append(session.get('img_path'))
    compare.main(compare.parse_arguments(argv))
    return render_template('index.html')

@app.route('/loading', methods=['GET'])
def loading():
    operation = request.args.get("operation")
    if(operation == "comparing"):
        url = "/compare"
        return render_template('loading.html', url = url)
    else:
        url = "/"
        return render_template('loading.html', url = url)

@app.route('/test', methods=['GET', 'POST'])
def test():
    print("test")

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/align_train', methods=['POST', 'GET'])
def align_train_route():
    argv = []
    argv.append("data/images/train_raw")
    argv.append("data/images/train_aligned")
    argv.append("--image_size=160")
    argv.append("--type_data=True")

    mtcnn.main(mtcnn.parse_arguments(argv))

    # mtcnn.main(args)
    # img = request.files['img']
    # cred = g.cred
    # img.save("upload/train.jpg")
    # datas = trainImage(f"{cred.user_id}", f"{cred.nip}")
    # return jsonify(message=str(datas)+" Wajahh Berhasil di Registrasi"), 200
    return "mntp"

@app.route('/align_test', methods=['POST', 'GET'])
def align_test_route():
    argv = []
    argv.append("data/images/test_raw")
    argv.append("data/images/test_aligned")
    argv.append("--image_size=160")

    mtcnn.main(mtcnn.parse_arguments(argv))
    return "mntp"

@app.route('/train', methods=['POST', 'GET'])
def train_route():
    argv = []
    argv.append("TRAIN")
    argv.append("data/images/train_aligned")
    argv.append("models/20180408-102900.pb")
    argv.append("models/my_classifier.pkl")

    classifier.main(classifier.parse_arguments(argv))
    return "mntp"

@app.route('/classify', methods=['POST', 'GET'])
def classify_route():
    argv = []
    argv.append("CLASSIFY")
    argv.append("data/images/test_aligned")
    argv.append("models/20180408-102900.pb")
    argv.append("models/my_classifier.pkl")

    classifier.main(classifier.parse_arguments(argv))
    return "mntp"

if __name__ == '__main__':
    app.run(debug=True)