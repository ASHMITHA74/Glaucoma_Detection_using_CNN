from flask import Flask, render_template, request, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
import os


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static')

# Set the size of the input image
img_width, img_height = 256, 256

# Load the trained model
model = load_model('C:/Users/91890/Desktop/Acrima_model.h5') # load your trained model here

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = Image.open(file)
        img = img.resize((img_width, img_height))
        img = np.array(img)
        img = img / 255.0
        pred = model.predict(np.array([img]))
        result = []
        conf_scores = []
        for x, y in pred:
            if x >= 0.6:
                result.append("Glaucoma")
            else:
                result.append("Normal")
            conf_scores.append(x)
            conf_scores.append(y)
        return render_template('predict.html', prediction=result[0], file=file, confidence_scores=conf_scores)
    else:
        return jsonify({"error": "Invalid file format or file not selected"})


if __name__ == '__main__':
    app.run(debug=True)
