from flask import Flask, flash, request, redirect, url_for, render_template
import os
import TestingData
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/Uploaded/'

app.secret_key = 'secret key'
app.config['IMAGE_UPLOADS'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/output', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        basedir = os.path.abspath(os.path.dirname(__file__))
        file.save(os.path.join(basedir, app.config["IMAGE_UPLOADS"], filename))
        caption = TestingData.mainPredict('static/Uploaded')
        flash(caption)
        return output2(filename='result_img.png')
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


# use decorators to link the function to a url
@app.route('/')
def home():
    return render_template('index.html')  # return a string


@app.route('/about')
def about():
    return render_template('about.html')  # render a template


@app.route('/contact')
def contact():
    return render_template('contact.html')  # render a template


@app.route('/output')
def output():
    return render_template('output.html')


@app.route('/output2/<filename>')
def output2(filename):
    return render_template('output2.html', filename = filename)


# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
