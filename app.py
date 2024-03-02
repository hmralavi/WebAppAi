from flask import Flask, render_template, send_from_directory
from flask import request as frequest
import requests
from dotenv import load_dotenv
import os
import io
from PIL import Image
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from docx import Document
import numpy as np
from tensorflow.keras.models import load_model
 
app = Flask(__name__)
 
app.config['DUMP'] = os.path.join('static', 'dump')  # path to the folder where the data created by the app are stored
app.config['UPLOAD']  = os.path.join('static', 'uploads') # path to the folder where the data uploaded by user are stored

load_dotenv() # load environmental variables

apikey = os.environ['HF_API_KEY'] # load huggingface api key from thr environmental variables
 
# Home page
@app.route('/')
@app.route('/index.html')
def homepage():
    return render_template('index.html')

# Text2image page
@app.route('/text2image.html', methods=['GET', 'POST'])
def text2image():
    if frequest.method == 'GET':
        return render_template("text2image.html")
    usertext = frequest.form['usertext']
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0" # the model used for text2image generation. this model can be changed.
    headers = {"Authorization": f"Bearer {apikey}"}
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    try:
        image_bytes = query({"inputs": usertext})
        image = Image.open(io.BytesIO(image_bytes))
        save_path = os.path.join(app.config['DUMP'], 'img.jpg')
        image.save(save_path)
        return send_from_directory(app.config['DUMP'], 'img.jpg')
    except:
        return render_template("text2image.html", err_msg="The server is not responding. Please try again.", input_text=usertext)

# Text summarization page
@app.route('/textsum.html', methods=['GET', 'POST'])
def textsum():
    if frequest.method == 'GET':
        return render_template("textsum.html", min_length=30, max_length=130)
    usertxt = frequest.form['usertext']
    min_length = int(frequest.form['min_length'])
    max_length = int(frequest.form['max_length'])
    if 'userfile' in frequest.files:
        docfile = frequest.files['userfile']
        if docfile.filename != '':
            filename = secure_filename(docfile.filename)
            filename = 'doc' + os.path.splitext(filename)[1]
            filepath = os.path.join(app.config['UPLOAD'], filename)
            docfile.save(filepath)
            usertxt = read_document(filepath)
    if usertxt.strip() == "":
        return render_template("textsum.html", err_msg="Provide a text or a file please.", input_text=usertxt, min_length=min_length, max_length=max_length)
    usertxt_path = os.path.join(app.config['DUMP'], 'usertxt.txt')
    with open(usertxt_path, 'w', encoding='utf-8') as file:
        file.write(usertxt)
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {apikey}"}
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    try:
        sumtxt = query({"inputs": usertxt, "parameters": {"do_sample": False, "min_length": min_length, "max_length": max_length}})[0]['summary_text']
        sumtxt_path = os.path.join(app.config['DUMP'], 'sumtext.txt')
        with open(sumtxt_path, 'w', encoding='utf-8') as file:
            file.write(sumtxt)
        return render_template("textsum.html", input_text=usertxt, display="d-inline-block", sum_txt=sumtxt, min_length=min_length, max_length=max_length)
    except:
        return render_template("textsum.html", err_msg="The server is not responding. Please try again.", input_text=usertxt, min_length=min_length, max_length=max_length)

# Skin lesion classification page
@app.route('/skinlesion.html', methods=['GET', 'POST'])
def skinlesion():
    if frequest.method == "GET":
        return render_template("skinlesion.html")
    userimg = frequest.files['userimg']
    if userimg.filename == "":
        return render_template("skinlesion.html", err_msg="Provide an image please.")
    filename = secure_filename(userimg.filename)
    filename = 'lesion' + os.path.splitext(filename)[1]
    filepath = os.path.join(app.config['UPLOAD'], filename)
    userimg.save(filepath)
    pred = get_skin_lesion_prediction(filepath)
    return render_template("skinlesion.html", user_lesion_image=filepath, lesion_label=pred, display="d-inline-block")

def read_document(filepath):
    if filepath.endswith('.pdf'):
        text = read_pdf(filepath)
    elif filepath.endswith('.txt'):
        text = read_txt(filepath)
    elif filepath.endswith('.docx'):
        text = read_docx(filepath)
    else:
        text = None
    return text

def read_pdf(filepath):
    with open(filepath, "rb") as f:
        text = ''
        reader = PdfReader(f)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def read_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def read_docx(filepath):
    doc = Document(filepath)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def get_skin_lesion_prediction(img_path):
    # this function uses a pretrained keras model to classify an input image into 7 types of skin lesions.
    # the 7 classes are:
    classes = ["Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
               "Basal cell carcinoma", 
               "benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses",
               "Dermatofibroma",
               "Melanoma", 
               "Melanocytic nevi",
               "Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage"]  
    my_model=load_model("static/models/HAM10000.h5") # load the pretrained model
    SIZE = 32 # Resize to same size as training images
    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    img = img/255. # Scale pixel values
    img = np.expand_dims(img, axis=0)  
    pred = my_model.predict(img)                
    pred_class = classes[np.argmax(pred)] # Convert prediction to class name
    return pred_class

if __name__ == '__main__':
    app.run(debug=True, port=10000)