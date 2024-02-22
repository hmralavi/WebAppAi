from flask import Flask, render_template, send_from_directory
from flask import request as frequest
import requests
from dotenv import load_dotenv
import os
import io
from PIL import Image
import random
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from docx import Document
 
app = Flask(__name__)
 
app.config['DUMP'] = os.path.join('static', 'dump')
app.config['UPLOAD']  = os.path.join('static', 'uploads')

load_dotenv()

apikey = os.environ['HF_API_KEY']
 
@app.route('/')
@app.route('/index.html')
def homepage():
    return render_template('index.html')

@app.route('/text2image.html', methods=['GET', 'POST'])
def text2image():
    if frequest.method == 'GET':
        return render_template("text2image.html")
    usertext = frequest.form['usertext']
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
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
        return render_template("text2image.html", err_msg=f"The server is not responding. Please try again.", input_text=usertext)

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
        return render_template("textsum.html", err_msg=f"Provide a text or a file please.", input_text=usertxt, min_length=min_length, max_length=max_length)
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
        return render_template("textsum.html", input_text=usertxt, show_sum="show-sum", sum_txt=sumtxt, min_length=min_length, max_length=max_length)
    except:
        return render_template("textsum.html", err_msg=f"The server is not responding. Please try again.", input_text=usertxt, min_length=min_length, max_length=max_length)

def gen_random_text2image_prompt():
    with open("static/assets/random_image_titles.txt", 'r') as f:
        lines = f.readlines()
    i = random.randint(0, len(lines)-1)
    return lines[i]

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

if __name__ == '__main__':
    app.run(debug=True, port=10000)