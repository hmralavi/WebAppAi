from flask import Flask, render_template, send_from_directory
from flask import request as frequest
import requests
from dotenv import load_dotenv
import os
import io
from PIL import Image
 
app = Flask(__name__)
 
app.config['DUMP'] = os.path.join('static', 'gen_imgs')

load_dotenv()

apikey = os.environ['HF_API_KEY']
 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_text', methods=['POST'])
def submit_text():
    text = frequest.form['text']
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {apikey}"}
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    image_bytes = query({"inputs": text})
    image = Image.open(io.BytesIO(image_bytes))
    save_path = os.path.join(app.config['DUMP'], 'img.jpg')
    image.save(save_path)
    return send_from_directory(app.config['DUMP'], 'img.jpg')

if __name__ == '__main__':
    app.run(debug=True, port=10000)