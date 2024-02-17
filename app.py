from flask import Flask, render_template, send_from_directory
from flask import request as frequest
import requests
from dotenv import load_dotenv
import os
import io
from PIL import Image
import random
 
app = Flask(__name__)
 
app.config['DUMP'] = os.path.join('static', 'gen_imgs')

load_dotenv()

apikey = os.environ['HF_API_KEY']
 
@app.route('/')
@app.route('/index.html')
def homepage():
    return render_template('index.html')

@app.route('/text2image.html', methods=['GET', 'POST'])
def text2image():
    if frequest.method == 'GET':
        return render_template("text2image.html", input_prompt=gen_random_text2image_prompt())
    if "gen_img_btn" in frequest.form:
        usertext = frequest.form['usertext']
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        headers = {"Authorization": f"Bearer {apikey}"}
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.content
        image_bytes = query({"inputs": usertext})
        image = Image.open(io.BytesIO(image_bytes))
        save_path = os.path.join(app.config['DUMP'], 'img.jpg')
        image.save(save_path)
        return render_template("text2image.html", input_prompt=usertext, output_prompt=usertext, output_img=save_path)
    elif "gen_txt_btn" in frequest.form:
        return render_template("text2image.html", input_prompt=gen_random_text2image_prompt())

def gen_random_text2image_prompt():
    with open("static/materials/random_image_titles.txt", 'r') as f:
        lines = f.readlines()
    i = random.randint(0, len(lines)-1)
    return lines[i]

if __name__ == '__main__':
    app.run(debug=True, port=10000)