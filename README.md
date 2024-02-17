# WebAppAi
This is a web application containing some simple image & text Ai tools; created with python and flask.

This application uses pretrained Ai models from https://huggingface.co/. Hence, you need to create a huggingface API key (free).

This application is deployed with render.com accessible at https://ai-btfi.onrender.com/.

## Deploy settings (render.com)

Runtime: `Python 3`

Build command: `python3 -m pip install --upgrade pip && pip install -r requirements.txt`

Start command: `gunicorn app:app`

### Environment Variables
Add these environment variables: 

`PYTHON_VERSION` =   `3.9.18`

`HF_API_KEY`     =   `your hugging face API key`