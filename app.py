import io
import json
import torch.nn as nn 
import torchvision.transforms as transforms
import gdown
from PIL import Image
from flask import Flask, jsonify, request, render_template
from caption import *

app = Flask(__name__)

device = "cpu"
url="https://drive.google.com/uc?id=1---vD9czhFbkX2fjBdL4mpMevRRkThR_&export=download"
output="model.pth.tar"
gdown.download(url, output, quiet=False)
md5 = 'fa837a88f0c40c513d975104edf3da17'
gdown.cached_download(url, output, md5=md5, postprocess=gdown.extractall)

checkpoint = torch.load('model.pth.tar', map_location=str(device))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
with open('wordmap.json', 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

@app.route('/', methods=['POST'])
def predict():
    #file = request.files['file']
    #beam_size=3
    #seq, alphas = caption_image_beam_search(encoder, decoder, file, word_map, beam_size)
    #words = [rev_word_map[ind] for ind in seq]
    #return jsonify({'caption': words})
    return {'success':2}


if __name__ == '__main__':
    app.run()