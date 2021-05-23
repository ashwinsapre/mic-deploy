import io
import json
import torch.nn as nn 
import torchvision.transforms as transforms
import gdown
from PIL import Image
from flask import Flask, jsonify, request, render_template
from caption import *

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
url="https://drive.google.com/uc?id=1---vD9czhFbkX2fjBdL4mpMevRRkThR_&export=download"
output="model.pth.tar"
gdown.download(url, output, quiet=False)
md5 = '251e16d46507539f68b64dc084500eda'
gdown.cached_download(url, output, md5=md5)

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
    file = request.files['file']
    mean = np.load('mean.npy')
    sample = Image.open(sample_path).convert('RGB')
    w = min(sample.size[0], sample.size[1])
    sample = sample.resize((w, w))
    sample = np.array(sample).reshape(1, -1)
    mean = np.resize(mean,(w, w, 3)).reshape(1, -1)
    similarity_score = cosine_similarity(mean, sample)
    if similarity_score <= 0.85:
    	return jsonify({'error':1,'caption': 'The image is not an x-ray, please try again.'})
    else:
        beam_size=3
        seq, alphas = caption_image_beam_search(encoder, decoder, file, word_map, beam_size)
        words = [rev_word_map[ind] for ind in seq]
        words.pop(0)
        words.pop()
        for n, i in enumerate(words):
            if i == '<stop>':
                words[n] = '.'
        temp=str(words[0]).capitalize()
        words[0]=temp

        return jsonify({'error':0, 'caption': words})

if __name__ == '__main__':
    app.run()