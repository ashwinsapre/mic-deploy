import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_similarity

def similarity_check(sample_path):
	url="https://drive.google.com/uc?id=1IhwcaL-JKxb4kCl_Lr151fjpWuIHn0Mv&export=download"
	output="mean.npy"
	gdown.download(url, output, quiet=False)
	md5 = '251e16d46507539f68b64dc084500eda'
	gdown.cached_download(url, output, md5=md5)
	mean = np.load('mean.npy')
	sample = Image.open(sample_path).convert('RGB')
	w = min(sample.size[0], sample.size[1])
	sample = sample.resize((w, w))
	sample = np.array(sample).reshape(1, -1)
	mean = np.resize(mean,(w, w, 3)).reshape(1, -1)
	return cosine_similarity(mean, sample)