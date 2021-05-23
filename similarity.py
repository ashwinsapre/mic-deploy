import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_similarity
import gdown

def similarity_check(sample_path, mean):
	sample = Image.open(sample_path).convert('RGB')
	w = min(sample.size[0], sample.size[1])
	sample = sample.resize((w, w))
	sample = np.array(sample).reshape(1, -1)
	mean = np.resize(mean,(w, w, 3)).reshape(1, -1)
	return cosine_similarity(mean, sample)