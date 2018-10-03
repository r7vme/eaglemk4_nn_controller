#!/usr/bin/env python3
# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
Script to run VAE model
'''

import os
import tensorflow as tf
import random
import numpy as np

from PIL import Image
from model import reset_graph, ConvVAE

# Parameters for training
DATA_DIR = "record"

def load_image(filename):
  data = np.zeros((1, 64, 64, 3), dtype=np.uint8)
  image = Image.open(os.path.join(DATA_DIR, filename))
  image_6464 = image.resize((64, 64))
  data = np.asarray(image_6464)
  image.close()
  return data

# load dataset from record/*. only use first 10K, sorted by filename.
filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]

reset_graph()

vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)

vae.load_json(os.path.join('tf_vae', 'vae.json'))

img = load_image(filelist[0])
norm_img = np.copy(img).astype(np.float)/255.0
norm_img = norm_img.reshape(1, 64, 64, 3)

z = vae.encode(norm_img)

decoded = vae.decode(z.reshape(1, 64)) * 255.
decoded = np.round(decoded).astype(np.uint8)
decoded = decoded.reshape(64, 64, 3)

decoded_img = Image.fromarray(decoded)
decoded_img.save('decoded_' + filelist[0])

orig_img = Image.fromarray(img)
orig_img.save('orig_' + filelist[0])
