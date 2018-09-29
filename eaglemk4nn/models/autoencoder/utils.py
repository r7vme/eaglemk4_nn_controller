"""
Modified from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/utils.py
"""
import os
import numpy as np
import scipy.misc
import tensorflow as tf

def save_images(images, size, image_path, gray=False):
    return imsave(inverse_transform(images), size, image_path, gray=gray)

def imsave(images, size, path, gray=False):
    return scipy.misc.imsave(path, merge(images, size, gray=gray))

def save(sess, saver, checkpoint_dir, step, name):
  """Save tensorflow model checkpoint"""
  model_name = name
  model_dir = name
  checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def load(sess, saver, checkpoint_dir, name):
  """Load tensorflow model checkpoint"""
  print(" [*] Reading checkpoints: {}".format(checkpoint_dir))

  model_dir = name
  checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    print(" [*] Checkpoints read: {}".format(ckpt_name))
    return True
  else:
    print(" [!] Failed reading.")
    return False

def inverse_transform(images):
    return (images+1.)/2.

def merge(images, size, gray=False):
    h, w = images.shape[1], images.shape[2]
    if gray:
      img = np.zeros((h * size[0], w * size[1]))
    else:
      img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img
