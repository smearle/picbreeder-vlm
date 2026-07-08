import tensorflow as tf
import numpy as np

MODEL_NAME = "SigLIP2-B-alignet"  # name of the model directory

images = np.zeros((8, 224, 224, 3), dtype=np.float32) # f32[B H W C]

m = tf.saved_model.load(MODEL_NAME)
forward = m.signatures['serving_default']
output = forward(images=images)