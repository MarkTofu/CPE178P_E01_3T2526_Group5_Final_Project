"""
Smoke-test for the preprocessing pipeline.
Run before training to catch environment issues.

Run: python check.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224

# Fake a 300×200 RGB image (arbitrary size — resize must handle it)
raw    = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)
resized = tf.image.resize(raw, [IMG_SIZE, IMG_SIZE])
normed  = preprocess_input(resized.numpy())

assert normed.shape == (IMG_SIZE, IMG_SIZE, 3), f'shape wrong: {normed.shape}'
assert normed.min() >= -1 - 1e-5, f'min {normed.min():.5f} below -1'
assert normed.max() <=  1 + 1e-5, f'max {normed.max():.5f} above  1'

print(f'✓ preprocessing OK  shape={normed.shape}  range=[{normed.min():.4f}, {normed.max():.4f}]')
