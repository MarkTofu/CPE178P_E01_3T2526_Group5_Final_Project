"""
Philippine Bill CNN — Training Script
MobileNetV2 transfer learning via TensorFlow / Keras

Data layout:
  data/
    train/  20/  50/  100/  500/  1000/   ← .jpg / .jpeg / .png
    val/    20/  50/  100/  500/  1000/

Run:
  py -m pip install -r requirements.txt
  py train.py

Output: model/saved_model/
Convert for web app:
  py -m pip install tensorflowjs
  tensorflowjs_converter --input_format tf_saved_model model/saved_model model/tfjs
"""

import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model

# ── Config ────────────────────────────────────────────────────────────────────

CLASSES  = ['20', '50', '100', '500', '1000']
IMG_SIZE = 224
BATCH    = 16
EPOCHS   = 20
DATA_DIR = pathlib.Path('data')

# ── Data ──────────────────────────────────────────────────────────────────────

def make_dataset(split):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        label_mode='categorical',
        class_names=CLASSES,
        shuffle=(split == 'train'),
    )
    # No preprocessing here — Rescaling layer inside model handles it
    return ds.prefetch(tf.data.AUTOTUNE)

# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet',
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # ── Advanced Augmentation Group ──
    x = layers.RandomFlip('horizontal_and_vertical')(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomBrightness(0.2)(x)
    x = layers.RandomContrast(0.2)(x)
    
    # ── Normalization ──
    x = layers.Rescaling(scale=1./127.5, offset=-1)(x)
    
    # ── Feature Extraction ──
    x = base(x, training=False)
    
    # ── Regularized Classification Head ──
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)  
    
    out = layers.Dense(len(CLASSES), activation='softmax', name='bill_predictions')(x)
    model = Model(inputs=inputs, outputs=out)

    model.compile(
        # Lower learning rate (2e-4) keeps gradient updates stable and avoids steep drops
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_ds = make_dataset('train')
    val_ds   = make_dataset('val')

    model = build_model()
    model.summary()

    # Increased patience to 6 to give the validation curve room to breathe
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    pathlib.Path('model').mkdir(exist_ok=True)

    plt.plot(history.history['accuracy'],     label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.tight_layout()
    plt.savefig('model/accuracy.png')
    print('✓ Graph saved → model/accuracy.png')

    model.export('model/saved_model')
    print('\n✓ Model saved → model/saved_model/')