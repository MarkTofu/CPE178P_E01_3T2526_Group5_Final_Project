# Philippine Paper Bill Detector

An AI-powered web app that identifies Philippine paper bill denominations from a photo. Upload a picture of a bill and the model will tell you what denomination it is along with a confidence score.

**Supported denominations:** ₱20, ₱50, ₱100, ₱500, ₱1000

Built with MobileNetV2 transfer learning and TensorFlow.js. Runs entirely in the browser — no backend, no server required beyond a simple local file server.

---

## Running the Web App

**Requirements:** Python 3.11

1. Clone the repo
   ```bash
   git clone https://https://github.com/MarkTofu/CPE178P_E01_3T2526_Group5_Final_Project
   cd YOUR_REPO_NAME
   ```

2. Start a local server
   ```bash
   py -m http.server 8000
   ```

3. Open your browser and go to `http://localhost:8000`

That's it. The trained model is included in the repo under `model/tfjs/`.

---

## Retraining the Model

Only needed if you want to train on your own photos.

**Requirements:** Python 3.11

**Photo requirements:** 100 training + 25 validation photos per denomination, in JPG or PNG format.

**Folder structure:**
```
data/
  train/  20/  50/  100/  500/  1000/
  val/    20/  50/  100/  500/  1000/
```

1. Install dependencies
   ```bash
   py -m pip install -r requirements.txt
   ```

2. Verify your environment
   ```bash
   py check.py
   ```

3. Train the model
   ```bash
   py train.py
   ```
   This saves the model to `model/saved_model/` and an accuracy graph to `model/accuracy.png`.

4. Convert for the web app (run in Google Colab — see below)

### Converting the model in Google Colab

Open [Google Colab](https://colab.research.google.com) and run these cells:

**Cell 1**
```python
!pip install tensorflowjs
```

**Cell 2**
```python
from google.colab import files
files.upload()  # upload saved_model.zip (zip your model/saved_model/ folder first)
```

**Cell 3**
```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model

CLASSES  = ['20', '50', '100', '500', '1000']
IMG_SIZE = 224

orig = tf.saved_model.load('saved_model')
w    = {v.name: v for v in orig.variables}

base   = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                     include_top=False, pooling='avg', weights='imagenet')
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x      = layers.Rescaling(scale=1./127.5, offset=-1)(inputs)
x      = base(x, training=False)
x      = layers.Dense(128, name='dense')(x)
out    = layers.Dense(len(CLASSES), activation='softmax', name='bill_predictions')(x)
clean  = Model(inputs=inputs, outputs=out)

clean.get_layer('dense').set_weights([
    w['dense/kernel:0'].numpy(), w['dense/bias:0'].numpy()
])
clean.get_layer('bill_predictions').set_weights([
    w['bill_predictions/kernel:0'].numpy(), w['bill_predictions/bias:0'].numpy()
])

clean.export('saved_model_clean')
!tensorflowjs_converter --input_format tf_saved_model saved_model_clean tfjs
```

**Cell 4**
```python
!zip -r tfjs.zip tfjs
files.download('tfjs.zip')
```

Unzip `tfjs.zip` and place the `tfjs/` folder inside your `model/` folder, then re-run the web app.
