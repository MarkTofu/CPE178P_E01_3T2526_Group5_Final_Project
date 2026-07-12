# Philippine Paper Bill Detector

An AI-powered desktop app that identifies Philippine paper bill denominations from a photo. Upload a picture of a bill and the model will tell you what denomination it is along with a confidence score.

**Supported denominations:** ₱20, ₱50, ₱100, ₱500, ₱1000

Built with MobileNetV2 transfer learning and Flet.

---

## Running the App

**Requirements:** Python 3.11

1. Clone the repo
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. Install dependencies
   ```bash
   py -m pip install -r requirements.txt
   ```

3. Run the app
   ```bash
   py app.py
   ```

---

## Retraining the Model

Only needed if you want to train on your own photos.

**Photo requirements:** 100 training + 25 validation photos per denomination, JPG or PNG.

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

3. Train
   ```bash
   py train.py
   ```
   Saves the model to `model/saved_model/` and a graph to `model/accuracy.png`.
