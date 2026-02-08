Bird vs Drone — Flask Web App

Files added:
- app.py — Flask application (upload/predict, webcam API, evaluate page)
- templates/index.html — main UI (upload + webcam)
- templates/evaluate.html — evaluation UI and results
- static/style.css — custom styles
- static/app.js — frontend JS for webcam + upload
- static/tmp/ — temporary folder for generated confusion matrix images

How to run
1. Create and activate a Python virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Put your Keras model file `bird_drone_classifier_model.h5` in the project root (same folder as `app.py`).

4. Start the Flask app:

```powershell
python app.py
```

5. Open http://localhost:8501 in your browser.

Notes
- Webcam feature uses getUserMedia and sends frames to the server as base64 POST requests. For local testing, Chrome/Edge work well. HTTPS is required for webcam in some browsers when not on localhost.
- Evaluation page runs predictions across the images folder you specify and generates a confusion matrix image saved to `static/tmp/`.
- For large datasets, evaluation will be slow on CPU; consider using a GPU or running smaller subsets for testing.

If you want, I can:
- Add a download button for evaluation CSV export
- Add authentication or admin UI
- Containerize the app (Dockerfile)

*** End Patch