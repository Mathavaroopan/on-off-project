from flask import Flask, render_template, request
from flask_socketio import SocketIO
import os
import mne
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import time
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
socketio = SocketIO(app)

# Load trained CNN model
model = load_model("cnn_eeg_model.h5")

# Ensure upload folder exists
os.makedirs("uploads", exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test")
def test():
    filepath = "uploads/first-try.edf"
    if os.path.exists(filepath):
        sequence = process_and_classify(filepath)
        return {"status": "success", "sequence": sequence}
    return {"status": "error", "message": "Test file not found!"}

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return {"status": "error", "message": "No file uploaded"}
    
    file = request.files["file"]
    if file.filename == '':
        return {"status": "error", "message": "No file selected"}
    
    if not file.filename.endswith('.edf'):
        return {"status": "error", "message": "Please upload an EDF file"}
    
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    
    try:
        sequence = process_and_classify(filepath)
        return {"status": "success", "sequence": sequence}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def process_and_classify(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=True)
    events, event_id = mne.events_from_annotations(raw)

    # Get ON and OFF event IDs
    on_ids = [v for k, v in event_id.items() if k.lower() == "on"]
    off_ids = [v for k, v in event_id.items() if k.lower() == "off"]

    if not on_ids or not off_ids:
        print("Missing ON/OFF events.")
        return []

    events_of_interest = [e for e in events if e[2] in on_ids + off_ids]
    events_of_interest = np.array(events_of_interest)

    tmin, tmax = -1, 2
    baseline = (None, 0)

    epochs = mne.Epochs(raw, events_of_interest, tmin=tmin, tmax=tmax,
                        baseline=baseline, detrend=1, preload=True)
    
    X = epochs.get_data()
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

    # Standardize
    scaler = StandardScaler()
    X_scaled = np.zeros_like(X)
    for i in range(X.shape[1]):
        X_scaled[:, i, :, 0] = scaler.fit_transform(X[:, i, :, 0])

    # Predict
    predictions = model.predict(X_scaled)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Create sequence of states with timing
    state_sequence = []
    for i, label in enumerate(predicted_labels):
        state = "on" if label == 1 else "off"
        state_sequence.append({"state": state, "index": i})
        socketio.emit("eeg_update", {"state": state, "index": i})
        time.sleep(0.5)  # Delay for visual effect
    
    return state_sequence

if __name__ == "__main__":
    socketio.run(app, debug=True)
