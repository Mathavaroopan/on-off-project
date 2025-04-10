from flask import Flask, render_template, request
from flask_socketio import SocketIO
import os
import mne
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import time
import tensorflow as tf
import serial
import serial.tools.list_ports
import json
import atexit

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
socketio = SocketIO(app)

# Load trained CNN model
model = load_model("cnn_eeg_model.h5")

# Ensure upload folder exists
os.makedirs("uploads", exist_ok=True)

# Global arduino connection
arduino = None

def cleanup_arduino():
    global arduino
    if arduino:
        try:
            arduino.close()
            print("Arduino connection closed")
        except:
            pass

# Register cleanup function
atexit.register(cleanup_arduino)

def get_arduino():
    """Get or create Arduino connection"""
    global arduino
    
    if arduino and arduino.is_open:
        return arduino
        
    # Close any existing connection
    cleanup_arduino()
    
    try:
        # Find Arduino port
        ports = list(serial.tools.list_ports.comports())
        port = 'COM7'  # Default
        for p in ports:
            if "CH340" in p.description or "Arduino" in p.description:
                port = p.device
                break
                
        # Open new connection
        arduino = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        print(f"Connected to Arduino on {port}")
        return arduino
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        return None

def control_arduino_leds(state):
    """Control Arduino LEDs with better error handling"""
    try:
        ser = get_arduino()
        if ser:
            command = f"{state}\n"
            ser.write(command.encode())
            time.sleep(0.1)
            
            while ser.in_waiting:
                response = ser.readline().decode().strip()
                print(f"Arduino says: {response}")
        else:
            print("Could not connect to Arduino")
    except Exception as e:
        print(f"Error controlling LEDs: {e}")
        cleanup_arduino()

def process_and_classify(filepath):
    print("Starting EEG processing...")
    raw = mne.io.read_raw_edf(filepath, preload=True)
    events, event_id = mne.events_from_annotations(raw)

    # Get ON and OFF event IDs
    on_ids = [v for k, v in event_id.items() if k.lower() == "on"]
    off_ids = [v for k, v in event_id.items() if k.lower() == "off"]

    if not on_ids or not off_ids:
        print("Missing ON/OFF events.")
        return []

    print(f"Found events - ON IDs: {on_ids}, OFF IDs: {off_ids}")
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
    print("Making predictions...")
    predictions = model.predict(X_scaled)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Create sequence of states with timing
    state_sequence = []
    print("\nStarting state sequence:")
    for i, label in enumerate(predicted_labels):
        state = "on" if label == 1 else "off"
        print(f"\nState {i+1}: {state}")
        state_sequence.append({"state": state, "index": i})
        socketio.emit("eeg_update", {"state": state, "index": i})
        control_arduino_leds(state)  # Control Arduino LEDs
        time.sleep(0.5)  # Delay for visual effect
    
    return state_sequence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test")
def test():
    filepath = "uploads/first-try.edf"
    if os.path.exists(filepath):
        print("\nStarting test with first-try.edf")
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
        print(f"\nProcessing uploaded file: {file.filename}")
        sequence = process_and_classify(filepath)
        return {"status": "success", "sequence": sequence}
    except Exception as e:
        print(f"Error processing file: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("\nStarting Flask application...")
    socketio.run(app, debug=True, use_reloader=False)  # Disable reloader
