from flask import Flask, send_from_directory, request, jsonify
import os
import subprocess
import json
import psutil
from threading import Thread

app = Flask(__name__, static_folder='../web')

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)
RESULTS_DIR = os.path.join(PARENT_DIR, 'results')
CONFIG_PATH = os.path.join(PARENT_DIR, 'config.json')

# Global state for training process
training_process = None

@app.route('/')
def index():
    return send_from_directory('../web', 'dashboard.html')

@app.route('/web/<path:path>')
def serve_web(path):
    return send_from_directory('../web', path)

@app.route('/results/<path:path>')
def serve_results(path):
    return send_from_directory('../results', path)

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'GET':
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({})
    elif request.method == 'POST':
        new_config = request.json
        with open(CONFIG_PATH, 'w') as f:
            json.dump(new_config, f, indent=4)
        return jsonify({"status": "success", "config": new_config})

@app.route('/api/train', methods=['POST'])
def start_training():
    global training_process
    
    if training_process and training_process.poll() is None:
        return jsonify({"status": "error", "message": "Training already in progress"}), 400
        
    data = request.json
    dataset = data.get('dataset', 'modular')
    epochs = str(data.get('epochs', 100))
    batch_size = str(data.get('batch_size', 512))
    lr = str(data.get('lr', 1e-3))
    
    # Construct command
    # python src/train.py --dataset ...
    cmd = [
        'python', 'src/train.py',
        '--dataset', dataset,
        '--epochs', epochs,
        '--batch_size', batch_size,
        '--lr', lr
    ]
    
    # Run in backend
    try:
        training_process = subprocess.Popen(
            cmd, 
            cwd=PARENT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return jsonify({"status": "started", "pid": training_process.pid})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_training():
    global training_process
    if training_process and training_process.poll() is None:
        training_process.terminate()
        return jsonify({"status": "stopped"})
    return jsonify({"status": "no running process"})

@app.route('/api/status', methods=['GET'])
def get_status():
    global training_process
    
    is_running = training_process is not None and training_process.poll() is None
    
    # Read history.json if exists for progress
    history = {}
    history_path = os.path.join(RESULTS_DIR, 'history.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except:
            pass
            
    return jsonify({
        "is_running": is_running,
        "history": history
    })

if __name__ == '__main__':
    print("Starting Grokking Dashboard on http://localhost:5000")
    app.run(port=5000, debug=True)
