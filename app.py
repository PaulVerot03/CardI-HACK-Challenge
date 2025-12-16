import os
import glob
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, Response, send_from_directory, flash
from multiprocessing import Process
from werkzeug.utils import secure_filename
import training_script

app = Flask(__name__)
# It's safer to use a secret key for flashing messages
app.config['SECRET_KEY'] = os.urandom(24)

# Define constants for directories
DATA_DIR = 'data'
OUTPUT_DIR = 'output'

training_process = None
current_training_timestamp = None

def setup_directories():
    """Ensure data and output directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run setup on startup
setup_directories()

@app.route('/')
def index():
    global training_process
    is_running = training_process is not None and training_process.is_alive()

    if not is_running:
        global current_training_timestamp
        current_training_timestamp = None

    # Check for data files
    train_csv_exists = os.path.exists(os.path.join(DATA_DIR, 'train.csv'))
    test_csv_exists = os.path.exists(os.path.join(DATA_DIR, 'test.csv'))

    # Find log and submission files
    log_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'training_*.log')), reverse=True)
    submission_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'submission_*.csv')), reverse=True)

    return render_template('index.html',
                           is_running=is_running,
                           log_files=log_files,
                           submission_files=submission_files,
                           train_csv_exists=train_csv_exists,
                           test_csv_exists=test_csv_exists)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'train_csv' not in request.files or 'test_csv' not in request.files:
        flash('No file part in the request', 'danger')
        return redirect(request.url)
    
    train_file = request.files['train_csv']
    test_file = request.files['test_csv']

    if train_file.filename == '' and test_file.filename == '':
        flash('No files selected for uploading', 'warning')
        return redirect(url_for('index'))

    if train_file and train_file.filename == 'train.csv':
        train_filename = secure_filename(train_file.filename)
        train_file.save(os.path.join(DATA_DIR, train_filename))
        flash('train.csv uploaded successfully!', 'success')

    if test_file and test_file.filename == 'test.csv':
        test_filename = secure_filename(test_file.filename)
        test_file.save(os.path.join(DATA_DIR, test_filename))
        flash('test.csv uploaded successfully!', 'success')
        
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train():
    global training_process, current_training_timestamp
    if training_process is None or not training_process.is_alive():
        n_estimators = int(request.form.get('n_estimators', 1000))
        n_jobs = int(request.form.get('n_jobs', 2))

        TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_training_timestamp = TIMESTAMP

        training_process = Process(target=training_script.run_training, args=(n_estimators, n_jobs, TIMESTAMP))
        training_process.start()

    return redirect(url_for('index'))

@app.route('/stream-logs')
def stream_logs():
    def generate():
        if current_training_timestamp:
            log_file = os.path.join(OUTPUT_DIR, f"training_{current_training_timestamp}.log")
            
            while not os.path.exists(log_file):
                time.sleep(0.2)
                if not (training_process and training_process.is_alive()):
                    yield "data: [STREAM_ERROR] Training process failed to start or create a log file.\n\n"
                    return

            with open(log_file, 'r') as f:
                while training_process and training_process.is_alive():
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    yield f"data: {line.rstrip()}\n\n"
                
                time.sleep(0.2)
                for line in f.readlines():
                    yield f"data: {line.rstrip()}\n\n"

            yield "data: [STREAM_END] Training finished.\n\n"
        else:
            yield "data: No active training process found.\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/view/<path:filename>')
def view_file(filename):
    # Security check now needs to be more flexible as filename is just the basename
    if not (filename.startswith('training_') or filename.startswith('submission_')):
        return "Invalid file", 400
    
    # Determine if it's a log or submission to look in the correct directory
    directory = OUTPUT_DIR
    
    try:
        return send_from_directory(directory, filename, as_attachment=False)
    except FileNotFoundError:
        return "File not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
