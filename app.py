import os
import glob
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, Response
from multiprocessing import Process
import training_script

app = Flask(__name__)
training_process = None
current_training_timestamp = None

@app.route('/')
def index():
    global training_process
    is_running = training_process is not None and training_process.is_alive()

    # If the process is finished, reset the timestamp
    if not is_running:
        global current_training_timestamp
        current_training_timestamp = None

    # Find log and submission files
    log_files = sorted(glob.glob('training_*.log'), reverse=True)
    submission_files = sorted(glob.glob('submission_*.csv'), reverse=True)

    return render_template('index.html',
                           is_running=is_running,
                           log_files=log_files,
                           submission_files=submission_files)

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
            log_file = f"training_{current_training_timestamp}.log"
            
            # Wait for the file to be created by the training process
            while not os.path.exists(log_file):
                time.sleep(0.2)
                # Check if the process died before creating the file
                if not (training_process and training_process.is_alive()):
                    yield "data: [STREAM_ERROR] Training process failed to start or create a log file.\n\n"
                    return

            with open(log_file, 'r') as f:
                # Start reading from the beginning of the file
                while training_process and training_process.is_alive():
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)  # No new line, wait a bit
                        continue
                    yield f"data: {line.rstrip()}\n\n"
                
                # After process finishes, read any remaining lines
                time.sleep(0.2) # Give a moment for final lines to be written
                for line in f.readlines():
                    yield f"data: {line.rstrip()}\n\n"

            yield "data: [STREAM_END] Training finished.\n\n"
        else:
            yield "data: No active training process found.\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/view/<path:filename>')
def view_file(filename):
    # Basic security check
    if not (filename.startswith('training_') and filename.endswith('.log')) and \
       not (filename.startswith('submission_') and filename.endswith('.csv')):
        return "Invalid file", 400

    try:
        return send_from_directory('.', filename, as_attachment=False)
    except FileNotFoundError:
        return "File not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)