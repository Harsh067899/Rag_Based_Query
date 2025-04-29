from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from lab_report_processor_final import process_lab_report

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'is_success': False, 'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'is_success': False, 'error': 'No selected file'})
    
    # Check if it's an image
    if not file.content_type.startswith('image/'):
        return jsonify({'is_success': False, 'error': 'File must be an image'})
    
    # Read the file contents
    image_bytes = file.read()
    
    # Process the image
    result = process_lab_report(image_bytes)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
