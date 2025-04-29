# Lab Report Processing API

A FastAPI-based service for extracting lab test data from medical reports using OCR (Optical Character Recognition) and RAG (Retrieval-Augmented Generation).

## Features

- Extract lab test names, values, and reference ranges from lab report images
- Determine if test values are outside normal ranges
- Compare performance of Tesseract OCR and EasyOCR
- Analyze lab results using RAG (Retrieval-Augmented Generation)
- User-friendly web interface with drag-and-drop functionality
- RESTful API endpoints for integration with other systems

## Demo

The application provides a user-friendly web interface:

1. Drag and drop a lab report image or click to browse
2. Optionally enter a query about the lab results
3. View the extracted lab tests in a table format
4. See which values are abnormal (outside reference ranges)
5. Get AI-powered analysis of your results via RAG

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system

#### Installing Tesseract OCR

On Windows:
- Download and install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Add the installation directory to your PATH environment variable

On macOS:
```bash
brew install tesseract
```

On Linux:
```bash
sudo apt-get install tesseract-ocr
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lab-report-processing.git
cd lab-report-processing
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python lab_report_api.py
```

5. Open your browser and go to `http://localhost:8000`

## API Documentation

### GET /

The root endpoint serves the web interface.

### POST /get-lab-tests

Extract lab tests from an image.

**Request:**
- Content-Type: multipart/form-data
- Body: file (image)

**Response:**
```json
{
  "is_success": true,
  "data": [
    {
      "test_name": "BILIRUBIN TOTAL",
      "test_value": 9.42,
      "bio_reference_range": "0.30- 1.20",
      "lab_test_out_of_range": true
    },
    {
      "test_name": "SGOT",
      "test_value": 162,
      "bio_reference_range": "0.00 - 46.00",
      "lab_test_out_of_range": true
    }
  ]
}
```

### POST /rag-lab-report

Process a lab report image and analyze with RAG.

**Request:**
- Content-Type: multipart/form-data
- Body:
  - file (image)
  - query (optional): A question about the lab results

**Response:**
```json
{
  "is_success": true,
  "data": [...],
  "rag_response": "The lab report shows elevated liver enzymes including SGOT (162) and SGPT (86), which are significantly above their normal ranges. This pattern suggests liver inflammation or damage..."
}
```

## Project Structure

- `lab_report_api.py` - FastAPI application
- `lab_report_processor.py` - Tesseract OCR implementation
- `easyocr_processor.py` - EasyOCR implementation
- `compare_ocr.py` - Comparison of OCR engines
- `enhance_ocr_output.py` - Enhanced formatting of OCR results
- `templates/` - HTML templates for web interface
- `static/` - Static files (CSS, images)

## Improving Results

To get better OCR results:
1. Use high-resolution images
2. Ensure good lighting and contrast
3. Avoid skewed or rotated images
4. Use images without handwritten text when possible

## License

MIT

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)
