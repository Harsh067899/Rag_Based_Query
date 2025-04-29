# Lab Report Processing API

This API processes lab report images to extract lab test names, values, and reference ranges using OCR and RAG (Retrieval-Augmented Generation).

## Features

- Extract lab test data from lab report images
- Process lab reports using RAG for enhanced analysis
- Upload PDF documents for RAG context
- Query document repository for information
- Web-based frontend for easy interaction with the API
- No external API dependencies - runs completely locally

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:

On Windows:
- Download and install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
- Add Tesseract to your PATH environment variable

On Linux:
```bash
sudo apt-get install tesseract-ocr
```

On macOS:
```bash
brew install tesseract
```

## Download Sample Data

Run the sample data downloader to get test lab report images:

```bash
python download_sample_data.py
```

This will download and extract sample lab report images to the `data` directory.

## Usage

### Starting the Server

```bash
python fastapi_server.py
```

This will start the FastAPI server at http://localhost:8000.

### Web Interface

Once the server is running, open your browser and navigate to:

```
http://localhost:8000
```

The web interface allows you to:
1. Upload lab report images
2. View extracted lab test data in a tabular format
3. Ask queries about the lab results using the RAG system
4. Clearly see which values are out of normal range

### API Endpoints

#### 1. Extract Lab Tests from Image

**Endpoint:** `/get-lab-tests`
**Method:** `POST`
**Parameters:** 
- `file`: Lab report image file (JPG, PNG, etc.)

**Example:**
```bash
python test_lab_report.py --image path/to/lab_report.jpg
```

#### 2. Upload PDF for RAG Context

**Endpoint:** `/upload-pdf`
**Method:** `POST`
**Parameters:**
- `file`: PDF file

**Example using curl:**
```bash
curl -X POST -F "file=@path/to/context.pdf" http://localhost:8000/upload-pdf
```

#### 3. Query Documents

**Endpoint:** `/query`
**Method:** `POST`
**Parameters:**
- `query`: Query string

**Example using curl:**
```bash
curl -X POST -d "query=What is the normal range for glucose?" http://localhost:8000/query
```

#### 4. Process Lab Report with RAG

**Endpoint:** `/rag-lab-report`
**Method:** `POST`
**Parameters:**
- `file`: Lab report image file
- `query`: (Optional) Query string for RAG analysis

**Example:**
```bash
python test_lab_report.py --image path/to/lab_report.jpg --query "Which tests are out of normal range?"
```

## Output Format

The `/get-lab-tests` and `/rag-lab-report` endpoints return JSON with the following structure:

```json
{
  "is_success": true,
  "data": [
    {
      "test_name": "Glucose",
      "test_value": 120,
      "bio_reference_range": "70-99 mg/dL",
      "lab_test_out_of_range": true
    },
    {
      "test_name": "Cholesterol",
      "test_value": 180,
      "bio_reference_range": "125-200 mg/dL",
      "lab_test_out_of_range": false
    }
  ]
}
```

The `lab_test_out_of_range` field is a boolean value indicating whether the test value is outside the normal reference range.

## Improving OCR Accuracy

To improve OCR accuracy:

1. Use high-quality images of lab reports
2. Ensure good lighting and contrast in the images
3. Avoid skewed or rotated images
4. Adjust the preprocessing parameters in `lab_report_processor.py` as needed

## Error Handling

If an error occurs during processing, the API will return a JSON response with:

```json
{
  "is_success": false,
  "error": "Error message"
}
```

## RAG Integration

The system uses Retrieval-Augmented Generation to enhance the analysis of lab reports:

1. Upload PDF documents to provide context for the RAG system
2. Process lab reports with a query to analyze the extracted data
3. The system retrieves relevant information from the documents and generates a response using a simple but effective local text processing algorithm

## Testing

Use the provided `test_lab_report.py` script to test the API:

```bash
python test_lab_report.py --image path/to/lab_report.jpg --query "What do these results mean?"
``` 