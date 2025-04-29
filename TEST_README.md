# Lab Report Image Testing Tools

This directory contains tools for testing the lab report OCR and analysis system with individual images.

## Test Scripts

### 1. Single Image Analysis

This script analyzes a single lab report image locally, displaying preprocessing results, extracted text, and lab test data.

```bash
python test_single_image.py --image path/to/image.jpg
```

**Options:**
- `--image`: Path to the lab report image (required)
- `--output`: Output directory name (optional, default: "single_image_results_[timestamp]")

**Output:**
- Visualizations of original and preprocessed images
- Table of extracted lab tests with highlighting for abnormal values
- JSON file with full extraction results
- Text file with all extracted text

### 2. API Test Call

This script simulates an API call to the lab report processing service for a single image.

```bash
python test_api_call.py --image path/to/image.jpg --query "Optional query for RAG"
```

**Options:**
- `--image`: Path to the lab report image (required)
- `--endpoint`: API endpoint URL (optional, default: "http://localhost:8000")
- `--query`: Optional query for RAG-based analysis

**Output:**
- API response summary
- JSON file with full API response
- If query is provided, includes the RAG analysis response

### 3. Convenience Scripts

#### Windows (test_image.bat)

```bash
test_image.bat path\to\image.jpg [option] [query]
```

#### Linux/Mac (test_image.sh)

```bash
./test_image.sh path/to/image.jpg [option] [query]
```

**Options:**
- `local`: Run local analysis only (default)
- `api`: Test API endpoint only
- `both`: Run both local analysis and API test
- `rag`: Test RAG analysis with a query (requires providing a query)

**Examples:**
```bash
# Windows
test_image.bat data\lab_report.jpg both
test_image.bat data\lab_report.jpg rag "What tests are abnormal?"

# Linux/Mac
./test_image.sh data/lab_report.jpg both
./test_image.sh data/lab_report.jpg rag "What tests are abnormal?"
```

## Setting Up

1. Make sure you have installed all the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure Tesseract OCR is installed on your system

3. For API testing, make sure the FastAPI server is running:
   ```bash
   python fastapi_server.py
   ```

## Example Usage

### Testing OCR on a Single Image

```bash
python test_single_image.py --image data/sample_lab_report.jpg
```

This will:
- Process the image with different preprocessing methods
- Extract text using OCR
- Identify lab tests, values, and reference ranges
- Generate visualizations and summary reports

### Testing the API with RAG Query

```bash
python test_api_call.py --image data/sample_lab_report.jpg --query "What lab values are abnormal?"
```

This will:
- Send the image to the API
- Process it with the query
- Return the extracted lab tests and a RAG-generated response
- Save the results for review

### Quick Testing with Convenience Script

```bash
# Windows
test_image.bat data\sample_lab_report.jpg both "Are any values abnormal?"

# Linux/Mac
./test_image.sh data/sample_lab_report.jpg both "Are any values abnormal?"
```

This will run both local analysis and API testing with the specified query.

## Output Directories

- `single_image_results_[timestamp]/`: Results from local image analysis
- `api_results/`: Results from API calls

Each directory contains visualizations, extracted data, and other relevant information for analysis. 