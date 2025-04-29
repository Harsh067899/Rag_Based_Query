from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import os
import tempfile
import numpy as np
import cv2
from typing import Dict, Any
from datetime import datetime
import json

# Import lab report processing functions
from lab_report_processor import process_lab_report, extract_text_from_image, extract_lab_tests

app = FastAPI(title="Simple Lab Report API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Simple Lab Report Analysis API",
        "endpoints": {
            "POST /analyze-image": "Upload lab report image to extract test data with detailed analysis"
        }
    }

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Check if file is an image
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"is_success": False, "error": "File must be an image"}
            )
        
        # Read file content
        image_bytes = await file.read()
        
        # Convert to numpy array for display
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract text from image
        text = extract_text_from_image(image_bytes)
        
        # Extract lab tests
        lab_tests = extract_lab_tests(text)
        
        # Create detailed result (similar to test_single_image.py)
        result = {
            "is_success": True,
            "extraction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text_length": len(text),
            "tests_count": len(lab_tests),
            "abnormal_count": sum(1 for test in lab_tests if test['lab_test_out_of_range']),
            "extraction_method": "Enhanced OCR with multiple preprocessing techniques",
            "data": lab_tests,
            # Include a preview of the extracted text
            "extracted_text_preview": text[:1000] + "..." if len(text) > 1000 else text
        }
        
        return result
    except Exception as e:
        return {"is_success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("simple_lab_api:app", host="0.0.0.0", port=8000, reload=True) 