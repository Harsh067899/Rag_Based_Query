from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
from typing import Dict, List, Any

app = FastAPI(title="Simple OCR Lab Report API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def simple_preprocess(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize for better OCR if too small
    height, width = img.shape[:2]
    scaling_factor = 2000 / max(height, width)  # Scale to have max dimension of 2000px
    if scaling_factor > 1:  # Only upscale, don't downscale
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Simple threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh, img

def simple_extract_text(image):
    # Convert to PIL Image for pytesseract
    pil_img = Image.fromarray(image)
    
    # Try different OCR configurations to get better results
    configs = [
        r'--oem 3 --psm 6',  # Assume a single uniform block of text
        r'--oem 3 --psm 4',  # Assume a single column of text of variable sizes
        r'--oem 3 --psm 11'  # Sparse text. Find as much text as possible in no particular order
    ]
    
    # Extract text with each config and pick the best one
    texts = []
    for config in configs:
        text = pytesseract.image_to_string(pil_img, config=config)
        texts.append(text)
    
    # Use the text with the most content
    best_text = max(texts, key=len)
    return best_text

def simple_extract_tests(text):
    tests = []
    
    # Split text into lines
    lines = text.split('\n')
    
    # Multiple patterns to try for different lab report formats
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Try multiple pattern matching approaches
        
        # Pattern 1: Test name followed by value with possibly colon or space
        match1 = re.search(r'([A-Za-z][A-Za-z0-9\s\-]+)[\s:]+(\d+\.?\d*)', line)
        
        # Pattern 2: Test name followed by value in parentheses
        match2 = re.search(r'([A-Za-z][A-Za-z0-9\s\-]+)\s*\(?\s*(\d+\.?\d*)', line)
        
        # Pattern 3: Any words followed by digits
        match3 = re.search(r'([A-Za-z][A-Za-z0-9\s\-]{2,})\s+(\d+\.?\d*)', line)
        
        # Use the first match found
        match = match1 or match2 or match3
        
        if match:
            test_name = match.group(1).strip()
            test_value = match.group(2).strip()
            
            # Ignore very short test names (likely false positives)
            if len(test_name) < 2:
                continue
                
            # Various patterns for reference ranges
            range_patterns = [
                r'\(([^)]+)\)',  # Range in parentheses
                r'(\d+\s*[-–—]\s*\d+)',  # Digit-dash-digit with various dash types
                r'(\d+\s*to\s*\d+)',  # Digit "to" digit
                r'[<>]\s*(\d+)',  # Less than or greater than a number
                r'(\d+\s*[-–—]\s*\d+\s*[A-Za-z]+/[A-Za-z]+)'  # Range with units
            ]
            
            range_value = "Unknown"
            for pattern in range_patterns:
                range_match = re.search(pattern, line)
                if range_match:
                    range_value = range_match.group(1)
                    break
            
            # Add to tests - avoid duplicates
            if not any(t["test_name"].lower() == test_name.lower() for t in tests):
                tests.append({
                    "test_name": test_name,
                    "test_value": test_value,
                    "bio_reference_range": range_value,
                    "lab_test_out_of_range": False  # Simplified - not checking range
                })
    
    return tests

@app.get("/")
async def root():
    return {
        "message": "Simple OCR Lab Report API",
        "endpoints": {
            "POST /analyze": "Upload lab report image for simple OCR analysis"
        }
    }

@app.post("/analyze")
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
        
        # Simple preprocessing
        processed_image, original_img = simple_preprocess(image_bytes)
        
        # Extract text
        text = simple_extract_text(processed_image)
        
        # Simple extraction of tests
        tests = simple_extract_tests(text)
        
        # If no tests found with the processed image, try with original
        if len(tests) == 0:
            # Convert original to grayscale
            gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            gray_original = cv2.cvtColor(gray_original, cv2.COLOR_RGB2GRAY)
            
            # Try extracting text with different settings
            text2 = pytesseract.image_to_string(Image.fromarray(gray_original), config=r'--oem 3 --psm 3')
            tests = simple_extract_tests(text2)
            
            # If still no tests found, combine texts
            if len(tests) == 0:
                all_text = text + "\n" + text2
                tests = simple_extract_tests(all_text)
        
        # Create result
        result = {
            "is_success": True,
            "text": text,
            "tests_count": len(tests),
            "data": tests
        }
        
        return result
    except Exception as e:
        return {"is_success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("simple_ocr_api:app", host="0.0.0.0", port=8000, reload=True) 