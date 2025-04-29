import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import io
from typing import Dict, List, Any, Tuple, Optional

def preprocess_image(image_bytes):
    """
    Enhanced preprocessing pipeline for better OCR results
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        Preprocessed image
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize image to improve OCR (maintain aspect ratio)
    height, width = img.shape[:2]
    scaling_factor = 2000 / max(height, width)  # Scale to have max dimension of 2000px
    if scaling_factor > 1:  # Only upscale, don't downscale
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges while reducing noise
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    
    # Apply thresholding
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Try another preprocessing technique if specific format is detected
    # This uses adaptive thresholding which may work better for some reports
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Return both versions for OCR
    return opening, adaptive_thresh

def extract_text_from_image(image_bytes):
    """
    Extract text from an image using OCR with improved config and multiple preprocessing approaches
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        Extracted text
    """
    # Preprocess the image with multiple techniques
    opening, adaptive = preprocess_image(image_bytes)
    
    # Convert to PIL Images for pytesseract
    pil_img_opening = Image.fromarray(opening)
    pil_img_adaptive = Image.fromarray(adaptive)
    
    # Configure Tesseract parameters for better results with tables
    # psm modes:
    # 3 = Fully automatic page segmentation, but no OSD
    # 4 = Assume a single column of text of variable sizes
    # 6 = Assume a single uniform block of text
    
    # Try different OCR configurations to get the best results
    custom_config_1 = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    custom_config_2 = r'--oem 3 --psm 4 -c preserve_interword_spaces=1'
    custom_config_3 = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
    
    # Extract text using all methods
    text_results = []
    
    # Try different combinations of preprocessing and OCR configs
    text_results.append(pytesseract.image_to_string(pil_img_opening, config=custom_config_1))
    text_results.append(pytesseract.image_to_string(pil_img_adaptive, config=custom_config_1))
    text_results.append(pytesseract.image_to_string(pil_img_opening, config=custom_config_2))
    text_results.append(pytesseract.image_to_string(pil_img_adaptive, config=custom_config_3))
    
    # Select the best result based on length and presence of key indicators
    best_text = ""
    best_score = 0
    
    for text in text_results:
        # Score each result based on length and presence of key terms
        score = len(text)
        
        # Check for lab report indicators
        indicators = ['test', 'reference', 'range', 'result', 'normal', 'value', 'parameter', 'unit']
        for indicator in indicators:
            if indicator in text.lower():
                score += 50  # Bonus for each indicator found
        
        if score > best_score:
            best_score = score
            best_text = text
    
    return best_text

def parse_reference_range(range_text: str) -> Dict[str, Any]:
    """
    Parse reference range text into structured data
    
    Args:
        range_text: Text containing reference range
        
    Returns:
        Dictionary with min and max values if available
    """
    result = {"original": range_text.strip()}
    
    # Common reference range patterns
    patterns = [
        r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)',  # 70-99
        r'(\d+\.?\d*)\s*to\s*(\d+\.?\d*)',  # 70 to 99
        r'<\s*(\d+\.?\d*)',  # <100
        r'>\s*(\d+\.?\d*)',  # >10
        r'≤\s*(\d+\.?\d*)',  # ≤100
        r'≥\s*(\d+\.?\d*)',   # ≥10
        r'(\d+\.?\d*)\s*[–—]\s*(\d+\.?\d*)'  # Using em dash or en dash: 70–99
    ]
    
    for pattern in patterns:
        match = re.search(pattern, range_text)
        if match:
            if len(match.groups()) == 2:  # Range with min and max
                result["min"] = float(match.group(1))
                result["max"] = float(match.group(2))
                result["type"] = "range"
                return result
            elif '<' in pattern or '≤' in pattern:  # Upper limit only
                result["max"] = float(match.group(1))
                result["type"] = "upper_limit"
                return result
            elif '>' in pattern or '≥' in pattern:  # Lower limit only
                result["min"] = float(match.group(1))
                result["type"] = "lower_limit"
                return result
    
    # If no pattern matches, return just the original text
    result["type"] = "unknown"
    return result

def check_value_within_range(value: float, range_info: Dict[str, Any]) -> bool:
    """
    Check if a value is within the given reference range
    
    Args:
        value: The test value
        range_info: Dictionary with range information
        
    Returns:
        True if the value is out of range, False otherwise
    """
    range_type = range_info.get("type", "unknown")
    
    if range_type == "range" and "min" in range_info and "max" in range_info:
        return value < range_info["min"] or value > range_info["max"]
    elif range_type == "upper_limit" and "max" in range_info:
        return value > range_info["max"]
    elif range_type == "lower_limit" and "min" in range_info:
        return value < range_info["min"]
    
    # Default to False if range can't be determined
    return False

def extract_lab_tests(text: str) -> List[Dict[str, Any]]:
    """
    Extract lab test data from text with improved pattern matching
    
    Args:
        text: Text extracted from lab report image
        
    Returns:
        List of dictionaries containing lab test information
    """
    lab_tests = []
    
    # Split text into lines for easier processing
    lines = text.split('\n')
    
    # Pre-process lines to fix common OCR issues
    cleaned_lines = []
    for line in lines:
        # Replace common OCR errors
        cleaned = line.replace('0', '0').replace('l', '1').replace('O', '0')
        # Remove excess whitespace
        cleaned = ' '.join(cleaned.split())
        if cleaned:  # Skip empty lines
            cleaned_lines.append(cleaned)
    
    # Enhanced patterns for different lab report formats
    patterns = [
        # Common formats with name, value, and range
        r'([A-Za-z\s\(\)\/\-\+]{2,})\s*(\d+\.?\d*)\s*([A-Za-z0-9\/\.\-\s\<\>\=\±]+)',
        
        # Format with test name, value, units, reference range
        r'([A-Za-z\s\(\)\/\-\+]{2,})\s*(\d+\.?\d*)\s*([A-Za-z\/]+)\s*([A-Za-z0-9\/\.\-\s\<\>\=\±]+)',
        
        # Format with test code, test name, value, reference range
        r'([A-Z0-9]+)\s+([A-Za-z\s\(\)\/\-\+]{2,})\s*(\d+\.?\d*)\s*([A-Za-z0-9\/\.\-\s\<\>\=\±]+)'
    ]
    
    # Try to extract tests using line by line approach
    for line in cleaned_lines:
        # Try each pattern for this line
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, line)
            if match:
                try:
                    if i == 0:  # First pattern: name, value, range
                        test_name = match.group(1).strip()
                        test_value = float(match.group(2).strip())
                        reference_range = match.group(3).strip()
                    elif i == 1:  # Second pattern: name, value, units, range
                        test_name = f"{match.group(1).strip()} ({match.group(3).strip()})"
                        test_value = float(match.group(2).strip())
                        reference_range = match.group(4).strip()
                    elif i == 2:  # Third pattern: code, name, value, range
                        test_name = f"{match.group(2).strip()} ({match.group(1).strip()})"
                        test_value = float(match.group(3).strip())
                        reference_range = match.group(4).strip()
                    
                    # Parse reference range
                    range_info = parse_reference_range(reference_range)
                    
                    # Check if value is out of range
                    out_of_range = check_value_within_range(test_value, range_info)
                    
                    lab_test = {
                        "test_name": test_name,
                        "test_value": test_value,
                        "bio_reference_range": reference_range,
                        "lab_test_out_of_range": out_of_range
                    }
                    
                    lab_tests.append(lab_test)
                    break  # Stop trying patterns for this line once we've found a match
                except (ValueError, IndexError):
                    # Skip entries that don't match the expected format
                    continue
    
    # If we didn't find any tests, try more aggressive approaches
    if not lab_tests:
        # Try detecting tables by looking for rows with consistent number of columns
        table_rows = []
        for line in cleaned_lines:
            # Check if line has multiple columns (multiple spaces or tabs)
            if re.search(r'\s{2,}|\t+', line):
                # Split columns by whitespace
                columns = re.split(r'\s{2,}|\t+', line)
                if len(columns) >= 3:  # Need at least 3 columns: name, value, range
                    table_rows.append(columns)
        
        # Group by number of columns to find the most common format
        column_counts = {}
        for row in table_rows:
            count = len(row)
            if count not in column_counts:
                column_counts[count] = []
            column_counts[count].append(row)
        
        # Process the most common format
        if column_counts:
            most_common_count = max(column_counts.keys(), key=lambda k: len(column_counts[k]))
            most_common_rows = column_counts[most_common_count]
            
            for row in most_common_rows:
                # Try to identify which columns contain what data
                test_name = None
                test_value = None
                reference_range = None
                
                for i, col in enumerate(row):
                    # Test names usually don't have numbers or are in the first column
                    if i == 0 or not re.search(r'\d', col):
                        if not test_name:
                            test_name = col.strip()
                    
                    # Test values are usually just numbers
                    elif re.match(r'^[\d\.]+$', col.strip()):
                        if not test_value:
                            try:
                                test_value = float(col.strip())
                            except ValueError:
                                continue
                    
                    # Reference ranges often have a dash, comparator or the word 'normal'
                    elif any(term in col.lower() for term in ['-', '<', '>', 'to', 'normal']):
                        if not reference_range:
                            reference_range = col.strip()
                
                if test_name and test_value and reference_range:
                    # Parse reference range
                    range_info = parse_reference_range(reference_range)
                    
                    # Check if value is out of range
                    out_of_range = check_value_within_range(test_value, range_info)
                    
                    lab_test = {
                        "test_name": test_name,
                        "test_value": test_value,
                        "bio_reference_range": reference_range,
                        "lab_test_out_of_range": out_of_range
                    }
                    
                    lab_tests.append(lab_test)
    
    # Remove duplicates based on test name
    unique_tests = {}
    for test in lab_tests:
        name = test["test_name"].lower()
        if name not in unique_tests:
            unique_tests[name] = test
    
    return list(unique_tests.values())

def process_lab_report(image_bytes) -> Dict[str, Any]:
    """
    Process lab report image and extract lab test data
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        Dictionary with extraction results
    """
    try:
        # Extract text from image
        text = extract_text_from_image(image_bytes)
        
        # Extract lab tests from text
        lab_tests = extract_lab_tests(text)
        
        return {
            "is_success": True,
            "data": lab_tests
        }
    except Exception as e:
        return {
            "is_success": False,
            "error": str(e)
        } 