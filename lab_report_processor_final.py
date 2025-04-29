import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import io
from typing import Dict, List, Any, Tuple, Optional
import time

def preprocess_image(image_bytes):
    """
    Advanced preprocessing pipeline optimized for lab report OCR
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        Dict of preprocessed images with different techniques
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Create a set of preprocessed images for different elements
    preprocessed_images = {}
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize for better OCR if too small
    height, width = img.shape[:2]
    scaling_factor = 2000 / max(height, width)  # Scale to have max dimension of 2000px
    if scaling_factor > 1:  # Only upscale, don't downscale
        gray = cv2.resize(gray, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
    
    # 1. Preprocessing for test names - higher contrast to make text clearer
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    test_names_img = clahe.apply(gray)
    _, test_names_img = cv2.threshold(test_names_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images["test_names"] = test_names_img
    
    # 2. Preprocessing for numeric values - optimize for digits
    # Apply bilateral filter to reduce noise while preserving edges
    values_img = cv2.bilateralFilter(gray, 11, 17, 17)
    _, values_img = cv2.threshold(values_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images["values"] = values_img
    
    # 3. Preprocessing for reference ranges - adaptive thresholding
    # Often reference ranges have lower contrast and varied formats
    ranges_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    preprocessed_images["ranges"] = ranges_img
    
    # 4. General preprocessing for overall structure - combined approach
    # This helps with understanding the overall structure of the document
    general_img = clahe.apply(gray)
    bilateral = cv2.bilateralFilter(general_img, 9, 15, 15)
    _, general_img = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images["general"] = general_img
    
    return preprocessed_images, img

def extract_text_with_targeted_configs(preprocessed_images):
    """
    Extract text using targeted OCR configurations for different elements
    
    Args:
        preprocessed_images: Dict of preprocessed images for different elements
        
    Returns:
        Dict with extracted text for different elements
    """
    extracted_text = {}
    
    # Configuration for test names
    # PSM 4 - Assume a single column of text of variable sizes
    # Only whitelist relevant characters for test names
    test_names_config = r'--oem 3 --psm 4 -c preserve_interword_spaces=1'
    test_names_img = Image.fromarray(preprocessed_images["test_names"])
    extracted_text["test_names"] = pytesseract.image_to_string(test_names_img, config=test_names_config)
    
    # Configuration for numeric values
    # PSM 6 - Assume a single uniform block of text
    # Whitelist digits, decimal points and units
    values_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=0123456789./%()<>-+ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
    values_img = Image.fromarray(preprocessed_images["values"])
    extracted_text["values"] = pytesseract.image_to_string(values_img, config=values_config)
    
    # Configuration for reference ranges
    # PSM 6 - Assume a single uniform block of text
    ranges_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    ranges_img = Image.fromarray(preprocessed_images["ranges"])
    extracted_text["ranges"] = pytesseract.image_to_string(ranges_img, config=ranges_config)
    
    # General configuration for overall structure
    # PSM 3 - Fully automatic page segmentation
    general_config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
    general_img = Image.fromarray(preprocessed_images["general"])
    extracted_text["general"] = pytesseract.image_to_string(general_img, config=general_config)
    
    return extracted_text

def parse_reference_range(range_text: str) -> Dict[str, Any]:
    """
    Parse reference range text into structured data
    
    Args:
        range_text: Text containing reference range
        
    Returns:
        Dictionary with min and max values if available
    """
    result = {"original": range_text.strip()}
    
    # Fix common OCR errors
    cleaned_text = range_text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
    
    # Common reference range patterns
    patterns = [
        # Standard ranges with numeric bounds
        (r'(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)', 'range'),  # 70-99, 70–99 (en dash), 70—99 (em dash)
        (r'(\d+\.?\d*)\s*to\s*(\d+\.?\d*)', 'range'),  # 70 to 99
        
        # Upper limits
        (r'[<≤]\s*(\d+\.?\d*)', 'upper_limit'),  # <100, ≤100
        
        # Lower limits
        (r'[>≥]\s*(\d+\.?\d*)', 'lower_limit'),  # >10, ≥10
        
        # Specific formats with units embedded
        (r'(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)\s*(mg/dl|g/dl|U/L|IU/L|mmol/L)', 'range'),  # 70-99 mg/dl
    ]
    
    for pattern, pattern_type in patterns:
        match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if match:
            if pattern_type == 'range':
                try:
                    result["min"] = float(match.group(1))
                    result["max"] = float(match.group(2))
                    result["type"] = "range"
                    return result
                except ValueError:
                    pass
            elif pattern_type == 'upper_limit':
                try:
                    result["max"] = float(match.group(1))
                    result["type"] = "upper_limit"
                    return result
                except ValueError:
                    pass
            elif pattern_type == 'lower_limit':
                try:
                    result["min"] = float(match.group(1))
                    result["type"] = "lower_limit"
                    return result
                except ValueError:
                    pass
    
    # If no pattern matches, try to find any numbers that might be reference values
    numbers = re.findall(r'(\d+\.?\d*)', cleaned_text)
    if len(numbers) == 2:
        try:
            # Assume these are min-max if there are exactly two numbers
            result["min"] = float(numbers[0])
            result["max"] = float(numbers[1])
            result["type"] = "range"
            return result
        except ValueError:
            pass
    elif len(numbers) == 1:
        try:
            # For a single number, check surrounding text for clues
            if any(w in cleaned_text.lower() for w in ['less', 'below', 'under', '<', '≤']):
                result["max"] = float(numbers[0])
                result["type"] = "upper_limit"
            elif any(w in cleaned_text.lower() for w in ['more', 'above', 'over', '>', '≥']):
                result["min"] = float(numbers[0])
                result["type"] = "lower_limit"
            else:
                # Assume it's a normal value if no clear indicator
                result["equals"] = float(numbers[0])
                result["type"] = "equals"
            return result
        except ValueError:
            pass
    
    # If we couldn't parse anything, mark it unknown
    result["type"] = "unknown"
    return result

def check_value_within_range(value: float, range_info: Dict[str, Any]) -> bool:
    """
    Check if a value is outside the given reference range
    
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
    elif range_type == "equals" and "equals" in range_info:
        # Allow a 10% deviation for equals
        deviation = range_info["equals"] * 0.1
        return abs(value - range_info["equals"]) > deviation
    
    # Default to False if range can't be determined
    return False

def extract_unit_from_value(value_text: str) -> Tuple[float, str]:
    """
    Extract numeric value and unit from a value text
    
    Args:
        value_text: Text containing a value and possibly a unit
        
    Returns:
        Tuple of (numeric_value, unit)
    """
    # Common units in lab tests
    units = ['mg/dl', 'g/dl', 'mg/dL', 'g/dL', 'mmol/L', 'ng/ml', 'pg/ml', 
            'U/L', 'IU/L', 'mIU/L', 'mmHg', '%', 'mEq/L', 'mm/hr', 'μg/dL', 
            'μmol/L', 'mcg/dL', 'mg/L', 'g/L', 'ug/dL', 'uIU/mL']
    
    # Default values
    numeric_value = None
    unit = ""
    
    # First, try to extract any number
    number_match = re.search(r'(\d+\.?\d*)', value_text)
    if number_match:
        try:
            numeric_value = float(number_match.group(1))
        except ValueError:
            pass
    
    # Then try to find a unit
    for u in units:
        if u.lower() in value_text.lower():
            unit = u
            break
    
    # If no standard unit is found, try a more generic pattern
    if not unit:
        unit_match = re.search(r'(\d+\.?\d*)\s*([a-zA-Z/%]+)', value_text)
        if unit_match and len(unit_match.groups()) > 1:
            unit = unit_match.group(2).strip()
    
    return numeric_value, unit

def extract_lab_tests(extracted_text: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Extract lab test data from OCR extracted text
    
    Args:
        extracted_text: Dictionary with extracted text for different elements
        
    Returns:
        List of dictionaries containing lab test information
    """
    lab_tests = []
    
    # Combine text from all extractions
    combined_text = extracted_text["general"]
    
    # Split text into lines
    lines = []
    for line in combined_text.split('\n'):
        line = line.strip()
        if line:
            # Fix common OCR errors
            line = (line.replace('O', '0').replace('o', '0')
                     .replace('l', '1').replace('I', '1'))
            lines.append(line)
    
    # Pattern matching for different lab report formats
    patterns = [
        # Pattern 1: Test name followed by value and reference range
        r'([A-Za-z\s\(\)\/\-\+]{2,})\s+(\d+\.?\d*)\s*([A-Za-z0-9\/\.\-\s\<\>\=\±]+)',
        
        # Pattern 2: Test name with value and possibly unit
        r'([A-Za-z\s\(\)\/\-\+]{2,})\s+(\d+\.?\d*)\s*([A-Za-z\/]+)',
        
        # Pattern 3: Test name with H or L indicator (high/low) and value
        r'([A-Za-z\s\(\)\/\-\+]{2,})\s+(\d+\.?\d*\s*[HL])\s*([A-Za-z0-9\/\.\-\s\<\>\=\±]+)',
    ]
    
    # Process each line to extract test information
    for line in lines:
        # Skip lines that look like headers or footers
        if any(header in line.lower() for header in ["patient", "doctor", "date", "report", "laboratory", "page", "specimen"]):
            continue
            
        # Skip short lines
        if len(line) < 5:
            continue
            
        # Try each pattern
        found_match = False
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                found_match = True
                
                try:
                    # Extract test name, value and reference range
                    test_name = match.group(1).strip()
                    
                    # Extract numeric value and unit
                    value_text = match.group(2).strip()
                    numeric_value, unit = extract_unit_from_value(value_text)
                    
                    if not numeric_value:
                        continue
                    
                    # Extract reference range
                    if len(match.groups()) >= 3:
                        reference_range = match.group(3).strip()
                    else:
                        reference_range = ""
                    
                    # Skip suspicious test names
                    if len(test_name) < 2 or test_name.isdigit():
                        continue
                    
                    # Parse reference range
                    range_info = parse_reference_range(reference_range)
                    
                    # Check if value is out of range
                    out_of_range = check_value_within_range(numeric_value, range_info)
                    
                    lab_test = {
                        "test_name": test_name,
                        "test_value": numeric_value,
                        "test_unit": unit,
                        "bio_reference_range": reference_range,
                        "lab_test_out_of_range": out_of_range
                    }
                    
                    lab_tests.append(lab_test)
                    break
                except (ValueError, IndexError):
                    continue
        
        # If no pattern matched but line contains numbers, try manual extraction
        if not found_match and re.search(r'\d+\.?\d*', line):
            # Split line at the first number
            parts = re.split(r'(\d+\.?\d*)', line, 1)
            if len(parts) >= 3:
                test_name = parts[0].strip()
                
                # If test name is valid
                if len(test_name) >= 2 and not test_name.isdigit():
                    value_text = parts[1] + parts[2].split()[0] if len(parts) > 2 and len(parts[2].split()) > 0 else parts[1]
                    numeric_value, unit = extract_unit_from_value(value_text)
                    
                    if numeric_value:
                        # Assume anything after value is reference range
                        remaining_text = " ".join(parts[2].split()[1:]) if len(parts) > 2 and len(parts[2].split()) > 1 else ""
                        range_info = parse_reference_range(remaining_text)
                        out_of_range = check_value_within_range(numeric_value, range_info)
                        
                        lab_test = {
                            "test_name": test_name,
                            "test_value": numeric_value,
                            "test_unit": unit,
                            "bio_reference_range": remaining_text,
                            "lab_test_out_of_range": out_of_range
                        }
                        
                        lab_tests.append(lab_test)
    
    # Deduplicate tests by name
    unique_tests = {}
    for test in lab_tests:
        test_name_lower = test["test_name"].lower()
        
        if test_name_lower not in unique_tests:
            unique_tests[test_name_lower] = test
        elif len(test["test_name"]) > len(unique_tests[test_name_lower]["test_name"]):
            # Keep the one with the longer name (likely more complete)
            unique_tests[test_name_lower] = test
    
    # Convert dictionary back to list
    return list(unique_tests.values())

def process_lab_report(image_bytes) -> Dict[str, Any]:
    """
    Process a lab report image using OCR
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        Dictionary with extracted lab test data
    """
    start_time = time.time()
    
    try:
        # Preprocess the image
        preprocessed_images, original_img = preprocess_image(image_bytes)
        
        # Extract text with targeted OCR configurations
        extracted_text = extract_text_with_targeted_configs(preprocessed_images)
        
        # Extract lab tests from the text
        lab_tests = extract_lab_tests(extracted_text)
        
        # Count tests that are out of range
        tests_out_of_range = sum(1 for test in lab_tests if test["lab_test_out_of_range"])
        
        # Format the output according to the required API response
        result = {
            "is_success": True,
            "data": [{
                "test_name": test["test_name"],
                "test_value": test["test_value"],
                "bio_reference_range": test["bio_reference_range"],
                "lab_test_out_of_range": test["lab_test_out_of_range"]
            } for test in lab_tests]
        }
        
        return result
    except Exception as e:
        return {
            "is_success": False,
            "error": str(e)
        } 