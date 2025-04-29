import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import io
from typing import Dict, List, Any, Tuple, Optional
import time

def preprocess_image_enhanced(image_bytes):
    """
    Advanced preprocessing pipeline for better OCR results
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        List of preprocessed images with different techniques
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
    
    # Create a list of preprocessed images with different techniques
    preprocessed_images = []
    
    # 1. Basic grayscale with OTSU thresholding
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(("otsu", thresh1))
    
    # 2. Bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh2 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(("bilateral", thresh2))
    
    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(("clahe", thresh3))
    
    # 4. Adaptive thresholding which may work better for some reports
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    preprocessed_images.append(("adaptive", adaptive))
    
    # 5. Combination of techniques: bilateral + CLAHE + adaptive
    bilateral_enhanced = clahe.apply(bilateral)
    adaptive2 = cv2.adaptiveThreshold(
        bilateral_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    preprocessed_images.append(("combined", adaptive2))
    
    # 6. Noise removal using morphological operations
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    preprocessed_images.append(("morphology", closing))
    
    return preprocessed_images, img

def extract_text_with_multiple_configs(image_bytes):
    """
    Extract text using multiple preprocessing techniques and OCR configurations
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        Dictionary with text results from different methods
    """
    # Preprocess the image with multiple techniques
    preprocessed_images, original = preprocess_image_enhanced(image_bytes)
    
    # Define different OCR configurations
    configs = [
        ("default", r'--oem 3 --psm 6'),
        ("single_column", r'--oem 3 --psm 4'),
        ("sparse", r'--oem 3 --psm 11'),
        ("table", r'--oem 3 --psm 6 -c preserve_interword_spaces=1'),
        ("table_with_digits", r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=0123456789.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-+()[]{}:;,/<>!@#$%^&*=_\| ')
    ]
    
    results = {}
    best_text = ""
    best_score = 0
    
    # Try different combinations of preprocessing and OCR configs
    for preprocess_name, preprocess_img in preprocessed_images:
        for config_name, config in configs:
            method_name = f"{preprocess_name}_{config_name}"
            
            try:
                # Convert to PIL Image for pytesseract
                pil_img = Image.fromarray(preprocess_img)
                
                # Extract text
                text = pytesseract.image_to_string(pil_img, config=config)
                
                # Calculate score based on length and key indicators
                score = len(text)
                
                # Check for lab report indicators
                indicators = ['test', 'reference', 'range', 'result', 'normal', 'value', 'parameter', 'unit', 
                              'blood', 'urine', 'serum', 'total', 'bilirubin', 'protein', 'albumin', 'globulin',
                              'report', 'laboratory', 'sgot', 'sgpt', 'alkaline', 'phosphatase', 'gamma']
                
                for indicator in indicators:
                    if indicator in text.lower():
                        score += 20  # Bonus for each indicator found
                
                # Additional bonus for numerical values with units
                unit_patterns = r'(\d+\.?\d*)\s*(mg\/dl|g\/dl|U\/L|IU\/L|mmol\/L|µmol\/L|ng\/ml|pg\/ml)'
                unit_matches = re.findall(unit_patterns, text, re.IGNORECASE)
                score += len(unit_matches) * 30
                
                # Store result
                results[method_name] = {
                    "text": text,
                    "score": score
                }
                
                # Update best text if this is the highest score
                if score > best_score:
                    best_score = score
                    best_text = text
                    
            except Exception as e:
                results[method_name] = {
                    "text": f"Error: {str(e)}",
                    "score": 0
                }
    
    # Sort methods by score for debugging purposes
    sorted_methods = sorted(results.keys(), key=lambda k: results[k]["score"], reverse=True)
    
    return {
        "best_text": best_text,
        "best_method": sorted_methods[0] if sorted_methods else "none",
        "best_score": best_score,
        "all_results": results,
        "sorted_methods": sorted_methods
    }

def parse_reference_range_enhanced(range_text: str) -> Dict[str, Any]:
    """
    Enhanced reference range parsing with more patterns
    
    Args:
        range_text: Text containing reference range
        
    Returns:
        Dictionary with min and max values if available
    """
    result = {"original": range_text.strip()}
    
    # Remove common OCR errors
    cleaned_text = range_text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
    
    # Common reference range patterns
    patterns = [
        # Standard ranges
        (r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', 'range'),  # 70-99
        (r'(\d+\.?\d*)\s*to\s*(\d+\.?\d*)', 'range'),  # 70 to 99
        (r'between\s*(\d+\.?\d*)\s*and\s*(\d+\.?\d*)', 'range'),  # between 70 and 99
        (r'(\d+\.?\d*)\s*–\s*(\d+\.?\d*)', 'range'),  # using en dash: 70–99
        (r'(\d+\.?\d*)\s*—\s*(\d+\.?\d*)', 'range'),  # using em dash: 70—99
        
        # Upper limits
        (r'<\s*(\d+\.?\d*)', 'upper_limit'),  # <100
        (r'less than\s*(\d+\.?\d*)', 'upper_limit'),  # less than 100
        (r'up to\s*(\d+\.?\d*)', 'upper_limit'),  # up to 100
        (r'≤\s*(\d+\.?\d*)', 'upper_limit'),  # ≤100
        
        # Lower limits
        (r'>\s*(\d+\.?\d*)', 'lower_limit'),  # >10
        (r'more than\s*(\d+\.?\d*)', 'lower_limit'),  # more than 10
        (r'at least\s*(\d+\.?\d*)', 'lower_limit'),  # at least 10
        (r'≥\s*(\d+\.?\d*)', 'lower_limit'),  # ≥10
        
        # Special formats with units
        (r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*(mg/dl|g/dl|U/L|IU/L)', 'range_with_unit'),  # 70-99 mg/dL
    ]
    
    for pattern, pattern_type in patterns:
        match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if match:
            if pattern_type == 'range' or pattern_type == 'range_with_unit':
                try:
                    result["min"] = float(match.group(1))
                    result["max"] = float(match.group(2))
                    result["type"] = "range"
                    if pattern_type == 'range_with_unit' and len(match.groups()) > 2:
                        result["unit"] = match.group(3)
                    return result
                except ValueError:
                    # Continue to next pattern if conversion fails
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
    
    # Check for single values that might be reference points
    single_value_pattern = r'(?<![0-9.])(\d+\.?\d*)(?![0-9.])'
    single_values = re.findall(single_value_pattern, cleaned_text)
    if len(single_values) == 1:
        try:
            value = float(single_values[0])
            # Heuristic: if there's a less than symbol in the text, it's an upper limit
            if any(s in cleaned_text.lower() for s in ["<", "less", "below", "under", "not more than"]):
                result["max"] = value
                result["type"] = "upper_limit"
                return result
            # If there's a greater than symbol, it's a lower limit
            elif any(s in cleaned_text.lower() for s in [">", "more", "above", "over", "not less than"]):
                result["min"] = value
                result["type"] = "lower_limit"
                return result
            # Otherwise assume it's a reference value (equals)
            else:
                result["equals"] = value
                result["type"] = "equals"
                return result
        except ValueError:
            pass
    
    # If no pattern matches, return just the original text
    result["type"] = "unknown"
    return result

def check_value_within_range_enhanced(value: float, range_info: Dict[str, Any]) -> bool:
    """
    Enhanced check if a value is within the given reference range
    
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
        # Allow small deviation for equals (e.g. 5.0 equals 5.0±0.5)
        deviation = range_info["equals"] * 0.05  # 5% deviation
        return abs(value - range_info["equals"]) > deviation
    
    # Default to False if range can't be determined
    return False

def split_test_name_from_value(text: str) -> Tuple[str, str, str]:
    """
    Split a text line into test name, value, and reference range
    
    Args:
        text: Line containing test information
        
    Returns:
        Tuple of (test_name, value_text, reference_range_text)
    """
    # Find numbers in the text
    number_matches = list(re.finditer(r'(\d+\.?\d*)', text))
    
    if not number_matches:
        return text, "", ""
    
    # Assume the first number is the test value
    first_number_match = number_matches[0]
    test_name = text[:first_number_match.start()].strip()
    value_text = first_number_match.group()
    
    # Assume the reference range starts after the first number
    reference_range_text = text[first_number_match.end():].strip()
    
    # Common prefixes for reference ranges
    range_prefixes = ["normal", "range", "reference", "interval", "(", "["]
    
    # Try to find a better split point for the reference range
    for prefix in range_prefixes:
        prefix_pos = reference_range_text.lower().find(prefix)
        if prefix_pos > 0:
            # Split at this position and update value_text and reference_range_text
            value_part = reference_range_text[:prefix_pos].strip()
            reference_range_text = reference_range_text[prefix_pos:].strip()
            
            # Check if there are additional numbers in value_part that should be part of value_text
            additional_values = re.findall(r'(\d+\.?\d*)', value_part)
            if additional_values:
                value_text += " " + value_part
            break
    
    return test_name, value_text, reference_range_text

def extract_lab_tests_enhanced(text: str) -> List[Dict[str, Any]]:
    """
    Enhanced lab test extraction with more sophisticated patterns
    
    Args:
        text: Text extracted from lab report image
        
    Returns:
        List of dictionaries containing lab test information
    """
    lab_tests = []
    
    # Split text into lines and clean them
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line:
            # Fix common OCR errors
            line = (line.replace('O', '0').replace('o', '0')
                       .replace('l', '1').replace('I', '1')
                       .replace('S', '5').replace('B', '8'))
            lines.append(line)
    
    # Combine short lines that might be split test results
    combined_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # If this line is short and doesn't contain a number, try to combine with next line
        if len(line) < 15 and not re.search(r'\d', line) and i + 1 < len(lines):
            combined_lines.append(line + " " + lines[i + 1])
            i += 2
        else:
            combined_lines.append(line)
            i += 1
    
    # Enhanced patterns for different lab report formats
    patterns = [
        # Format: Test name followed by value and reference range
        r'([A-Za-z\s\(\)\/\-\+]{2,})\s+(\d+\.?\d*)\s*([A-Za-z0-9\/\.\-\s\<\>\=\±]+)',
        
        # Format: Test name with value and units
        r'([A-Za-z\s\(\)\/\-\+]{2,})\s+(\d+\.?\d*)\s*([A-Za-z\/]+)',
        
        # Format: Test code, test name, value, reference range
        r'([A-Z0-9]+)\s+([A-Za-z\s\(\)\/\-\+]{2,})\s+(\d+\.?\d*)',
        
        # Format: Just test name and value
        r'([A-Za-z\s\(\)\/\-\+]{2,})\s+(\d+\.?\d*)',
    ]
    
    # Process each line
    for line in combined_lines:
        # Skip lines that look like headers or footers
        if any(header in line.lower() for header in ["patient", "doctor", "date", "report", "laboratory", "page", "ref", "specimen"]):
            continue
            
        # Skip lines that are too short
        if len(line) < 5:
            continue
        
        found_match = False
        
        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                found_match = True
                
                try:
                    # Extract test name and value based on the pattern
                    if len(match.groups()) >= 3:
                        test_name = match.group(1).strip()
                        test_value = float(match.group(2).strip())
                        reference_range = match.group(3).strip()
                    else:
                        test_name = match.group(1).strip()
                        test_value = float(match.group(2).strip())
                        reference_range = line[match.end():].strip()
                    
                    # Additional validation: Skip suspicious test names
                    if len(test_name) < 2 or test_name.isdigit():
                        continue
                        
                    # Parse reference range
                    range_info = parse_reference_range_enhanced(reference_range)
                    
                    # Check if value is out of range
                    out_of_range = check_value_within_range_enhanced(test_value, range_info)
                    
                    lab_test = {
                        "test_name": test_name,
                        "test_value": test_value,
                        "bio_reference_range": reference_range,
                        "lab_test_out_of_range": out_of_range
                    }
                    
                    lab_tests.append(lab_test)
                    break
                except (ValueError, IndexError):
                    # Continue to next pattern if extraction fails
                    continue
        
        # If no pattern matched, try manual splitting
        if not found_match and re.search(r'\d+\.?\d*', line):
            test_name, value_text, reference_range = split_test_name_from_value(line)
            
            if test_name and value_text:
                try:
                    test_value = float(value_text)
                    
                    # Skip suspicious test names
                    if len(test_name) < 2 or test_name.isdigit():
                        continue
                    
                    # Parse reference range
                    range_info = parse_reference_range_enhanced(reference_range)
                    
                    # Check if value is out of range
                    out_of_range = check_value_within_range_enhanced(test_value, range_info)
                    
                    lab_test = {
                        "test_name": test_name,
                        "test_value": test_value,
                        "bio_reference_range": reference_range,
                        "lab_test_out_of_range": out_of_range
                    }
                    
                    lab_tests.append(lab_test)
                except ValueError:
                    continue
    
    # Remove duplicates by test name
    unique_tests = {}
    for test in lab_tests:
        test_name = test["test_name"].lower()
        
        # Keep the test with the longest name if there are duplicates
        if test_name not in unique_tests or len(test["test_name"]) > len(unique_tests[test_name]["test_name"]):
            unique_tests[test_name] = test
    
    return list(unique_tests.values())

def process_lab_report_enhanced(image_bytes) -> Dict[str, Any]:
    """
    Process a lab report image with enhanced OCR and extraction
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        Dictionary with extracted lab test data
    """
    start_time = time.time()
    
    try:
        # Extract text with multiple OCR configurations and select the best
        extraction_result = extract_text_with_multiple_configs(image_bytes)
        best_text = extraction_result["best_text"]
        
        # Extract lab tests from the best text
        lab_tests = extract_lab_tests_enhanced(best_text)
        
        # Count tests that are out of range
        tests_out_of_range = sum(1 for test in lab_tests if test["lab_test_out_of_range"])
        
        result = {
            "is_success": True,
            "extraction_method": extraction_result["best_method"],
            "extraction_score": extraction_result["best_score"],
            "extraction_time": time.time() - start_time,
            "data": lab_tests,
            "tests_count": len(lab_tests),
            "abnormal_count": tests_out_of_range
        }
        
        return result
    except Exception as e:
        return {
            "is_success": False,
            "error": str(e),
            "extraction_time": time.time() - start_time
        } 