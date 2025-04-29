import cv2
import numpy as np
import easyocr
import re
from typing import Dict, List, Any, Tuple, Optional
import time

# Initialize EasyOCR reader (only do this once to save time)
reader = None

def get_reader(languages=['en']):
    """
    Get or initialize EasyOCR reader
    
    Args:
        languages: List of languages to detect
        
    Returns:
        EasyOCR reader instance
    """
    global reader
    if reader is None:
        print("Initializing EasyOCR reader (this may take a moment)...")
        reader = easyocr.Reader(languages, gpu=False)
    return reader

def preprocess_image_for_easyocr(image_bytes):
    """
    Preprocess image for better EasyOCR results
    
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
    
    # Apply CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced, img

def extract_text_with_easyocr(image_bytes, languages=['en']):
    """
    Extract text from an image using EasyOCR
    
    Args:
        image_bytes: Binary image data
        languages: List of languages to detect
        
    Returns:
        Tuple of (extracted text, result objects, processing time)
    """
    start_time = time.time()
    
    # Get reader
    reader = get_reader(languages)
    
    # Preprocess the image
    preprocessed_img, original_img = preprocess_image_for_easyocr(image_bytes)
    
    # Perform OCR
    results = reader.readtext(preprocessed_img)
    
    # Combine text from results
    text = ""
    for (bbox, text_result, prob) in results:
        text += text_result + " "
        
        # Add newline after potential table row
        if bbox[1][0] - bbox[0][0] > 0.5 * original_img.shape[1]:
            text += "\n"
    
    processing_time = time.time() - start_time
    
    return text.strip(), results, processing_time

def extract_table_structure(results, original_img):
    """
    Attempt to extract table structure from EasyOCR results
    
    Args:
        results: EasyOCR results
        original_img: Original image
        
    Returns:
        List of rows with extracted cells
    """
    # Sort results by Y position (rows) then X position (columns)
    img_height = original_img.shape[0]
    row_threshold = img_height * 0.02  # 2% of image height is threshold for same row
    
    # Group results by rows based on Y position
    rows = []
    current_row = []
    last_y = -row_threshold * 2
    
    # Sort by top Y coordinate
    sorted_results = sorted(results, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)
    
    for result in sorted_results:
        bbox, text, prob = result
        
        # Calculate center Y of this text box
        current_y = (bbox[0][1] + bbox[2][1]) / 2
        
        # Check if this is a new row
        if current_y - last_y > row_threshold:
            if current_row:
                # Sort current row by X position
                current_row.sort(key=lambda x: x[0][0][0])
                rows.append(current_row)
            current_row = [result]
        else:
            current_row.append(result)
        
        last_y = current_y
    
    # Add the last row
    if current_row:
        current_row.sort(key=lambda x: x[0][0][0])
        rows.append(current_row)
    
    # Convert to text
    text_rows = []
    for row in rows:
        text_row = [text for (_, text, _) in row]
        text_rows.append(text_row)
    
    return text_rows

def extract_lab_tests_from_easyocr(table_rows):
    """
    Extract lab test data from table rows detected by EasyOCR
    
    Args:
        table_rows: List of rows with text cells
        
    Returns:
        List of dictionaries containing lab test information
    """
    lab_tests = []
    
    # Process each row
    for row in table_rows:
        # Skip rows that are too short
        if len(row) < 2:
            continue
        
        # Try to identify test name, value and reference range
        test_name = None
        test_value = None
        reference_range = None
        
        # For rows with exactly 3-4 columns, assume test name, value, reference range
        if 3 <= len(row) <= 4:
            potential_test_name = row[0]
            potential_value = row[1]
            potential_range = row[2] if len(row) >= 3 else ""
            
            # Check if second column is a number (test value)
            try:
                test_value = float(potential_value.replace(',', '.'))
                test_name = potential_test_name
                reference_range = potential_range
            except ValueError:
                # Try looking at other columns for the value
                for i, cell in enumerate(row):
                    if i == 0:  # Skip first column which is likely the test name
                        continue
                    try:
                        # Try to extract numbers from the cell
                        number_match = re.search(r'(\d+\.?\d*)', cell)
                        if number_match:
                            test_value = float(number_match.group(1))
                            test_name = potential_test_name
                            # Use remaining text as reference range
                            if i < len(row) - 1:
                                reference_range = " ".join(row[i+1:])
                            break
                    except ValueError:
                        continue
        
        # If we can't identify the structure, try regex pattern matching
        if test_name is None or test_value is None:
            # Join row text
            row_text = " ".join(row)
            
            # Try pattern matching
            # Pattern: Test name followed by value and optional reference range
            pattern = r'([A-Za-z\s\(\)\/\-\+]{2,})\s*(\d+\.?\d*)\s*([A-Za-z0-9\/\.\-\s\<\>\=\±]*)'
            match = re.search(pattern, row_text)
            
            if match:
                test_name = match.group(1).strip()
                try:
                    test_value = float(match.group(2).strip())
                    reference_range = match.group(3).strip() if match.group(3) else ""
                except ValueError:
                    continue
        
        # If we found a test, add it
        if test_name and test_value is not None:
            # Parse the reference range
            is_out_of_range = False
            if reference_range:
                # Look for range patterns like "10-20" or "<10" or ">20"
                range_match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', reference_range)
                if range_match:
                    min_val = float(range_match.group(1))
                    max_val = float(range_match.group(2))
                    is_out_of_range = test_value < min_val or test_value > max_val
                elif '<' in reference_range:
                    max_match = re.search(r'<\s*(\d+\.?\d*)', reference_range)
                    if max_match:
                        max_val = float(max_match.group(1))
                        is_out_of_range = test_value >= max_val
                elif '>' in reference_range:
                    min_match = re.search(r'>\s*(\d+\.?\d*)', reference_range)
                    if min_match:
                        min_val = float(min_match.group(1))
                        is_out_of_range = test_value <= min_val
            
            lab_test = {
                "test_name": test_name,
                "test_value": test_value,
                "bio_reference_range": reference_range if reference_range else "Not specified",
                "lab_test_out_of_range": is_out_of_range
            }
            
            lab_tests.append(lab_test)
    
    # Remove duplicates
    unique_tests = {}
    for test in lab_tests:
        name = test["test_name"].lower()
        if name not in unique_tests:
            unique_tests[name] = test
    
    return list(unique_tests.values())

def process_lab_report_with_easyocr(image_bytes):
    """
    Process lab report image using EasyOCR
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        Dictionary with extraction results and timing information
    """
    try:
        # Start timing
        total_start_time = time.time()
        
        # Extract text using EasyOCR
        text, results, ocr_time = extract_text_with_easyocr(image_bytes)
        
        # Get the original image for table extraction
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract table structure
        table_extraction_start = time.time()
        table_rows = extract_table_structure(results, original_img)
        table_extraction_time = time.time() - table_extraction_start
        
        # Extract lab tests
        test_extraction_start = time.time()
        lab_tests = extract_lab_tests_from_easyocr(table_rows)
        test_extraction_time = time.time() - test_extraction_start
        
        # If no structured data, fallback to our regex patterns
        if not lab_tests:
            from lab_report_processor import extract_lab_tests
            lab_tests = extract_lab_tests(text)
        
        total_time = time.time() - total_start_time
        
        return {
            "is_success": True,
            "data": lab_tests,
            "timing": {
                "ocr_time": ocr_time,
                "table_extraction_time": table_extraction_time,
                "test_extraction_time": test_extraction_time,
                "total_time": total_time
            },
            "raw_text": text
        }
    except Exception as e:
        return {
            "is_success": False,
            "error": str(e)
        } 