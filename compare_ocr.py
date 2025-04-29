import argparse
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import difflib

# Import our OCR implementations
from lab_report_processor import process_lab_report
from easyocr_processor import process_lab_report_with_easyocr

def compare_ocr_methods(image_path, output_dir="ocr_comparison_results"):
    """
    Compare Tesseract OCR and EasyOCR on the same lab report image
    
    Args:
        image_path: Path to the lab report image
        output_dir: Directory to save comparison results
    """
    print(f"Comparing OCR methods on: {image_path}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Get the original image for display
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    
    # Process with Tesseract OCR
    print("Processing with Tesseract OCR...")
    tesseract_start_time = time.time()
    tesseract_result = process_lab_report(image_bytes)
    tesseract_time = time.time() - tesseract_start_time
    
    # Process with EasyOCR
    print("Processing with EasyOCR...")
    easyocr_result = process_lab_report_with_easyocr(image_bytes)
    
    # Save results to JSON
    with open(os.path.join(output_dir, "tesseract_result.json"), 'w', encoding='utf-8') as f:
        json.dump(tesseract_result, f, indent=2)
    
    with open(os.path.join(output_dir, "easyocr_result.json"), 'w', encoding='utf-8') as f:
        json.dump(easyocr_result, f, indent=2)
    
    # Extract data for comparison
    tesseract_success = tesseract_result.get("is_success", False)
    easyocr_success = easyocr_result.get("is_success", False)
    
    tesseract_tests = tesseract_result.get("data", []) if tesseract_success else []
    easyocr_tests = easyocr_result.get("data", []) if easyocr_success else []
    
    tesseract_text = ""
    easyocr_text = easyocr_result.get("raw_text", "") if easyocr_success else ""
    
    # Get Tesseract timing
    tesseract_timing = {
        "total_time": tesseract_time
    }
    
    # Get EasyOCR timing
    easyocr_timing = easyocr_result.get("timing", {}) if easyocr_success else {}
    
    # Compare results
    comparison = {
        "image_path": image_path,
        "comparison_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tesseract": {
            "success": tesseract_success,
            "test_count": len(tesseract_tests),
            "abnormal_count": sum(1 for test in tesseract_tests if test.get("lab_test_out_of_range")),
            "timing": tesseract_timing
        },
        "easyocr": {
            "success": easyocr_success,
            "test_count": len(easyocr_tests),
            "abnormal_count": sum(1 for test in easyocr_tests if test.get("lab_test_out_of_range")),
            "timing": easyocr_timing
        }
    }
    
    # Compare test results - find tests in both results
    common_tests = []
    tesseract_only = []
    easyocr_only = []
    
    # Create dictionaries of tests by name for easier comparison
    tesseract_dict = {test.get("test_name", "").lower(): test for test in tesseract_tests}
    easyocr_dict = {test.get("test_name", "").lower(): test for test in easyocr_tests}
    
    # Find common and exclusive tests
    for name, test in tesseract_dict.items():
        if name in easyocr_dict:
            match = {
                "test_name": test.get("test_name"),
                "tesseract_value": test.get("test_value"),
                "easyocr_value": easyocr_dict[name].get("test_value"),
                "tesseract_range": test.get("bio_reference_range"),
                "easyocr_range": easyocr_dict[name].get("bio_reference_range"),
                "tesseract_abnormal": test.get("lab_test_out_of_range"),
                "easyocr_abnormal": easyocr_dict[name].get("lab_test_out_of_range"),
                "value_match": test.get("test_value") == easyocr_dict[name].get("test_value")
            }
            common_tests.append(match)
        else:
            tesseract_only.append(test)
    
    for name, test in easyocr_dict.items():
        if name not in tesseract_dict:
            easyocr_only.append(test)
    
    comparison["common_tests"] = common_tests
    comparison["tesseract_only"] = tesseract_only
    comparison["easyocr_only"] = easyocr_only
    comparison["agreement_rate"] = len(common_tests) / max(1, len(tesseract_tests) + len(easyocr_tests) - len(common_tests))
    comparison["value_match_rate"] = sum(1 for test in common_tests if test["value_match"]) / max(1, len(common_tests))
    
    # Save comparison
    with open(os.path.join(output_dir, "comparison.json"), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    
    # Generate visual comparison
    plt.figure(figsize=(15, 10))
    
    # Display original image
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis('off')
    
    # Display comparison metrics
    plt.subplot(2, 2, 2)
    plt.title("OCR Comparison")
    plt.axis('off')
    
    metrics_text = (
        f"Tesseract: {len(tesseract_tests)} tests found in {tesseract_timing.get('total_time', 0):.2f}s\n"
        f"EasyOCR: {len(easyocr_tests)} tests found in {easyocr_timing.get('total_time', 0):.2f}s\n\n"
        f"Agreement rate: {comparison['agreement_rate']*100:.1f}%\n"
        f"Value match rate: {comparison['value_match_rate']*100:.1f}%\n\n"
        f"Common tests: {len(common_tests)}\n"
        f"Tesseract only: {len(tesseract_only)}\n"
        f"EasyOCR only: {len(easyocr_only)}"
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Timing comparison
    plt.subplot(2, 2, 3)
    plt.title("Processing Time (seconds)")
    tesseract_time_val = tesseract_timing.get('total_time', 0)
    easyocr_time_val = easyocr_timing.get('total_time', 0)
    
    plt.bar(['Tesseract', 'EasyOCR'], [tesseract_time_val, easyocr_time_val])
    plt.ylabel('Seconds')
    
    # Test count comparison
    plt.subplot(2, 2, 4)
    plt.title("Tests Detected")
    plt.bar(['Tesseract', 'EasyOCR'], [len(tesseract_tests), len(easyocr_tests)])
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ocr_comparison.png"))
    
    # Also create a detailed table comparing the common tests
    if common_tests:
        fig, ax = plt.subplots(figsize=(12, len(common_tests) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        for test in common_tests:
            row = [
                test["test_name"],
                f"{test['tesseract_value']} / {test['easyocr_value']}",
                "✓" if test["value_match"] else "✗",
                f"{test['tesseract_range']} / {test['easyocr_range']}",
                f"{test['tesseract_abnormal']} / {test['easyocr_abnormal']}"
            ]
            table_data.append(row)
        
        # Set row colors based on value match
        row_colors = []
        for test in common_tests:
            if test["value_match"]:
                row_colors.append(['#ccffcc'] * 5)  # Light green for matching values
            else:
                row_colors.append(['#ffcccc'] * 5)  # Light red for mismatched values
                
        table = ax.table(cellText=table_data, 
                      colLabels=['Test Name', 'Tesseract / EasyOCR Value', 'Match', 
                                'Tesseract / EasyOCR Range', 'Abnormal (T/E)'],
                      loc='center', cellLoc='center', cellColours=row_colors)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        plt.title("Common Tests Comparison")
        plt.savefig(os.path.join(output_dir, "common_tests_comparison.png"))
    
    # Generate text file with the report
    with open(os.path.join(output_dir, "comparison_report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"OCR Comparison Report for {os.path.basename(image_path)}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Summary:\n")
        f.write(f"- Tesseract found {len(tesseract_tests)} tests in {tesseract_timing.get('total_time', 0):.2f} seconds\n")
        f.write(f"- EasyOCR found {len(easyocr_tests)} tests in {easyocr_timing.get('total_time', 0):.2f} seconds\n")
        f.write(f"- Agreement rate: {comparison['agreement_rate']*100:.1f}%\n")
        f.write(f"- Value match rate: {comparison['value_match_rate']*100:.1f}%\n\n")
        
        f.write("Common Tests:\n")
        if common_tests:
            for i, test in enumerate(common_tests):
                f.write(f"{i+1}. {test['test_name']}\n")
                f.write(f"   - Tesseract value: {test['tesseract_value']} (Range: {test['tesseract_range']})\n")
                f.write(f"   - EasyOCR value: {test['easyocr_value']} (Range: {test['easyocr_range']})\n")
                f.write(f"   - Value match: {'✓' if test['value_match'] else '✗'}\n\n")
        else:
            f.write("No common tests found.\n\n")
        
        f.write("Tests found only by Tesseract:\n")
        if tesseract_only:
            for i, test in enumerate(tesseract_only):
                f.write(f"{i+1}. {test['test_name']}: {test['test_value']} (Range: {test['bio_reference_range']})\n")
        else:
            f.write("None\n\n")
        
        f.write("\nTests found only by EasyOCR:\n")
        if easyocr_only:
            for i, test in enumerate(easyocr_only):
                f.write(f"{i+1}. {test['test_name']}: {test['test_value']} (Range: {test['bio_reference_range']})\n")
        else:
            f.write("None\n")
    
    print(f"\nComparison complete! Results saved to {output_dir}")
    print(f"- Tesseract found {len(tesseract_tests)} tests in {tesseract_timing.get('total_time', 0):.2f} seconds")
    print(f"- EasyOCR found {len(easyocr_tests)} tests in {easyocr_timing.get('total_time', 0):.2f} seconds")
    print(f"- Agreement rate: {comparison['agreement_rate']*100:.1f}%")
    print(f"- Value match rate: {comparison['value_match_rate']*100:.1f}%")
    
    return output_dir, comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Tesseract OCR and EasyOCR on lab report images")
    parser.add_argument("--image", required=True, help="Path to the lab report image")
    parser.add_argument("--output", default="ocr_comparison_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found")
    else:
        output_dir, _ = compare_ocr_methods(args.image, args.output) 