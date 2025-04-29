import argparse
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime

from lab_report_processor import (
    preprocess_image, 
    extract_text_from_image, 
    extract_lab_tests, 
    process_lab_report
)

def analyze_single_image(image_path, output_dir="single_image_results"):
    """
    Perform detailed analysis on a single lab report image
    
    Args:
        image_path: Path to the lab report image
        output_dir: Directory to save results
    """
    print(f"Analyzing lab report image: {image_path}")
    
    # Create output directory
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
    
    # Process image with multiple approaches
    opening, adaptive = preprocess_image(image_bytes)
    
    # Extract text from image
    text = extract_text_from_image(image_bytes)
    
    # Extract lab tests
    lab_tests = extract_lab_tests(text)
    
    # Save extraction results
    result = {
        "image_path": image_path,
        "extraction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text_length": len(text),
        "tests_found": len(lab_tests),
        "tests_data": lab_tests,
        "tests_out_of_range": sum(1 for test in lab_tests if test['lab_test_out_of_range'])
    }
    
    # Save result as JSON
    with open(os.path.join(output_dir, "extraction_result.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    # Save extracted text
    with open(os.path.join(output_dir, "extracted_text.txt"), 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Display and save visualizations
    plt.figure(figsize=(20, 15))
    
    # Display original and preprocessed images
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("Preprocessed Image (Method 1)")
    plt.imshow(opening, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title("Preprocessed Image (Method 2)")
    plt.imshow(adaptive, cmap='gray')
    plt.axis('off')
    
    # Create a summary of extracted tests
    plt.subplot(2, 2, 4)
    plt.title("Extracted Lab Tests")
    plt.axis('off')
    test_text = "\n".join([
        f"{i+1}. {test['test_name']}: {test['test_value']} " + 
        f"(Reference: {test['bio_reference_range']}) " +
        ("❌ OUT OF RANGE" if test['lab_test_out_of_range'] else "✓ Normal")
        for i, test in enumerate(lab_tests)
    ])
    
    if not test_text:
        test_text = "No lab tests detected"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, test_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props, wrap=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "analysis_results.png"))
    
    # Create a standalone table of test results
    fig, ax = plt.subplots(figsize=(10, len(lab_tests) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    if lab_tests:
        table_data = [[test['test_name'], 
                      test['test_value'], 
                      test['bio_reference_range'], 
                      "Out of Range" if test['lab_test_out_of_range'] else "Normal"] 
                     for test in lab_tests]
        
        row_colors = []
        for test in lab_tests:
            if test['lab_test_out_of_range']:
                row_colors.append(['#ffcccc'] * 4)  # Light red for out of range
            else:
                row_colors.append(['#ccffcc'] * 4)  # Light green for normal
                
        table = ax.table(cellText=table_data, colLabels=['Test Name', 'Value', 'Reference Range', 'Status'],
                         loc='center', cellLoc='center', cellColours=row_colors)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        plt.title(f"Lab Tests Found: {len(lab_tests)}")
    else:
        plt.title("No lab tests found in the image")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_results_table.png"))
    
    # Print summary
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print(f"Found {len(lab_tests)} lab tests")
    print(f"Tests out of normal range: {result['tests_out_of_range']}")
    
    # Show a few examples
    if lab_tests:
        print("\nExamples of detected tests:")
        for i, test in enumerate(lab_tests[:5]):  # Show up to 5 examples
            print(f"{i+1}. {test['test_name']}: {test['test_value']} " + 
                  f"(Reference: {test['bio_reference_range']}) " +
                  ("OUT OF RANGE" if test['lab_test_out_of_range'] else "Normal"))
        
        if len(lab_tests) > 5:
            print(f"... and {len(lab_tests) - 5} more")
    
    return output_dir, result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OCR and data extraction on a single lab report image")
    parser.add_argument("--image", required=True, help="Path to the lab report image")
    parser.add_argument("--output", default="single_image_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found")
    else:
        output_dir, _ = analyze_single_image(args.image, args.output) 