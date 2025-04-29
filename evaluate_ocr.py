import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from lab_report_processor import extract_text_from_image, extract_lab_tests
import numpy as np
import time
import json
from collections import Counter
from datetime import datetime

def evaluate_ocr_performance(directory_path, output_dir="ocr_evaluation"):
    """
    Evaluate OCR performance on a directory of lab report images
    
    Args:
        directory_path: Path to directory containing lab report images
        output_dir: Directory to save evaluation results
    """
    print(f"Evaluating OCR performance on images in {directory_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    image_files = []
    
    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory_path, filename))
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Prepare results data
    results = []
    all_test_names = []
    all_reference_ranges = []
    processing_times = []
    
    # Process each image
    for image_path in image_files:
        print(f"Processing {image_path}...")
        
        # Read the image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Time the OCR process
        start_time = time.time()
        text = extract_text_from_image(image_bytes)
        ocr_time = time.time() - start_time
        
        # Time the lab test extraction process
        start_time = time.time()
        lab_tests = extract_lab_tests(text)
        extraction_time = time.time() - start_time
        
        # Record processing times
        processing_times.append({
            'image': os.path.basename(image_path),
            'ocr_time_sec': ocr_time,
            'extraction_time_sec': extraction_time,
            'total_time_sec': ocr_time + extraction_time
        })
        
        # If no tests found, add a placeholder result
        if not lab_tests:
            results.append({
                'image': os.path.basename(image_path),
                'tests_found': 0,
                'tests_out_of_range': 0,
                'success': False
            })
            continue
        
        # Collect test names and reference ranges
        for test in lab_tests:
            all_test_names.append(test['test_name'])
            all_reference_ranges.append(test['bio_reference_range'])
        
        # Count out-of-range tests
        out_of_range_count = sum(1 for test in lab_tests if test['lab_test_out_of_range'])
        
        # Add results
        results.append({
            'image': os.path.basename(image_path),
            'tests_found': len(lab_tests),
            'tests_out_of_range': out_of_range_count,
            'success': True
        })
        
        # Save raw test data for the image
        with open(os.path.join(output_dir, f"{os.path.basename(image_path)}_tests.json"), 'w') as f:
            json.dump({
                'image': os.path.basename(image_path),
                'tests': lab_tests
            }, f, indent=2)
    
    # Create results summary DataFrame
    results_df = pd.DataFrame(results)
    processing_times_df = pd.DataFrame(processing_times)
    
    # Calculate statistics
    success_rate = results_df['success'].mean() * 100
    avg_tests_found = results_df['tests_found'].mean()
    avg_tests_out_of_range = results_df['tests_out_of_range'].mean()
    avg_ocr_time = processing_times_df['ocr_time_sec'].mean()
    avg_extraction_time = processing_times_df['extraction_time_sec'].mean()
    avg_total_time = processing_times_df['total_time_sec'].mean()
    
    # Count test name occurrences
    test_name_counts = Counter(all_test_names)
    common_tests = test_name_counts.most_common(10)
    
    # Generate report
    report = f"""# OCR Performance Evaluation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total images processed: {len(image_files)}
- Success rate: {success_rate:.2f}%
- Average tests found per image: {avg_tests_found:.2f}
- Average tests out of range per image: {avg_tests_out_of_range:.2f}
- Average OCR processing time: {avg_ocr_time:.2f} sec
- Average extraction processing time: {avg_extraction_time:.2f} sec
- Average total processing time: {avg_total_time:.2f} sec

## Most Common Test Names
"""
    
    for test_name, count in common_tests:
        report += f"- {test_name}: {count} occurrences\n"
    
    # Save the report
    with open(os.path.join(output_dir, "evaluation_report.md"), 'w') as f:
        f.write(report)
    
    # Save detailed results
    results_df.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)
    processing_times_df.to_csv(os.path.join(output_dir, "processing_times.csv"), index=False)
    
    # Plot statistics
    plt.figure(figsize=(12, 10))
    
    # Tests found distribution
    plt.subplot(2, 2, 1)
    plt.hist(results_df['tests_found'], bins=10)
    plt.title('Tests Found Distribution')
    plt.xlabel('Number of Tests')
    plt.ylabel('Frequency')
    
    # Processing times
    plt.subplot(2, 2, 2)
    plt.bar(['OCR', 'Extraction', 'Total'], 
            [avg_ocr_time, avg_extraction_time, avg_total_time])
    plt.title('Average Processing Times')
    plt.ylabel('Time (seconds)')
    
    # Common tests pie chart (top 5)
    plt.subplot(2, 2, 3)
    top_tests = dict(test_name_counts.most_common(5))
    plt.pie(top_tests.values(), labels=top_tests.keys(), autopct='%1.1f%%')
    plt.title('Top 5 Most Common Tests')
    
    # Success vs. Failure
    plt.subplot(2, 2, 4)
    success_counts = results_df['success'].value_counts()
    plt.pie(success_counts.values, labels=['Success', 'Failure'] if len(success_counts) > 1 else ['Success'], 
            autopct='%1.1f%%', colors=['green', 'red'] if len(success_counts) > 1 else ['green'])
    plt.title('Success Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_plots.png"))
    
    print(f"Evaluation complete! Report saved to {output_dir}/evaluation_report.md")
    print(f"See {output_dir} directory for detailed results and visualizations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OCR performance on lab report images")
    parser.add_argument("--dir", help="Path to directory containing lab report images")
    parser.add_argument("--output", default="ocr_evaluation", help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    if args.dir:
        evaluate_ocr_performance(args.dir, args.output)
    else:
        # Default to data directory if it exists
        if os.path.exists("data"):
            evaluate_ocr_performance("data")
        else:
            print("Please specify a directory containing lab report images (--dir)")
            print("Alternatively, download sample data using download_sample_data.py") 