import argparse
import os
import subprocess
import time

def run_ocr_comparison(image_path):
    """
    Run OCR comparison on an image
    
    Args:
        image_path: Path to the image to analyze
        
    Returns:
        Output directory containing results
    """
    print(f"Running OCR comparison on {image_path}...")
    
    # Run compare_ocr.py on the image
    cmd = f"python compare_ocr.py --image {image_path}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for process to complete
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print("Error running OCR comparison:")
        print(stderr.decode())
        return None
    
    # Extract the output directory from the output
    stdout_text = stdout.decode()
    for line in stdout_text.split('\n'):
        if "Results saved to" in line:
            # Extract the directory path
            output_dir = line.split("Results saved to")[1].strip()
            return output_dir
            
    # Default to looking for the most recent directory
    results_dirs = [d for d in os.listdir() if d.startswith("ocr_comparison_results_")]
    if results_dirs:
        # Sort by creation time
        return sorted(results_dirs, reverse=True)[0]
    
    return None

def run_enhanced_output(results_dir, query=None, output_format="text"):
    """
    Generate enhanced output from OCR comparison results
    
    Args:
        results_dir: Directory containing OCR comparison results
        query: Optional query for RAG analysis
        output_format: Output format (text, markdown, html)
        
    Returns:
        None
    """
    print(f"Generating enhanced output from {results_dir}...")
    
    # Create command with optional query
    cmd = f"python enhance_ocr_output.py --results-dir {results_dir} --format {output_format}"
    if query:
        cmd += f" --query \"{query}\""
        
    # Run the command
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for process to complete
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print("Error generating enhanced output:")
        print(stderr.decode())
    else:
        print(stdout.decode())

def main():
    parser = argparse.ArgumentParser(description="Run full OCR analysis pipeline")
    parser.add_argument("--image", required=True, help="Path to lab report image to analyze")
    parser.add_argument("--query", help="Optional query for RAG analysis")
    parser.add_argument("--format", choices=["text", "markdown", "html"], default="text",
                        help="Output format (default: text)")
    args = parser.parse_args()
    
    # Step 1: Run OCR comparison
    results_dir = run_ocr_comparison(args.image)
    
    if results_dir:
        print(f"OCR comparison completed. Results in {results_dir}")
        
        # Step 2: Generate enhanced output
        time.sleep(1)  # Small delay to ensure files are fully written
        run_enhanced_output(results_dir, args.query, args.format)
    else:
        print("OCR comparison failed.")

if __name__ == "__main__":
    main() 