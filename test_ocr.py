import os
import cv2
import pytesseract
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from lab_report_processor import preprocess_image, extract_text_from_image, extract_lab_tests

def display_image_and_text(image_path):
    """
    Display the original image, preprocessed image, and extracted text
    
    Args:
        image_path: Path to the lab report image
    """
    # Read the image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Get the original image
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    
    # Preprocess the image
    preprocessed_img = preprocess_image(image_bytes)
    
    # Extract text
    text = extract_text_from_image(image_bytes)
    
    # Extract lab tests
    lab_tests = extract_lab_tests(text)
    
    # Display original and preprocessed images
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 1, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.title("Preprocessed Image")
    plt.imshow(preprocessed_img, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = "ocr_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"ocr_images_{os.path.basename(image_path)}.png"))
    
    # Save the extracted text
    with open(os.path.join(output_dir, f"ocr_text_{os.path.basename(image_path)}.txt"), 'w', encoding='utf-8') as f:
        f.write("EXTRACTED TEXT:\n\n")
        f.write(text)
        f.write("\n\n")
        f.write("EXTRACTED LAB TESTS:\n\n")
        for test in lab_tests:
            f.write(f"Test Name: {test['test_name']}\n")
            f.write(f"Test Value: {test['test_value']}\n")
            f.write(f"Reference Range: {test['bio_reference_range']}\n")
            f.write(f"Out of Range: {test['lab_test_out_of_range']}\n")
            f.write("-" * 50 + "\n")
    
    # Print summary
    print(f"Processed {image_path}")
    print(f"Found {len(lab_tests)} lab tests")
    print(f"Results saved to {output_dir} directory")
    print("\nExamples of detected lab tests:")
    
    # Display a few examples of detected tests
    for i, test in enumerate(lab_tests[:3]):
        print(f"{i+1}. {test['test_name']}: {test['test_value']} ({test['bio_reference_range']}) - " + 
              ("OUT OF RANGE" if test['lab_test_out_of_range'] else "Normal"))
    
    if len(lab_tests) > 3:
        print(f"... and {len(lab_tests) - 3} more")

def test_directory(directory_path):
    """
    Process all images in a directory
    
    Args:
        directory_path: Path to directory containing lab report images
    """
    print(f"Testing OCR on images in {directory_path}")
    
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
    
    # Process each image
    for image_path in image_files:
        display_image_and_text(image_path)
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OCR on lab report images")
    parser.add_argument("--image", help="Path to a single lab report image")
    parser.add_argument("--dir", help="Path to directory containing lab report images")
    
    args = parser.parse_args()
    
    if args.image:
        display_image_and_text(args.image)
    elif args.dir:
        test_directory(args.dir)
    else:
        # Default to data directory if it exists
        if os.path.exists("data"):
            test_directory("data")
        else:
            print("Please specify an image file (--image) or directory (--dir)")
            print("Alternatively, download sample data using download_sample_data.py") 