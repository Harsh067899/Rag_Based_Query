import requests
import os
import json
import sys
from pprint import pprint

def test_lab_report_api(image_path, api_url="http://localhost:8000/get-lab-tests"):
    """
    Test the lab report API with a local image
    
    Args:
        image_path: Path to the image file
        api_url: URL of the API endpoint
        
    Returns:
        Response from the API
    """
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found")
        return None
    
    # Open the file in binary mode
    with open(image_path, "rb") as file:
        # Create a multipart form request
        files = {"file": (os.path.basename(image_path), file, "image/jpeg")}
        
        print(f"Sending request to {api_url} with file {image_path}")
        try:
            response = requests.post(api_url, files=files)
            
            # Check if response is successful
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"Exception occurred: {e}")
            return None

if __name__ == "__main__":
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_lab_report_api.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = test_lab_report_api(image_path)
    
    if result:
        print("\nAPI Response:")
        print("=" * 50)
        print(f"Success: {result.get('is_success', False)}")
        
        if result.get('is_success', False):
            print("\nExtracted Lab Tests:")
            for idx, test in enumerate(result.get('data', []), 1):
                print(f"\nTest #{idx}:")
                print(f"  Name: {test.get('test_name', 'N/A')}")
                print(f"  Value: {test.get('test_value', 'N/A')}")
                print(f"  Reference Range: {test.get('bio_reference_range', 'N/A')}")
                print(f"  Out of Range: {'Yes' if test.get('lab_test_out_of_range', False) else 'No'}")
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")
    else:
        print("No response received from API") 