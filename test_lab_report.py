import requests
import argparse
import json
import os

def test_get_lab_tests(image_path):
    """
    Test the /get-lab-tests endpoint with a lab report image
    
    Args:
        image_path: Path to the lab report image
    """
    url = "http://localhost:8000/get-lab-tests"
    
    # Prepare file for upload
    files = {"file": (os.path.basename(image_path), open(image_path, "rb"), "image/jpeg")}
    
    # Send request
    response = requests.post(url, files=files)
    
    # Print response
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))

def test_rag_lab_report(image_path, query):
    """
    Test the /rag-lab-report endpoint with a lab report image and query
    
    Args:
        image_path: Path to the lab report image
        query: Query string
    """
    url = "http://localhost:8000/rag-lab-report"
    
    # Prepare file for upload and query parameter
    files = {"file": (os.path.basename(image_path), open(image_path, "rb"), "image/jpeg")}
    data = {"query": query}
    
    # Send request
    response = requests.post(url, files=files, data=data)
    
    # Print response
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test lab report processing API")
    parser.add_argument("--image", required=True, help="Path to lab report image")
    parser.add_argument("--query", help="Query string for RAG-based analysis")
    
    args = parser.parse_args()
    
    # Test /get-lab-tests endpoint
    print("Testing /get-lab-tests endpoint...")
    test_get_lab_tests(args.image)
    
    # If query is provided, test /rag-lab-report endpoint
    if args.query:
        print("\nTesting /rag-lab-report endpoint...")
        test_rag_lab_report(args.image, args.query) 