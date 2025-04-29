import argparse
import requests
import json
import os
from urllib.parse import urljoin
import time

def test_api_call(image_path, api_endpoint="http://localhost:8000", query=None):
    """
    Test the lab report processing API with a single image
    
    Args:
        image_path: Path to the lab report image
        api_endpoint: Base URL of the API
        query: Optional query string for RAG analysis
    
    Returns:
        API response as JSON
    """
    # Choose the appropriate endpoint
    if query:
        endpoint = urljoin(api_endpoint, "/rag-lab-report")
        print(f"Testing /rag-lab-report endpoint with query: '{query}'")
    else:
        endpoint = urljoin(api_endpoint, "/get-lab-tests")
        print(f"Testing /get-lab-tests endpoint (no query)")
    
    # Prepare file for upload
    filename = os.path.basename(image_path)
    files = {"file": (filename, open(image_path, "rb"), "image/jpeg")}
    
    # Add query if provided
    data = {}
    if query:
        data["query"] = query
    
    # Measure response time
    start_time = time.time()
    
    # Send the request
    try:
        response = requests.post(endpoint, files=files, data=data)
        response_time = time.time() - start_time
        
        # Check response status
        if response.status_code == 200:
            result = response.json()
            
            # Create output directory for results
            output_dir = "api_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save raw API response
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_api_response.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            
            # Print summary
            print(f"API Response received in {response_time:.2f} seconds")
            print(f"Status: {'Success' if result.get('is_success') else 'Failed'}")
            
            if result.get('is_success'):
                tests_found = len(result.get('data', []))
                abnormal_count = sum(1 for test in result.get('data', []) if test.get('lab_test_out_of_range'))
                
                print(f"Tests found: {tests_found}")
                print(f"Abnormal tests: {abnormal_count}")
                
                # Show a few examples
                if tests_found > 0:
                    print("\nExamples of detected tests:")
                    for i, test in enumerate(result.get('data', [])[:3]):  # Show up to 3 examples
                        print(f"{i+1}. {test['test_name']}: {test['test_value']} " + 
                            f"(Reference: {test['bio_reference_range']}) " +
                            ("OUT OF RANGE" if test['lab_test_out_of_range'] else "Normal"))
                    
                    if tests_found > 3:
                        print(f"... and {tests_found - 3} more")
                
                # Show RAG response if available
                if 'rag_response' in result:
                    print("\nRAG Analysis:")
                    print(result['rag_response'])
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            print(f"\nFull results saved to {output_file}")
            return result
        else:
            print(f"API call failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error making API request: {str(e)}")
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the lab report processing API with a single image")
    parser.add_argument("--image", required=True, help="Path to the lab report image")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="API endpoint")
    parser.add_argument("--query", help="Optional query for RAG analysis")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found")
    else:
        test_api_call(args.image, args.endpoint, args.query) 