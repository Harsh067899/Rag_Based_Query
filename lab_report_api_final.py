from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict

# Import our lab report processor
from lab_report_processor_final import process_lab_report

app = FastAPI(title="Lab Report Processing API", 
              description="API for extracting lab test data from medical reports")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """
    Root endpoint with API info
    """
    return {
        "name": "Lab Report Processing API",
        "description": "API for extracting lab test data from medical reports",
        "endpoints": [
            {"path": "/get-lab-tests", "method": "POST", "description": "Extract lab tests from an image"}
        ]
    }

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    """
    Extract lab tests from an image
    
    Args:
        file: Image file containing lab report
        
    Returns:
        JSON response with extracted lab test data in the format:
        {
            "is_success": true,
            "data": [
                {
                    "test_name": "BILIRUBIN TOTAL",
                    "test_value": 9.42,
                    "bio_reference_range": "0.30- 1.20",
                    "lab_test_out_of_range": true
                },
                ...
            ]
        }
    """
    try:
        # Check if file is an image
        content_type = file.content_type
        if not content_type or not content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"is_success": False, "error": "File must be an image"}
            )
        
        # Read file content
        image_bytes = await file.read()
        
        # Process with our lab report processor
        result = process_lab_report(image_bytes)
        
        # Return the result
        if result.get("is_success", False):
            return result
        else:
            return JSONResponse(
                status_code=500,
                content={"is_success": False, "error": result.get("error", "Failed to process lab report")}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"is_success": False, "error": str(e)}
        )

if __name__ == "__main__":
    print("Lab Report Processing API starting...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run("lab_report_api_final:app", host="0.0.0.0", port=8000, reload=True) 