from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import json
from typing import Dict

# Import the analyze_single_image function from your existing file
from test_single_image import analyze_single_image

app = FastAPI(title="Lab Report Analysis API")

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
    return {
        "message": "Lab Report Analysis API",
        "endpoints": {
            "POST /analyze": "Upload a lab report image for analysis"
        }
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Check if file is an image
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"is_success": False, "error": "File must be an image"}
            )
        
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            # Read file content and write to temp file
            content = await file.read()
            temp.write(content)
            temp_path = temp.name
        
        try:
            # Process the image using the existing analyze_single_image function
            # It returns the output directory and the results dictionary
            output_dir, result = analyze_single_image(temp_path, output_dir="temp_results")
            
            # Return the result as JSON
            return JSONResponse(content=result)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"is_success": False, "error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run("single_image_api:app", host="0.0.0.0", port=5000, reload=True) 