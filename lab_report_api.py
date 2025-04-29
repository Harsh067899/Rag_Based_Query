from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from lab_report_processor_final import process_lab_report

app = FastAPI(title="Lab Report Analysis API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    return {
        "message": "Lab Report Analysis API",
        "endpoints": {
            "POST /get-lab-tests": "Upload lab report image to extract test data"
        }
    }

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    try:
        # Check if file is an image
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"is_success": False, "error": "File must be an image"}
            )
        
        # Read file content
        image_bytes = await file.read()
        
        # Process the image using lab report processor
        result = process_lab_report(image_bytes)
        
        return result
    except Exception as e:
        return {"is_success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("lab_report_api:app", host="0.0.0.0", port=8000, reload=True) 