import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from dotenv import load_dotenv
from typing import Optional

from lab_report_processor import process_lab_report
from extract import extract_text_from_pdfs
from generate import generate_response
from preprocess import preprocess_text
from retrieve import create_vectorizer, retrieve

# Load environment variables for any non-Gemini configs
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Lab Report Processing API",
    description="API for processing lab reports and extracting lab test data using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Global variables to store vectorizer and processed text matrix
vectorizer = None
X = None
texts = []
processed_texts = []

# Root route to serve the frontend
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    """
    Extract lab test data from an image
    
    Args:
        file: Image file containing lab report
        
    Returns:
        JSON response with extracted lab test data
    """
    try:
        # Read file contents
        content = await file.read()
        
        # Process lab report image
        result = process_lab_report(content)
        
        # Return JSON response
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file for RAG processing
    
    Args:
        file: PDF file
        
    Returns:
        JSON response with success status
    """
    global vectorizer, X, texts, processed_texts
    
    try:
        # Check if file is a PDF
        if not file.filename.endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"is_success": False, "error": "File must be a PDF"}
            )
        
        # Save uploaded file to disk temporarily
        with open(file.filename, "wb") as f:
            f.write(await file.read())
        
        # Extract text from PDF
        pdf_texts = extract_text_from_pdfs([file.filename])
        texts.extend(pdf_texts)
        
        # Preprocess text
        new_processed_texts = preprocess_text(pdf_texts)
        processed_texts.extend(new_processed_texts)
        
        # Create vectorizer and transform texts
        vectorizer, X = create_vectorizer(processed_texts)
        
        # Clean up uploaded file
        os.remove(file.filename)
        
        return JSONResponse(
            content={
                "is_success": True,
                "message": "PDF uploaded and processed successfully"
            }
        )
    except Exception as e:
        # Clean up if file was saved
        if os.path.exists(file.filename):
            os.remove(file.filename)
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: str = Form(...)):
    """
    Query the documents using RAG
    
    Args:
        query: Query string
        
    Returns:
        JSON response with generated answer
    """
    global vectorizer, X, texts, processed_texts
    
    if not vectorizer or X is None or not texts:
        return JSONResponse(
            status_code=400,
            content={"is_success": False, "error": "No documents have been uploaded yet"}
        )
    
    try:
        # Retrieve relevant texts using enhanced retrieval
        top_indices = retrieve(query, X, vectorizer, top_k=5, use_enhanced=True, texts=texts)
        retrieved_texts = [texts[i] for i in top_indices]
        
        # Generate response
        response = generate_response(retrieved_texts, query)
        
        return JSONResponse(
            content={
                "is_success": True,
                "answer": response
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag-lab-report")
async def rag_lab_report(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None)
):
    """
    Process lab report image and query the extracted data using RAG
    
    Args:
        file: Image file containing lab report
        query: Optional query to process the extracted lab tests
        
    Returns:
        JSON response with processed lab test data and optional RAG response
    """
    global vectorizer, X, texts
    
    try:
        # Read file contents
        content = await file.read()
        
        # Process lab report image
        result = process_lab_report(content)
        
        # If we successfully extracted data, add it to our knowledge base
        if result.get("is_success", False) and result.get("data"):
            lab_text = ""
            for test in result.get("data", []):
                lab_text += f"Test: {test['test_name']}, Value: {test['test_value']}, "
                lab_text += f"Reference Range: {test['bio_reference_range']}, "
                lab_text += f"Out of Range: {test['lab_test_out_of_range']}\n"
            
            # Add the lab report text to our corpus
            if lab_text:
                texts.append(lab_text)
                processed_text = preprocess_text([lab_text])[0]
                processed_texts.append(processed_text)
                # Update vectorizer with new text
                if len(texts) > 1:  # Only if we have more than one document
                    vectorizer, X = create_vectorizer(processed_texts)
        
        # If query is provided and RAG system is initialized, generate a response
        if query and vectorizer and X is not None and len(texts) > 0:
            # Use enhanced retrieval
            context_indices = retrieve(query, X, vectorizer, top_k=5, use_enhanced=True, texts=texts)
            context_texts = [texts[i] for i in context_indices]
            
            # Generate response
            response = generate_response(context_texts, query)
            
            result["rag_response"] = response
        
        # Return JSON response
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ensure templates directory exists
os.makedirs("templates", exist_ok=True)

if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=True) 