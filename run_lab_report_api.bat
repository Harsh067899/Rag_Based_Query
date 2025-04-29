@echo off
echo Lab Report API Server
echo =====================
echo.
echo Starting FastAPI server...
start cmd /k "python -m uvicorn lab_report_api_final:app --reload"
echo.
echo API server running at http://localhost:8000
echo.
echo Endpoints:
echo  - GET /: API information
echo  - POST /get-lab-tests: Upload an image to extract lab test data
echo.
echo Press any key to exit...
pause > nul 