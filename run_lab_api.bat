@echo off
echo Starting Simple Lab Report API Server...
echo.
echo Server will be available at http://127.0.0.1:8000
echo API docs will be at http://127.0.0.1:8000/docs
echo.
python -m uvicorn simple_lab_api:app --reload
pause 