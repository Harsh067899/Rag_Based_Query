@echo off
echo Starting Lab Report Analysis API at port 5000...
echo.
echo Server will be available at http://127.0.0.1:5000
echo API docs will be at http://127.0.0.1:5000/docs
echo.
python -m uvicorn single_image_api:app --host 0.0.0.0 --port 5000 --reload
pause 