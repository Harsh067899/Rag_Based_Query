@echo off
echo Lab Report API Test Client
echo =========================
echo.

if "%~1"=="" (
    echo Usage: test_lab_api.bat [image_file]
    echo Example: test_lab_api.bat lab_report.jpg
    exit /b 1
)

if not exist "%~1" (
    echo Error: Image file "%~1" not found.
    exit /b 1
)

echo Testing API with image: %~1
echo.
python test_lab_report_api.py "%~1"
echo.
echo Press any key to exit...
pause > nul 