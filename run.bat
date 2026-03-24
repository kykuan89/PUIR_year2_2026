@echo off
setlocal

REM --- locate installed launcher directory ---
set BASE=%~dp0

REM --- use bundled python (recommended) ---
set PY=%BASE%python\python.exe

REM --- choose a writable venv location ---
set VENV=%LOCALAPPDATA%\YourAppName\.venv

REM --- app location ---
set APP=%BASE%app\app.py
set REQ=%BASE%app\requirements.txt
set PORT=8501

if not exist "%PY%" (
  echo [ERROR] Bundled python not found: %PY%
  pause
  exit /b 1
)

if not exist "%APP%" (
  echo [ERROR] app.py not found: %APP%
  pause
  exit /b 1
)

if not exist "%VENV%\Scripts\python.exe" (
  echo [INFO] Creating venv at %VENV%
  "%PY%" -m venv "%VENV%"
  if errorlevel 1 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
  )
)

echo [INFO] Installing requirements...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
if exist "%REQ%" (
  "%VENV%\Scripts\python.exe" -m pip install -r "%REQ%"
) else (
  "%VENV%\Scripts\python.exe" -m pip install streamlit openpyxl pandas plotly
)

echo [INFO] Starting Streamlit on http://localhost:%PORT%
start http://localhost:%PORT%
"%VENV%\Scripts\python.exe" -m streamlit run "%APP%" --server.port %PORT% --server.headless true

pause
endlocal