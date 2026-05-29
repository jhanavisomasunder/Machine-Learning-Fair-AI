@echo off
title FastAPI Backend Server
cd /d "%~dp0\backend"
echo Starting FastAPI Backend Server with Python 3.11...
"C:\PROGRA~1\Python311\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
