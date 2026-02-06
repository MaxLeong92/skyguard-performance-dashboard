@echo off
TITLE SkyGuard Enterprise Command
ECHO Starting SkyGuard Dashboard...
ECHO Please wait while the server launches...
cd /d "%~dp0"
py -m streamlit run "%~dp0app_fleet.py"
pause
