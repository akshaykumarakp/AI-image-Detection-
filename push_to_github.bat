@echo off
title Push to GitHub
cd /d "%~dp0"
echo ========================================================
echo   Pushing code to https://github.com/akshaykumarakp/AI-image-Detection-
echo ========================================================
echo.
git push -u origin main
echo.
echo ========================================================
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS! Your code has been pushed to GitHub.
) else (
    echo An error occurred during push.
)
echo ========================================================
pause
