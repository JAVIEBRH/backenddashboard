@echo off
echo ======================================
echo   INICIANDO BACKEND - AGUAS ANCUD
echo ======================================

:start
echo.
echo [%time%] Iniciando servidor...
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
echo.
echo [%time%] Servidor se cayo. Reiniciando en 3 segundos...
timeout /t 3 /nobreak >nul
goto start
