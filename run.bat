@echo off

echo Starting server...
start /B python server.py
timeout /t 3 >nul

echo Running client...
python client.py

echo Done. You may need to manually stop the server if it runs indefinitely.
pause
