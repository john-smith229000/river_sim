@echo off

rem Runs the refactored river simulation
rem The --config argument points to the parameters file.
C:\Users\vince\AppData\Local\Programs\Python\Python312\python.exe "%~dp0main.py" --config "%~dp0improved_config.yaml" %*

pause