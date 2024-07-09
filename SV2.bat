@echo off
start /min python speechV2.py
timeout /t 1 /nobreak > nul
exit