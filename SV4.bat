@echo off
start /min python speechV4.py
timeout /t 1 /nobreak > nul
exit