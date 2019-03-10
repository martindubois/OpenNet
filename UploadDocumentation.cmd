@echo off

rem Author     KMS - Martin Dubois, ing.
rem Copyright  (C) 2019 KMS. All rights reserved.
rem Product    OpenNet
rem File       UploadDocumentation.cmd

echo  Executing  OploadDocumentation.cmd  ...

rem  ===== Execution ========================================================

ftp.exe -i -s:UploadDocumentation.cmd.txt ftp.kms-quebec.com
if ERRORLEVEL 1 (
    echo  ERROR  ftp -s:UploadDocumentation.txt ftp.kms-quebec.com  failed - %ERRORLEVEL%
    pause
    exit /B 1
)

rem  ===== Fin ==============================================================

echo  OK
