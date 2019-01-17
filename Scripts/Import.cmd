@echo off

rem Author     KMS - Martin Dubois, ing.
rem Copyright  (C) 2019 KMS. All right reserved
rem Product    OpenNet
rem File       Import.cmd
rem Usage      Import.cmd {Destination}

echo Executing  Import.cmd %1  ...

rem ===== Initialisation ====================================================

set KMS_COPY_EXE="C:\Software\KmsTools\KmsCopy.exe"

rem ===== Verification ======================================================

if not exist %1 (
    echo USER ERROR  %1  does not exixt
    pause
    exit /B 1
)

if not exist %KMS_COPY_EXE% (
    echo FATAL ERROR  %KMS_COPY_EXE%  does not exist
    echo Install KmsTool 2.4.0 or higher
    pause
    exit /B 2
)

rem ===== Execution =========================================================

%KMS_COPY_EXE% . %1 Import.cmd.txt
if ERRORLEVEL 1 (
    echo ERROR  %KMS_COPY_EXE% . %1 Import.cmd.txt  failed - %ERRORLEVEL
    pause
    exit /B 3
)

rem ===== End ===============================================================

echo OK
