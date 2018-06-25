@echo off

rem Author   KMS - Martin Dubois, ing.
rem Product  OpenNet
rem File     Test.cmd

echo Executing  Test.cmd  ...

rem ===== Initialization ====================================================

set DEBUG=x64\Debug
set RELEASE=x64\Release

rem ===== Verification ======================================================

if not exist "%DEBUG%" (
    echo FATAL ERROR  "%DEBUG%"  does not exist
    echo Compile the product
    pause
    exit /B 2
)

if not exist "%RELEASE%" (
    echo FATAL ERROR  "%RELEASE%"  does not exist
    echo Compile the product
    pause
    exit /B 3
)

rem ===== Execution =========================================================

%DEBUG%\OpenNet_Test.exe
if ERRORLEVEL 1 (
    echo ERROR  %DEBUG%\OpenNet_Test.exe  failed - %ERRORLEVEL%
    pause
    exit /B 6
)

%DEBUG%\ONK_Test.exe
if ERRORLEVEL 1 (
    echo ERROR  %DEBUG%\ONK_Test.exe  failed - %ERRORLEVEL%
    pause
    exit /B 7
)

%RELEASE%\OpenNet_Test.exe
if ERRORLEVEL 1 (
    echo ERROR  %RELEASE%\OpenNet_Test.exe  failed - %ERRORLEVEL%
    pause
    exit /B 8
)

%RELEASE%\ONK_Test.exe
if ERRORLEVEL 1 (
    echo ERROR  %RELEASE%\ONK_Test.exe  failed - %ERRORLEVEL%
    pause
    exit /B 8
)

rem ===== End ===============================================================

echo OK
