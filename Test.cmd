@echo off

rem Author   KMS - Martin Dubois, ing.
rem Product  OpenNet
rem File     Test.cmd

echo Executing  Test.cmd  ...

rem ===== Initialization ====================================================

set CHK_INF_BAT="C:\Program Files (x86)\Windows Kits\10\Tools\x86\ChkInf\chkinf.bat"
set CHK_INF_OUT="htm"
set DEBUG=x64\Debug
set RELEASE=x64\Release

rem ===== Verification ======================================================

if not exist %CHK_INF_BAT% (
    echo FATAL ERROR  %INF_TEST_EXE%  does not exist
    echo Install Windows Kit 10
    pause
    exit /B 1
)

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

if exist %CHK_INF_OUT% rmdir /S /Q %CHK_INF_OUT%

call %CHK_INF_BAT% %RELEASE%\ONK_NDIS\ONK_NDIS.inf
if ERRORLEVEL 1 (
    echo ERROR  call %CHK_INF_BAT% %RELEASE%\ONK_NDIS\ONK_NDIS.inf  failed - %ERRORLEVEL%
    pause
    exit /B 4
)

if exist %CHK_INF_OUT% rmdir /S /Q %CHK_INF_OUT%

call %CHK_INF_BAT% %RELEASE%\ONK_Pro1000\ONK_Pro1000.inf
if ERRORLEVEL 1 (
    echo ERROR  call %CHK_INF_BAT% %RELEASE%\ONK_Pro1000\ONK_Pro1000.inf  failed - %ERRORLEVEL%
    pause
    exit /B 5
)

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
