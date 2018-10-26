@echo off

rem Author   KMS - Martin Dubois, ing.
rem Product  OpenNet
rem File     Scripts/Package.cmd
rem Usage    Package.cmd {Version_Type}

echo Executing  Package.cmd %1  ...

rem ===== Configuration =====================================================

set KMS_ZIP_EXE="C:\Software\KmsTools\KmsZip.exe"

rem ===== Verification ======================================================

if ""=="%1" (
    echo USER ERROR  Invalid command line
	echo Usage  Packet.cmd {Version_Type}
	pause
	exit /B 1
)

if not exist %KMS_ZIP_EXE% (
	echo FATAL ERROR  %KMS_ZIP_EXE%  does not exist
	echo Install KmsTools 2.4.0 or higher
	pause
	exit /B 2
)

rem ===== Execution =========================================================

%KMS_ZIP_EXE% OpenNet_Samples_%1.zip Package_Samples.txt .
if ERRORLEVEL 1 (
	echo ERROR  %KMS_ZIP_EXE% OpenNet_Samples_%1.zip Package_Samples.txt .  failed - %ERRORLEVEL%
	pause
	exit /B 3
)

%KMS_ZIP_EXE% OpenNet_SDK_%1.zip Package_SDK.txt .
if ERRORLEVEL 1 (
	echo ERROR  %KMS_ZIP_EXE% OpenNet_SDK_%1.zip Package_SDK.txt .  failed - %ERRORLEVEL%
	pause
	exit /B 4
)

%KMS_ZIP_EXE% OpenNet_SDK_DDK_%1.zip Package_SDK_DDK.txt .
if ERRORLEVEL 1 (
	echo ERROR  %KMS_ZIP_EXE% OpenNet_SDK_DDK_%1.zip Package_SDK_DDK.txt .  failed - %ERRORLEVEL%
	pause
	exit /B 5
)

%KMS_ZIP_EXE% OpenNet_SDK_Headers_%1.zip Package_SDK_Headers.txt .
if ERRORLEVEL 1 (
	echo ERROR  %KMS_ZIP_EXE% OpenNet_SDK_Headers_%1.zip Package_SDK_Headers.txt .  failed - %ERRORLEVEL%
	pause
	exit /B 6
)

rem ===== End ===============================================================

echo OK
