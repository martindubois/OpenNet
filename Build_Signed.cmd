@echo off

rem Author     KMS - Martin Dubois, ing.
rem Copyright  (C) 2019 KMS. All rights reserved.
rem Product    OpenNet
rem File       Build_End.cmd
rem Usage      .\Build_Signed.cmd

echo Executing  Build_Signed.cmd  ...

rem ===== Initialization ====================================================

set INNO_COMPIL32="C:\Program Files (x86)\Inno Setup 5\Compil32.exe"

set KMS_VERSION="C:\Software\KmsTools\KmsVersion.exe"

rem ===== Verification ======================================================

if not exist %INNO_COMPIL32% (
	echo FATAL ERROR  %INNO_COMPIL32%  does not exist
	echo Install Inno 5.6.1
	pause
	exit /B 1
)

if not exist %KMS_VERSION% (
    echo FATAL ERROR  %KMS_VERSION%  does not exist
	echo Install KmsTools version 2.4.0 or higher
	pause
	exit /B 2
)

rem ===== Execution =========================================================

%KMS_VERSION% Common\Version.h Export_Signed.cmd.txt OpenNet_Signed.iss
if ERRORLEVEL 1 (
	echo ERROR  %KMS_VERSION% Common\Version.h Export_Signed.cmd.txt OpenNet_Signed.iss  failed - %ERRORLEVEL%
	pause
	exit /B 3
)

%INNO_COMPIL32% /cc OpenNet_Signed.iss
if ERRORLEVEL 1 (
	echo ERROR  %INNO_COMPIL32% /cc OpenNet_Signed.iss  failed - %ERRORLEVEL%
	pause
	exit /B 4
)

%KMS_VERSION% -S Common\Version.h Export_Signed.cmd
if ERRORLEVEL 1 (
    echo ERROR  %KMS_VERSION% -S Common\Version.h Export_Signed.cmd  failed - %ERRORLEVEL%
	pause
	exit /B 5
)

rem ===== End ===============================================================

echo OK
