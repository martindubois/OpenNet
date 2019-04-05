@echo off

rem Author     KMS - Martin Dubois, ing.
rem Copyright  (C) 2018-2019 KMS. All rights reserved.
rem Product    OpenNet
rem File       Build.cmd
rem Usage      .\Build.cmd

echo Executing  Build.cmd  ...

rem ===== Initialization ====================================================

set INNO_COMPIL32="C:\Program Files (x86)\Inno Setup 5\Compil32.exe"
set KMS_VERSION="C:\Software\KmsTools\KmsVersion.exe"
set MSBUILD="C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe"
set OPTIONS="OpenNet.sln" /target:rebuild

rem ===== Verification ======================================================

if not exit %INNO_COMPIL32% (
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

if not exist %MSBUILD% (
	echo ERREUR FATAL  %MSBUILD%  does not exist
    echo Install Visual Studio 2017 Professional
	pause
	exit /B 3
)

rem ===== Execution =========================================================

%MSBUILD% %OPTIONS% /property:Configuration=Debug /property:Platform=x64
if ERRORLEVEL 1 (
	echo ERROR  %MSBUILD% %OPTIONS% /property:Configuration=Debug /property:Platform=x64  failed - %ERRORLEVEL%
	pause
	exit /B 4
)

%MSBUILD% %OPTIONS% /property:Configuration=Release /property:Platform=x64
if ERRORLEVEL 1 (
	echo ERROR  %MSBUILD% %OPTIONS% /property:Configuration=Release /property:Platform=x64  failed - %ERRORLEVEL%
	pause
	exit /B 5
)

call Test.cmd
if ERRORLEVEL 1 (
    echo ERROR  Test.cmd  failed - %ERRORLEVEL%
	pause
	exit /B 6
)

%KMS_VERSION% Common\Version.h Export.cmd.txt OpenNet.iss
if ERRORLEVEL 1 (
	echo ERROR  %KMS_VERSION% Common\Version.h Export.cmd.txt OpenNet.iss  failed - %ERRORLEVEL%
	pause
	exit /B 7
)

%INNO_COMPIL32% /cc OpenNet.iss
if ERRORLEVEL 1 (
	echo ERROR  %INNO_COMPIL32% /cc OpenNet.iss  failed - %ERRORLEVEL%
	pause
	exit /B 8
)

%KMS_VERSION% -S Common\Version.h Export.cmd
if ERRORLEVEL 1 (
    echo ERROR  %KMS_VERSION% -S Common\Version.h Export.cmd  failed - %ERRORLEVEL%
	pause
	exit /B 9
)

rem ===== End ===============================================================

echo OK
