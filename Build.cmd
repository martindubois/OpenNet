@echo off

rem Author     KMS - Martin Dubois, ing.
rem Copyright  (C) 2018-2019 KMS. All rights reserved.
rem Product    OpenNet
rem File       Build.cmd
rem Usage      .\Build.cmd

rem CODE REVIEW  2019-10-18  KMS - Martin Dubois, ing.

echo Executing  Build.cmd  ...

rem ===== Initialization ====================================================

set CERT_SHA=2D45F19469373612C5E626A8FCD4450759792859
set INNO_COMPIL32="C:\Program Files (x86)\Inno Setup 5\Compil32.exe"
set KMS_VERSION="C:\Software\KmsTools\KmsVersion.exe"
set MSBUILD="C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe"
set OPTIONS="OpenNet.sln" /target:rebuild
set SIGNTOOL_EXE="C:\Program Files (x86)\Windows Kits\10\Tools\bin\i386\signtool.exe"

rem ===== Verification ======================================================

if not exist %INNO_COMPIL32% (
	echo FATAL ERROR  %INNO_COMPIL32%  does not exist
	echo Install Inno 5.6.1
	pause
	exit /B 10
)

if not exist %KMS_VERSION% (
    echo FATAL ERROR  %KMS_VERSION%  does not exist
	echo Install KmsTools version 2.4.0 or higher
	pause
	exit /B 20
)

if not exist %MSBUILD% (
	echo ERREUR FATAL  %MSBUILD%  does not exist
    echo Install Visual Studio 2017 Professional
	pause
	exit /B 30
)

if not exist %SIGNTOOL_EXE% (
	echo FATAL ERROR  %SIGNTOOL_EXE%  does not exist
	echo Install the WDK
	pause
	exit /B 40
)

rem ===== Execution =========================================================

%MSBUILD% %OPTIONS% /property:Configuration=Debug /property:Platform=x64
if ERRORLEVEL 1 (
	echo ERROR  %MSBUILD% %OPTIONS% /property:Configuration=Debug /property:Platform=x64  failed - %ERRORLEVEL%
	pause
	exit /B 50
)

%MSBUILD% %OPTIONS% /property:Configuration=Release /property:Platform=x64
if ERRORLEVEL 1 (
	echo ERROR  %MSBUILD% %OPTIONS% /property:Configuration=Release /property:Platform=x64  failed - %ERRORLEVEL%
	pause
	exit /B 60
)

call Test.cmd
if ERRORLEVEL 1 (
    echo ERROR  Test.cmd  failed - %ERRORLEVEL%
	pause
	exit /B 70
)

call CreateCab.cmd
if ERRORLEVEL 1 (
	echo ERROR  CreateCab.cmd  failed - %ERRORLEVEL%
	pause
	exit /B 80
)

%KMS_VERSION% Common\Version.h Export.cmd.txt OpenNet.iss
if ERRORLEVEL 1 (
	echo ERROR  %KMS_VERSION% Common\Version.h Export.cmd.txt OpenNet.iss  failed - %ERRORLEVEL%
	pause
	exit /B 90
)

%INNO_COMPIL32% /cc OpenNet.iss
if ERRORLEVEL 1 (
	echo ERROR  %INNO_COMPIL32% /cc OpenNet.iss  failed - %ERRORLEVEL%
	pause
	exit /B 100
)

%SIGNTOOL_EXE% sign /fd sha256 /sha1 %CERT_SHA% /td sha256 /tr http://timestamp.digicert.com Installer/OpenNet_*.exe
if ERRORLEVEL 1 (
	echo ERROR  %SIGNTOOL_EXE% sign /fd sha256 /sha1 %CERT_SHA% /td sha256 /tr http://timestamp.digicert.com Installer/OpenNet_*.exe  failed - %ERRORLEVEL%
	pause
	exit /B 110
)

%KMS_VERSION% -S Common\Version.h Export.cmd
if ERRORLEVEL 1 (
    echo ERROR  %KMS_VERSION% -S Common\Version.h Export.cmd  failed - %ERRORLEVEL%
	pause
	exit /B 120
)

rem ===== End ===============================================================

echo OK
