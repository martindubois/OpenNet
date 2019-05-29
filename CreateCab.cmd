@echo off

rem Author     KMS - Martin Dubois, ing.
rem Copyright  (C) 2019 KMS. All rights reserved
rem Product    OpenNet
rem File       CreateCab.cmd
rem Usage      .\CreateCab.cmd

echo Executing  CreateCab.cmd  ...

rem ===== Initialisation ====================================================

set CERT_SHA=2D45F19469373612C5E626A8FCD4450759792859

set OPENNET_CAB=disk1\OpenNet.cab

set OPENNET_DDF=OpenNet.ddf

set SIGNTOOL_EXE="C:\Program Files (x86)\Windows Kits\10\Tools\bin\i386\signtool.exe"

rem ===== Verification ======================================================

if not exist %SIGNTOOL_EXE% (
	echo FATAL ERROR  %SIGNTOOL_EXE%  does not exist
	echo Install the WDK
	pause
	exit /B 1
)

rem ===== Execution =========================================================

makecab -f %OPENNET_DDF%
if ERRORLEVEL 1 (
	echo ERROR  makecab -f %OPENNET_DDF%  failed - %ERRORLEVEL%
	pause
	exit /B 2
)

%SIGNTOOL_EXE% sign /sha1 %CERT_SHA% %OPENNET_CAB%
if ERRORLEVEL 1 (
	echo ERROR  %SIGNTOOL_EXE% sign /sha1 %CERT_SHA% %OPENNET_CAB%  failed - %ERRORLEVEL%
	pause
	exit /B 3
)

rem ===== End ===============================================================

echo OK
