@echo off

rem Author / Auteur    KMS - Martin Dubois, ing.
rem Product / Produit  OpenNet
rem File / Fichier     Build.cmd

echo Executing  Build.cmd  ...

rem ===== Initialization / Initialisation ===================================

set KMS_VERSION="C:\Software\KmsTools\KmsVersion.exe"
set MSBUILD="C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"
set OPTIONS="OpenNet.sln" /target:rebuild

rem ===== Verification / Verification =======================================

if not exist %KMS_VERSION% (
    echo FATAL ERROR  %KMS_VERSION%  does not exist
	echo Install KmsTools version 2.4.0 or higher
	pause
	exit /B 1
)

if not exist %MSBUILD% (
	echo ERREUR FATAL  %MSBUILD%  does not exist
    echo Install Visual Studio 2015
	pause
	exit /B 2
)

rem ===== Execution / Execution =============================================

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

%KMS_VERSION% -S Common\Version.h Export.cmd
if ERRORLEVEL 1 (
    echo ERROR  %KMS_VERSION% -S Common\Version.h Export.cmd  failed - %ERRORLEVEL%
	pause
	exit /B 7
)

rem ===== End / Fin =========================================================

echo OK
