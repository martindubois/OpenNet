@echo off

rem Author   KMS - Martin Dubois, ing.
rem Product  KmsBase
rem File     RunDoxygen.cmd

echo Executing  RunDoxygen.cmd  ...

rem ===== Initialisation ====================================================

set DOXYFILE_EN="DoxyFile_en.txt"
set DOXYFILE_FR="DoxyFile_fr.txt"
set DOXYGEN="C:\Program Files\doxygen\bin\doxygen.exe"

rem ===== Verification ======================================================

if not exist %DOXYGEN% (
    echo FATAL ERROR  %DOXYGEN%  does not exist
	echo Install doxygen
	pause
	exit /B 1
)

rem ===== Execution =========================================================

%DOXYGEN% %DOXYFILE_EN%
if ERRORLEVEL 1 (
    echo ERROR  %DOXYGEN% %DOXYFILE_EN%  failed - %ERRORLEVEL%
	pause
	exit /B 2
)

%DOXYGEN% %DOXYFILE_FR%
if ERRORLEVEL 1 (
    echo ERROR  %DOXYGEN% %DOXYFILE_FR% failed - %ERRORLEVEL%
	pause
	exit /B 1
)

rem ===== End ===============================================================

echo OK
