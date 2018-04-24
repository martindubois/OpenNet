@echo off

rem Author / Auteur	   KMS - Martin Dubois, ing.
rem Product / Produit  OpenNet
rem File / Fichier     Export.cmd
rem Usage / Usage      Export.cmd {Ma.Mi.Bu} {Type}

echo  Executing  Export.cmd %1 %2  ...

rem ===== Initialization / Initialisation ===================================

set EXPORT_CMD_TXT="Export.cmd.txt"
set KMS_COPY="C:\Software\KmsTools\KmsCopy.exe"

if ""=="%2" (
	set DST="K:\Export\OpenNet\%1"
) else (
	set DST="K:\Export\OpenNet\%1_%2"
)

rem ===== Verification / Verification =======================================

if ""=="%1" (
    echo USER ERROR  Invalid command line
    echo Usage  Export.cmd {Ma.Mi.Bu} [Internal|RC|Test]
    pause
    exit /B 1
)

if exist %DST% (
    echo  USER ERROR  %DST%  already exist
    pause
    exit /B 2
)

if not exist %KMS_COPY% (
    echo FATAL ERROR  %KMS_COPY%  does not exist
    echo Install KmsTools version 2.4.0 or higher
    pause
	exit /B 3
)

rem ===== Execution / Execution =============================================

%KMS_COPY% . %DST% %EXPORT_CMD_TXT%
if ERRORLEVEL 1 (
    echo ERROR  %KMS_COPY% . %DST% %EXPORT_CMD_TXT%  failed - %ERRORLEVEL%
	pause
	exit /B 4
)

rem ===== End / Fin =========================================================

echo OK
