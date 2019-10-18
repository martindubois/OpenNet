@echo off

rem Author     KMS - Martin Dubois, ing.
rem Copyright  (C) 2019 KMS. All rights reserved.
rem Product    OpenNet
rem File       Export_Signed.cmd
rem Usage      .\Export_Signed.cmd {Ma.Mi.Bu} {Type}

rem CODE REVIEW  2019-07-26  KMS - Martin Dubois, ing.

echo  Executing  Export_Signed.cmd %1 %2  ...

rem ===== Initialization ====================================================

set EXPORT_TXT="Export.txt"

set EXPORT_CMD_TXT="Export_Signed.cmd.txt"

set KMS_COPY="C:\Software\KmsTools\KmsCopy.exe"

if ""=="%2" (
	set DST="K:\Export\OpenNet\%1_Windows_Signed"
) else (
	set DST="K:\Export\OpenNet\%1_%2_Windows_Signed"
)

rem ===== Verification ======================================================

if ""=="%1" (
    echo USER ERROR  Invalid command line
    echo Usage  Export.cmd {Ma.Mi.Bu} [Type]
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

rem ===== Execution =========================================================

%KMS_COPY% . %DST% %EXPORT_TXT%
if ERRORLEVEL 1 (
    echo ERROR  %KMS_COPY% . %DST% %EXPORT_TXT%  failed - %ERRORLEVEL%
	pause
	exit /B 40
)

%KMS_COPY% . %DST% %EXPORT_CMD_TXT%
if ERRORLEVEL 1 (
    echo ERROR  %KMS_COPY% . %DST% %EXPORT_CMD_TXT%  failed - %ERRORLEVEL%
	pause
	exit /B 50
)

rem ===== End ===============================================================

echo OK
