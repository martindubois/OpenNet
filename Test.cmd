@echo off

rem Author / Auteur    KMS - Martin Dubois, ing.
rem Product / Produit  OpenNet
rem File / Fichier     Test.cmd

echo Executing  Test.cmd  ...

rem ===== Initialization / Initialisation ===================================

set CHK_INF_BAT="C:\Program Files (x86)\Windows Kits\10\Tools\x86\ChkInf\chkinf.bat"
set CHK_INF_OUT="htm"

rem ===== Verification / Verification =======================================

if not exist %CHK_INF_BAT% (
    echo FATAL ERROR  %INF_TEST_EXE%  does not exist
    echo Install Windows Kit 10
    pause
    exit /B 1
)

rem ===== Execution / Execution =============================================

if exist %CHK_INF_OUT% rmdir /S /Q %CHK_INF_OUT%

call %CHK_INF_BAT% x64\Release\ONK_NDIS\ONK_NDIS.inf
if ERRORLEVEL 1 (
    echo ERROR  call %CHK_INF_BAT% ONK_NDIS\ONK_NDIS.inf  failed - %ERRORLEVEL%
    pause
    exit /B 2
)

rem ===== End / Fin =========================================================

echo OK
