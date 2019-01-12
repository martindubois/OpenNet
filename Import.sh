#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       Import.sh
# Usage      ./Import.sh

echo Excuting  Import.sh  ...

# ===== Dependencies =========================================================

KMS_BASE=~/Export/KmsBase/3.0.4_KI_Linux

# ===== Constants ============================================================

DST_FOLDER=$PWD/Import

# ===== Verification =========================================================

if [ ! -d $KMS_BASE ]
then
    echo FATAL ERROR  $KMS_BASE  does not exist
    exit 1
fi

# ===== Execution ============================================================

if [ ! -d $DST_FOLDER ]
then
    mkdir $DST_FOLDER
fi

cd $KMS_BASE

./Import.sh $DST_FOLDER

if [ 0 -ne $? ]
then
    echo ERROR  ./Import.sh $DST_FOLDER  failed - $?
    exit 2
fi

# ===== End ==================================================================

echo OK
