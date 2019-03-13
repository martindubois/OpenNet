#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       Scripts/CreateTestTree.sh
# Usage      ./CreateTestTree.sh

echo Excuting  CreateTestTree.sh  ...

# ==== Execution ============================================================

sudo insmod OpenNet/ONK_Pro1000/ONK_Pro1000.ko

if [ 0 != $? ] ; then
    echo ERROR  sudo insmod OpenNet/ONK_Pro1000/ONK_Pro1000.ko  failed - $?
    exit 1
fi

cd OpenNet/Binaries

if [ "" != "$1" ] ; then

    ./ONK_Test 0

    if [ 0 != $? ] ; then
        echo ERROR  ./ONK_Test 0  failed - $?
        exit 2
    fi

    ./ONK_Test 1

    if [ 0 != $? ] ; then
        echo ERROR  ./ONK_Test 1  failed - $?
        exit 3
    fi

    ./OpenNet_Test group 2

    if [ 0 != $? ] ; then
       echo ERROR  ./OpenNet_Test group 2  failed - $?
       exit 4
    fi

    ./OpenNet_Tool

fi

# ===== End =================================================================

echo OK
