#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       ONK_Pro1000/Test.sh
# Usage      ./Test.sh

echo Executing  ONK_Pro1000/Test.sh  ...

# ===== Initialisation ======================================================

FOLDER=$(pwd)

VERSION=$(uname -r)

# ===== Execution ===========================================================

sudo depmod -e -F /boot/System.map-$VERSION $VERSION $FOLDER/ONK_Pro1000.ko

if [ 0 != $? ]
then
    echo  sudo depmod -e -F /boot/System.map-$VERSION $VERSION $FOLDER/ONK_Pro1000.ko  failed - $?
    exit 1
fi

# ===== End =================================================================

echo OK
