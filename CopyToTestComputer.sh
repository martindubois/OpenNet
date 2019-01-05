#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       CopyToTestComputer.sh
# Usage      ./CopyToTestComputer.sh

echo Excuting  ImpCopyToTestComputer.sh  ...

# ===== Configuration =======================================================

TEST_COMPUTER=192.168.0.199

# ===== Execution ===========================================================

scp ONK_Pro1000/ONK_Pro1000.ko $TEST_COMPUTER:~/OpenNet/ONK_Pro1000/ONK_Pro1000.ko

if [ 0 != $? ] ; then
    echo ERROR  scp ONK_Pro1000/ONK_Pro1000.ko $TEST_COMPUTER:~/OpenNet/ONK_Pro1000/ONK_Pro1000.ko  failed - $?
    exit 1
fi

# ===== End =================================================================

echo OK
