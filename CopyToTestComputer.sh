#!/bin/sh

# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       CopyToTestComputer.sh
# Usage      ./CopyToTestComputer.sh

echo Excuting  CopyToTestComputer.sh  ...

# ===== Configuration =======================================================

TEST_COMPUTER=192.168.0.197

# ===== Initialisation ======================================================

DST_FOLDER=~/OpenNet

# ===== Execution ===========================================================

scp Binaries/ONK_Test $TEST_COMPUTER:$DST_FOLDER/Binaries/ONK_Test

RESULT=0

if [ 0 != $? ] ; then
    echo ERROR  scp Binaries/ONK_Test $TEST_COMPUTER:$DST_FOLDER/Binaries/ONK_Test  failed - $?
    RESULT=1
fi

scp Binaries/OpenNet.so $TEST_COMPUTER:$DST_FOLDER/Binaries/OpenNet.so

if [ 0 != $? ] ; then
    echo ERROR  scp Binaries/OpenNet.so $TEST_COMPUTER:$DST_FOLDER/Binaries/OpenNet.so  failed - $?
    RESULT=2
fi

scp Binaries/OpenNet_Test $TEST_COMPUTER:$DST_FOLDER/Binaries/OpenNet_Test

if [ 0 != $? ] ; then
    echo ERROR  scp Binaries/OpenNet_Test $TEST_COMPUTER:$DST_FOLDER/Binaries/OpenNet_Test  failed - $?
    RESULT=3
fi

scp Binaries/OpenNet_Tool $TEST_COMPUTER:$DST_FOLDER/Binaries/OpenNet_Tool

if [ 0 != $? ] ; then
    echo ERROR  scp Binaries/OpenNet_Tool $TEST_COMPUTER:$DST_FOLDER/Binaries/OpenNet_Tool  failed - $?
    RESULT=4
fi

scp Includes/OpenNetK/Kernel.h $TEST_COMPUTER:$DST_FOLDER/Includes/OpenNetK

if [ 0 != $? ] ; then
    echo ERROR  scp Includes/OpenNetK/Kernel.h $TEST_COMPUTER:$DST_FOLDER/Includes/OpenNetK  failed - $?
    RESULT=5
fi

scp Includes/OpenNetK/Kernel_CUDA.h $TEST_COMPUTER:$DST_FOLDER/Includes/OpenNetK

if [ 0 != $? ] ; then
    echo ERROR  scp Includes/OpenNetK/Kernel_CUDA.h $TEST_COMPUTER:$DST_FOLDER/Includes/OpenNetK  failed - $?
    RESULT=6
fi

scp Includes/OpenNetK/Types.h $TEST_COMPUTER:$DST_FOLDER/Includes/OpenNetK

if [ 0 != $? ] ; then
    echo ERROR  scp Includes/OpenNetK/Types.h $TEST_COMPUTER:$DST_FOLDER/Includes/OpenNetK  failed - $?
    RESULT=7
fi

scp ONK_Pro1000/ONK_Pro1000.ko $TEST_COMPUTER:$DST_FOLDER/ONK_Pro1000

if [ 0 != $? ] ; then
    echo ERROR  scp ONK_Pro1000/ONK_Pro1000.ko $TEST_COMPUTER:$DST_FOLDER/ONK_Pro1000  failed - $?
    RESULT=8
fi

scp Scripts/OpenNet_Tool/A00RRU_K620_18.04.txt $TEST_COMPUTER:$DST_FOLDER/Scripts/OpenNet_Tool

if [ 0 != $? ] ; then
    echo ERROR  scp Scripts/OpenNet_Tool/A00RRU_K620_18.04.txt $TEST_COMPUTER:$DST_FOLDER/Scripts/OpenNet_Tool  failed - $?
    RESULT=9
fi

# ===== End =================================================================

if [ 0 = $RESULT ] ; then
    echo OK
fi

exit $RESULT
